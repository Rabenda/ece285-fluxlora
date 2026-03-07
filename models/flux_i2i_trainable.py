import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
import inspect

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig, CLIPVisionModel
from peft import LoraConfig, get_peft_model


def _flux_unet_accepts_guidance(unet: FluxTransformer2DModel) -> bool:
    """
    檢查給定的 FluxTransformer2DModel 是否在 forward 簽名中接受 guidance 參數。
    這比依賴 diffusers 版本號更可靠，可同時相容舊版與 0.30+/0.36+。
    """
    try:
        sig = inspect.signature(unet.forward)
        return "guidance" in sig.parameters
    except Exception:
        # 若無法檢查簽名，保守地視為接受 guidance，以相容舊版行為。
        return True


class FluxI2ITrainable(nn.Module):
    """
    Full working version (drop-in replacement):
    - Fixes the "colored checkerboard / moire" collapse by using PixelUnshuffle/Shuffle for packing
      (instead of fragile view/permute ordering).
    - Keeps your proto + warmup + flatten target construction.
    - Ensures dtype consistency (bf16) before feeding FLUX to avoid RMSNorm dtype mismatch slow path.
    - Preview uses the same token-space x0 estimate and decodes through VAE.
    """

    def __init__(self, model_id="black-forest-labs/FLUX.1-schnell", proto_pool=32):
        super().__init__()
        print("[FluxI2ITrainable] Init FULL (pixel-unshuffle pack, bf16-safe, cartoonish target)...")

        # ---------- 0) counters ----------
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=True)

        # ---------- 1) 4-bit quant config ----------
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # ---------- 2) Load FLUX transformer ----------
        self.unet = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        self.unet.enable_gradient_checkpointing()
        # 檢查當前 unet 是否支援 guidance 參數（舊版支援，diffusers 0.30+ 的 schnell 通常不支援）
        self._accepts_guidance = _flux_unet_accepts_guidance(self.unet)

        # ---------- 3) CLIP vision ----------
        self.clip_vision = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to("cuda", dtype=torch.bfloat16)
        self.clip_vision.requires_grad_(False)

        # Projection layers
        self.vlm_proj = nn.Linear(1024, 4096).to("cuda", dtype=torch.bfloat16)
        self.pooled_proj = nn.Linear(1024, 768).to("cuda", dtype=torch.bfloat16)

        # ---------- 4) Load VAE ----------
        pipe = FluxPipeline.from_pretrained(
            model_id,
            transformer=None,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            torch_dtype=torch.bfloat16,
        )
        self.vae = pipe.vae.to("cuda", dtype=torch.float32)  # fp32 for stability
        self.vae.requires_grad_(False)
        self.scaling_factor = self.vae.config.scaling_factor

        # ---------- 5) Buffers ----------
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

        self.proto_pool = int(proto_pool)
        self.register_buffer("cartoon_proto_low", torch.zeros(1, 16, self.proto_pool, self.proto_pool))
        self.register_buffer("proto_initialized", torch.zeros((), dtype=torch.bool))

        self._bad_print_count = 0
        self._spike_print_count = 0
        self._last_spike_info = None

        # blur kernel (depthwise 3x3)
        k = torch.tensor([1., 2., 1.], dtype=torch.float32)
        k = (k[:, None] * k[None, :])
        k = k / k.sum()
        self.register_buffer("blur_kernel_3x3", k.view(1, 1, 3, 3))

    def setup_lora(self, rank=16, alpha=16):
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.01,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self._accepts_guidance = _flux_unet_accepts_guidance(self.unet)

    # --------- stable pack/unpack (CRITICAL FIX) ----------
    def _pack(self, latents):
        """
        latents: [B,16,H,W] -> tokens: [B, (H//2)*(W//2), 64]
        Uses pixel_unshuffle for a canonical subpixel->channel order.
        """
        b, c, h, w = latents.shape
        if (h % 2) != 0 or (w % 2) != 0:
            raise ValueError(f"H/W must be divisible by 2, got H={h}, W={w}")
        x = F.pixel_unshuffle(latents, downscale_factor=2)  # [B, 64, H//2, W//2]
        x = x.permute(0, 2, 3, 1).contiguous()              # [B, H//2, W//2, 64]
        return x.view(b, (h // 2) * (w // 2), c * 4)        # [B, N, 64]

    def _unpack(self, tokens, h, w):
        """
        tokens: [B, N, 64] -> latents: [B,16,H,W]
        Inverse of _pack via pixel_shuffle.
        """
        b, n, d = tokens.shape
        if d != 64:
            raise ValueError(f"token dim must be 64, got {d}")
        if (h % 2) != 0 or (w % 2) != 0:
            raise ValueError(f"H/W must be divisible by 2, got H={h}, W={w}")
        x = tokens.view(b, h // 2, w // 2, 64).permute(0, 3, 1, 2).contiguous()  # [B,64,H//2,W//2]
        x = F.pixel_shuffle(x, upscale_factor=2)                                   # [B,16,H,W]
        return x

    def _get_ids(self, h, w, device):
        """
        img_ids must align with token grid: (H//2, W//2) tokens.
        """
        h2, w2 = h // 2, w // 2
        h_range = torch.arange(h2, device=device)
        w_range = torch.arange(w2, device=device)
        grid_y, grid_x = torch.meshgrid(h_range, w_range, indexing="ij")
        img_ids = torch.zeros((h2, w2, 3), device=device)
        img_ids[..., 0], img_ids[..., 1] = grid_y, grid_x
        img_ids = img_ids.view(-1, 3).to(torch.bfloat16)
        txt_ids = torch.zeros((512, 3), device=device, dtype=torch.bfloat16)
        return img_ids, txt_ids

    def _blur_latent(self, z):
        """
        z: [B,16,H,W] depthwise 3x3 blur (float32 conv, cast back).
        """
        b, c, h, w = z.shape
        k = self.blur_kernel_3x3.to(z.device, dtype=torch.float32).repeat(c, 1, 1, 1)  # [16,1,3,3]
        return F.conv2d(z.float(), k, padding=1, groups=c).to(z.dtype)

    @torch.no_grad()
    def get_vlm_embedding(self, cond_image):
        """
        Backward-compatible helper used by older inference script.
        Returns projected CLIP token features with shape [B, 257, 4096].
        """
        dtype = torch.bfloat16
        x = (cond_image + 1.0) / 2.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.clip_mean.to(x)) / self.clip_std.to(x)
        clip_outputs = self.clip_vision(x.to(dtype))
        vlm_feats = clip_outputs.last_hidden_state
        return self.vlm_proj(vlm_feats)

    def forward(
        self,
        target_image,      # cartoon domain image if available; otherwise can be same as cond_image
        cond_image=None,   # real photo / identity source
        gamma=3.0,
        proto_momentum=0.99,
        mix_cartoon_latent=0.35,
        flat_strength=0.35,
        guidance_scale=1.0,
        preview_t=0.2,
        force_t=None,
        return_aux=False,
    ):
        device = target_image.device
        dtype = torch.bfloat16
        bsz = target_image.shape[0]
        if cond_image is None:
            cond_image = target_image

        # --------- 1) CLIP encoding ----------
        with torch.no_grad():
            x = (cond_image + 1.0) / 2.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - self.clip_mean.to(x)) / self.clip_std.to(x)
            clip_outputs = self.clip_vision(x.to(dtype))
            vlm_feats = clip_outputs.last_hidden_state   # [B,257,1024]
            vlm_pooled = clip_outputs.pooler_output      # [B,1024]

        # Condition injection（拆出 vlm_proj_out 便于 spike 时打印幅值）
        vlm_proj_out = self.vlm_proj(vlm_feats)
        encoder_hidden_states = torch.zeros((bsz, 512, 4096), device=device, dtype=dtype)
        encoder_hidden_states[:, :257, :] = vlm_proj_out

        pooled_projections = self.pooled_proj(vlm_pooled)  # [B,768]
        guidance = torch.full((bsz,), float(guidance_scale), device=device, dtype=dtype)

        # --------- 2) VAE encode & target_latents ----------
        with torch.no_grad():
            z_real = (self.vae.encode(cond_image.float()).latent_dist.mode() - self.vae.config.shift_factor) * self.scaling_factor
            z_cartoon = (self.vae.encode(target_image.float()).latent_dist.mode() - self.vae.config.shift_factor) * self.scaling_factor

            # proto update (low-res)
            low = F.adaptive_avg_pool2d(z_cartoon, (self.proto_pool, self.proto_pool))
            if not self.proto_initialized:
                self.cartoon_proto_low.copy_(low.mean(0, keepdim=True))
                self.proto_initialized.fill_(True)
            else:
                self.cartoon_proto_low.mul_(proto_momentum).add_(low.mean(0, keepdim=True), alpha=1 - proto_momentum)

            # style bias
            real_low = F.adaptive_avg_pool2d(z_real, (self.proto_pool, self.proto_pool))
            bias_low = (self.cartoon_proto_low.to(z_real) - real_low).clamp(-1.5, 1.5)

            bias_up = F.interpolate(bias_low, size=z_real.shape[-2:], mode="bilinear", align_corners=False)
            bias_up = self._blur_latent(bias_up)

            # gamma 调度只由训练脚本外层控制，此处不再做内层 warmup
            gamma_eff = gamma

            # domain pull + flatten
            z_mix = (1.0 - mix_cartoon_latent) * z_real + mix_cartoon_latent * z_cartoon
            z_lp = self._blur_latent(z_mix)
            z_flat = (1.0 - flat_strength) * z_mix + flat_strength * z_lp

            target_latents = (z_flat + gamma_eff * bias_up).clamp(-5.0, 5.0).to(dtype)  # [B,16,H,W]

        # --------- 3) Flow training in TOKEN space (correct tokenization) ----------
        h, w = target_latents.shape[-2], target_latents.shape[-1]
        packed_latents = self._pack(target_latents)              # [B,N,64]
        noise = torch.randn_like(packed_latents)                 # [B,N,64]
        if force_t is None:
            t = torch.rand((bsz,), device=device, dtype=dtype)   # [B]
        else:
            if not torch.is_tensor(force_t):
                force_t = torch.tensor(float(force_t), device=device, dtype=dtype)
            t = force_t.to(device=device, dtype=dtype).view(-1)
            if t.numel() == 1:
                t = t.repeat(bsz)
            elif t.numel() != bsz:
                raise ValueError(f"force_t length must be 1 or batch size ({bsz}), got {t.numel()}")
        x_t = (1.0 - t.view(-1, 1, 1)) * packed_latents + t.view(-1, 1, 1) * noise

        # dtype safety (avoid rms_norm mismatch)
        x_t = x_t.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)
        pooled_projections = pooled_projections.to(dtype)

        img_ids, txt_ids = self._get_ids(h, w, device)

        unet_kw = dict(
            hidden_states=x_t,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=t * 1000.0,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )
        if self._accepts_guidance:
            unet_kw["guidance"] = guidance
        v_pred = self.unet(**unet_kw)[0]

        v_pred_f = v_pred.float()
        target_v = (noise - packed_latents).float()
        loss = F.mse_loss(v_pred_f, target_v)

        def _print_diagnostics(tag):
            print(f"[{tag}]")
            print("  t mean:", t.float().mean().item())
            print("  gamma:", float(gamma), "gamma_eff:", float(gamma_eff))
            print("  packed_latents absmax:", packed_latents.float().abs().max().item())
            print("  x_t absmax:", x_t.float().abs().max().item())
            print("  v_pred absmax:", v_pred_f.abs().max().item(), "mean abs:", v_pred_f.abs().mean().item())
            print("  target_v absmax:", target_v.abs().max().item(), "mean abs:", target_v.abs().mean().item())
            print("  vlm_proj_out absmax:", vlm_proj_out.float().abs().max().item())
            print("  enc_hidden absmax:", encoder_hidden_states.float().abs().max().item())
            print("  pooled_proj absmax:", pooled_projections.float().abs().max().item())

        if not torch.isfinite(loss):
            self._last_spike_info = {
                "rf_loss": float("nan"),
                "v_pred_absmax": v_pred_f.abs().max().item(),
                "target_v_absmax": target_v.abs().max().item(),
                "t_mean": t.float().mean().item(),
                "vlm_proj_out_absmax": vlm_proj_out.float().abs().max().item(),
                "enc_hidden_absmax": encoder_hidden_states.float().abs().max().item(),
                "pooled_proj_absmax": pooled_projections.float().abs().max().item(),
                "trigger": "non_finite",
            }
            if self._bad_print_count < 5:
                _print_diagnostics("BAD non-finite loss")
                self._bad_print_count += 1
            if return_aux:
                return None, None, None
            return None, None

        if loss.item() > 1000.0:
            if self._spike_print_count < 5:
                _print_diagnostics("SPIKE loss > 1000")
                self._spike_print_count += 1
            self._last_spike_info = {
                "rf_loss": loss.item(),
                "v_pred_absmax": v_pred_f.abs().max().item(),
                "target_v_absmax": target_v.abs().max().item(),
                "t_mean": t.float().mean().item(),
                "trigger": "loss",
            }
        else:
            self._last_spike_info = None

        # --------- 4) Preview ----------
        # Stage3 需要 preview 带梯度，使 L_ID 能回传到 LoRA；其余情况用 no_grad 省显存
        need_preview_grad = self.training and return_aux
        preview_ctx = nullcontext() if need_preview_grad else torch.no_grad()
        with preview_ctx:
            t_p = torch.full((bsz,), float(preview_t), device=device, dtype=dtype)
            noise_p = torch.randn_like(packed_latents)
            x_tp = (1.0 - t_p.view(-1, 1, 1)) * packed_latents + t_p.view(-1, 1, 1) * noise_p

            x_tp = x_tp.to(dtype)

            unet_kw_p = dict(
                hidden_states=x_tp,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=t_p * 1000.0,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )
            if self._accepts_guidance:
                unet_kw_p["guidance"] = guidance
            v_p = self.unet(**unet_kw_p)[0]

            x0 = x_tp - t_p.view(-1, 1, 1) * v_p  # [B,N,64]

            out_latents = self._unpack(x0, h, w).float().clamp(-5.0, 5.0)  # [B,16,H,W]
            out_latents = (out_latents / self.scaling_factor) + self.vae.config.shift_factor
            out_image = self.vae.decode(out_latents).sample

        if not return_aux:
            return loss, out_image

        aux = {
            "sampled_t": t.detach(),
            "packed_latents": packed_latents.detach(),
        }
        return loss, out_image, aux
