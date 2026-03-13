"""
Second diagnostic: single-step x0 recovery (preview formula) vs multi-step Euler.

Usage (from project root):
  python scripts/check_single_step_x0.py --input ./data/real/xxx.jpg --output single_step_x0.png

Runs once x0 = x_t - t * v_pred (t=0.2), no multi-step ODE, then decode and save.
- If single-step is much better than 35-step Euler: multi-step sampler is the main issue; fix inference ODE/schedule.
- If single-step is also noisy/grid: issue is earlier (conditioning / img_ids / or base usage).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

from models.flux_i2i_trainable import FluxI2ITrainable


def main():
    p = argparse.ArgumentParser(description="Check single-step x0 recovery (preview formula).")
    p.add_argument("--input", type=str, default="./data/real/photo4.jpg", help="Input image path.")
    p.add_argument("--output", type=str, default="./single_step_x0.png", help="Output path.")
    p.add_argument("--t", type=float, default=0.2, help="Single-step t (match training preview_t).")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input not found: {args.input}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    print("Loading model (Stage1: no checkpoint, zero LoRA) ...")
    model = FluxI2ITrainable().to(device)
    model.setup_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    model.eval()

    img_tensor = transform(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)
    t_val = args.t

    with torch.no_grad():
        latents = model.vae.encode(img_tensor.to(torch.float32)).latent_dist.mode()
        latents = (latents - model.vae.config.shift_factor) * model.scaling_factor
        h, w = latents.shape[-2], latents.shape[-1]
        x_start = model._pack(latents).to(torch.bfloat16)
        noise = torch.randn_like(x_start)
        x_t = (1.0 - t_val) * x_start + t_val * noise

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device == "cuda")):
            vlm_feats = model.get_vlm_embedding(img_tensor)
            pooled = model.clip_vision(
                ((F.interpolate((img_tensor + 1.0) / 2.0, size=(224, 224), mode="bilinear", align_corners=False)
                  - model.clip_mean.to(img_tensor.device)) / model.clip_std.to(img_tensor.device)).to(torch.bfloat16)
            ).pooler_output
            pooled_projections = model.pooled_proj(pooled).to(torch.bfloat16)
        encoder_hidden_states = torch.zeros((1, 512, 4096), device=device, dtype=torch.bfloat16)
        encoder_hidden_states[:, :257, :] = vlm_feats
        img_ids, txt_ids = model._get_ids(h, w, device)
        guidance = 1.0
        guidance_t = torch.full((1,), float(guidance), device=device, dtype=torch.bfloat16)

        t_tensor = torch.full((1,), t_val, device=device, dtype=torch.bfloat16)
        unet_kw = dict(
            hidden_states=x_t,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=t_tensor * 1000,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )
        if getattr(model, "_accepts_guidance", False):
            unet_kw["guidance"] = guidance_t
        v_pred = model.unet(**unet_kw)[0]
        v_pred = torch.nan_to_num(v_pred, nan=0.0).clamp(-3.0, 3.0)

        # Single-step x0 (preview formula), no multi-step loop
        x0 = x_t - t_tensor.view(-1, 1, 1) * v_pred
        out_latents = model._unpack(x0, h, w).to(torch.float32)
        out_latents = out_latents.clamp(-5.0, 5.0)
        out_latents = (out_latents / model.scaling_factor) + model.vae.config.shift_factor
        final_image = model.vae.decode(out_latents).sample

    out_arr = (final_image[0].cpu().clamp(-1, 1) + 1) / 2
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    vutils.save_image(out_arr, out_path)
    print(f"Saved: {out_path}")
    print("  -> If much better than 35-step Euler: multi-step sampler is the cause; fix inference ODE/schedule.")
    print("  -> If also noisy/grid: issue is in conditioning / img_ids or FLUX usage.")


if __name__ == "__main__":
    main()
