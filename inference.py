"""
统一推理脚本：Stage1（原版 Flux）与 Stage2/3（加载训好的 checkpoint）共用。

- 不传 --checkpoint：使用 HuggingFace 预训练权重 + 零初始化 LoRA，即 Stage1 基线。
- 传 --checkpoint path/to/full_step_*.pt：加载该权重，即 Stage2/Stage3 推理。

支持单张（--input / --output）、批量（--input_dir / --output_dir）、
--organize（整理为 real/ 与 gen/）、--run_eval（Face-Sim / CLIP Score / FID）。

# Stage1 基线（不加载 checkpoint）
python inference.py --input ./data/real/photo4.jpg --output ./out/photo4.png

# Stage2/3 推理（加载训好的权重）
python inference.py --checkpoint ./checkpoints/full_step_000500.pt --input ./data/real/photo4.jpg --output ./out/photo4.png

# 批量 + 评估（Stage2/3）
python inference.py --checkpoint ./checkpoints/full_step_000500.pt \
  --input_dir ./data/real --output_dir ./stage3_results \
  --run_eval --reference_dir ./data/anime_ref --eval_output ./stage3_metrics.json
  
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.utils as vutils

from models.flux_i2i_trainable import FluxI2ITrainable

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _list_images(dir_path: str):
    return sorted(
        [f for f in os.listdir(dir_path) if f.lower().endswith(IMG_EXTENSIONS)],
        key=lambda x: Path(x).stem,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Unified inference: Stage1 (no ckpt) or Stage2/3 (with --checkpoint).")
    p.add_argument("--checkpoint", type=str, default=None, help="可选。训好的权重路径（如 full_step_000500.pt）。不传则为 Stage1 基线。")
    p.add_argument("--input", type=str, default="./data/inference/photo4.jpg", help="单张输入路径（与 --output 搭配）。")
    p.add_argument("--output", type=str, default="./inference_output.png", help="单张输出路径。")
    p.add_argument("--input_dir", type=str, default=None, help="批量：原图目录。与 --output_dir 同时指定时启用批量。")
    p.add_argument("--output_dir", type=str, default=None, help="批量：结果目录，将创建 real/ 与 gen/ 子目录并自动整理。")
    p.add_argument("--organize", action="store_true", help="单张模式：在 output 所在目录下创建 real/ 与 gen/ 并整理，便于评估。")
    p.add_argument("--t_stop", type=float, default=0.0, help="[FULL] 多步从 t=1 积分到 t_stop。越小越卡通。")
    p.add_argument("--t0", type=float, default=0.5, help="[LITE] 多步起始时间 t0，dt=t0/steps。")
    p.add_argument("--steps", type=int, default=20, help="ODE 积分步数。")
    p.add_argument("--single_step", action="store_true", help="单步模式（FULL=source-end probe，LITE=preview 同款）。")
    p.add_argument("--single_step_t", type=float, default=0.15, help="单步步长/step_scale。")
    p.add_argument("--noise_factor", type=float, default=0.02, help="[FULL] 起点 latent 噪声比例。")
    p.add_argument("--cond_scale_vlm", type=float, default=0.1, help="[FULL] 条件缩放 a1（vlm）。")
    p.add_argument("--cond_scale_pooled", type=float, default=0.05, help="[FULL] 条件缩放 a2（pooled）。")
    p.add_argument("--guidance", type=float, default=1.0, help="Guidance scale when UNet supports it. Schnell 推理时自动忽略。")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--run_eval", action="store_true", help="推理并整理后跑 Face-Sim / CLIP Score / FID。")
    p.add_argument("--real_dir", type=str, default=None, help="单张且未 --organize 时评估用原图目录；批量/organize 时无需传。")
    p.add_argument("--reference_dir", type=str, default=None, help="FID 参考图目录；不传则跳过 FID。")
    p.add_argument("--eval_output", type=str, default=None, help="评估结果 JSON 路径（如 stage1_metrics.json）。")
    return p.parse_args()


def _run_one(model, device, img_tensor, args):
    """对单张 img_tensor (1,C,H,W) 做一次推理，返回 (C,H,W) 在 [0,1]。
    根据 model._mode 选 FULL（I2I RF, t_stop/a1/a2）或 LITE（noise-conditioned, t0）。
    """
    steps = max(1, int(args.steps))
    is_lite = getattr(model, "_mode", "full") == "lite"
    with torch.no_grad():
        latents = model.vae.encode(img_tensor.to(torch.float32)).latent_dist.mode()
        latents = (latents - model.vae.config.shift_factor) * model.scaling_factor
        h, w = latents.shape[-2], latents.shape[-1]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            clip_in = F.interpolate((img_tensor + 1.0) / 2.0, size=(224, 224), mode="bilinear", align_corners=False)
            clip_in = (clip_in - model.clip_mean.to(img_tensor.device)) / model.clip_std.to(img_tensor.device)
            clip_outputs = model.clip_vision(clip_in.to(torch.bfloat16))
            vlm_feats_raw = clip_outputs.last_hidden_state
            vlm_proj_out = model.vlm_proj(vlm_feats_raw)
            vlm_proj_out = F.layer_norm(vlm_proj_out, (vlm_proj_out.shape[-1],))
            pooled_raw = clip_outputs.pooler_output
            pooled_projections = model.pooled_proj(pooled_raw)
            pooled_projections = F.layer_norm(pooled_projections, (pooled_projections.shape[-1],))
            if is_lite:
                vlm_proj_out = vlm_proj_out * 0.01
                pooled_projections = pooled_projections * 0.01
            else:
                vlm_proj_out = vlm_proj_out * args.cond_scale_vlm
                pooled_projections = pooled_projections * args.cond_scale_pooled
        encoder_hidden_states = torch.zeros((1, 512, 4096), device=device, dtype=torch.bfloat16)
        encoder_hidden_states[:, :257, :] = vlm_proj_out.to(torch.bfloat16)
        img_ids, txt_ids = model._get_ids(h, w, device)
        guidance_t = torch.full((1,), 0.0, device=device, dtype=torch.bfloat16)

    if is_lite:
        # LITE: x_t = (1-t)*x_start + t*noise, 多步 dt=t0/steps
        with torch.no_grad():
            x_start = model._pack(latents).to(torch.bfloat16)
            noise = torch.randn_like(x_start)
            t_use = float(args.single_step_t) if args.single_step else float(args.t0)
            x_t = (1.0 - t_use) * x_start + t_use * noise
        with torch.no_grad():
            if args.single_step:
                t_tensor = torch.full((1,), t_use, device=device, dtype=torch.bfloat16)
                unet_kw = dict(
                    hidden_states=x_t.to(torch.bfloat16),
                    encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
                    pooled_projections=pooled_projections.to(torch.bfloat16),
                    timestep=t_tensor,
                    img_ids=img_ids.to(torch.bfloat16),
                    txt_ids=txt_ids.to(torch.bfloat16),
                    return_dict=False,
                )
                if getattr(model, "_accepts_guidance", False):
                    unet_kw["guidance"] = guidance_t
                v_pred = model.unet(**unet_kw)[0]
                v_pred = torch.nan_to_num(v_pred, nan=0.0)
                x_t = x_t - t_tensor.view(-1, 1, 1) * v_pred
            else:
                dt = args.t0 / steps
                for i in range(steps):
                    current_t = args.t0 * (1 - i / steps)
                    t_tensor = torch.full((1,), current_t, device=device, dtype=torch.bfloat16)
                    unet_kw = dict(
                        hidden_states=x_t.to(torch.bfloat16),
                        encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
                        pooled_projections=pooled_projections.to(torch.bfloat16),
                        timestep=t_tensor,
                        img_ids=img_ids.to(torch.bfloat16),
                        txt_ids=txt_ids.to(torch.bfloat16),
                        return_dict=False,
                    )
                    if getattr(model, "_accepts_guidance", False):
                        unet_kw["guidance"] = guidance_t
                    v_pred = model.unet(**unet_kw)[0]
                    v_pred = torch.nan_to_num(v_pred, nan=0.0)
                    x_t = x_t - v_pred * dt
    else:
        # FULL: I2I RF，起点 packed_source，从 t=1 积到 t_stop
        with torch.no_grad():
            packed_real = model._pack(latents).to(torch.bfloat16)
            noise = torch.randn_like(packed_real)
            packed_source = packed_real + args.noise_factor * noise
            x_t = packed_source.clone()
        with torch.no_grad():
            if args.single_step:
                t_tensor = torch.full((1,), 1.0, device=device, dtype=torch.bfloat16)
                unet_kw = dict(
                    hidden_states=x_t.to(torch.bfloat16),
                    encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
                    pooled_projections=pooled_projections.to(torch.bfloat16),
                    timestep=t_tensor,
                    img_ids=img_ids.to(torch.bfloat16),
                    txt_ids=txt_ids.to(torch.bfloat16),
                    return_dict=False,
                )
                if getattr(model, "_accepts_guidance", False):
                    unet_kw["guidance"] = guidance_t
                v_pred = model.unet(**unet_kw)[0]
                v_pred = torch.nan_to_num(v_pred, nan=0.0)
                step_scale = max(1e-4, min(1.0, float(args.single_step_t)))
                x_t = x_t - step_scale * v_pred
            else:
                t_stop = max(0.0, min(1.0, float(args.t_stop)))
                total_span = 1.0 - t_stop
                dt = total_span / steps
                for i in range(steps):
                    current_t = 1.0 - i * dt
                    t_tensor = torch.full((1,), current_t, device=device, dtype=torch.bfloat16)
                    unet_kw = dict(
                        hidden_states=x_t.to(torch.bfloat16),
                        encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
                        pooled_projections=pooled_projections.to(torch.bfloat16),
                        timestep=t_tensor,
                        img_ids=img_ids.to(torch.bfloat16),
                        txt_ids=txt_ids.to(torch.bfloat16),
                        return_dict=False,
                    )
                    if getattr(model, "_accepts_guidance", False):
                        unet_kw["guidance"] = guidance_t
                    v_pred = model.unet(**unet_kw)[0]
                    v_pred = torch.nan_to_num(v_pred, nan=0.0)
                    x_t = x_t - v_pred * dt

    with torch.no_grad():
        out_latents = model._unpack(x_t, h, w).to(torch.float32)
        out_latents = out_latents.clamp(-5.0, 5.0)
        out_latents = (out_latents / model.scaling_factor) + model.vae.config.shift_factor
        final_image = model.vae.decode(out_latents).sample
    return (final_image[0].cpu().clamp(-1, 1) + 1) / 2


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    batch_mode = args.input_dir and args.output_dir and os.path.isdir(args.input_dir)

    if batch_mode:
        image_names = _list_images(args.input_dir)
        if not image_names:
            print(f"No images in {args.input_dir}.")
            return
        real_dir = os.path.join(args.output_dir, "real")
        gen_dir = os.path.join(args.output_dir, "gen")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
        mode_label = "Stage1 (no ckpt)" if args.checkpoint is None else f"Stage2/3 (ckpt: {args.checkpoint})"
        print(f"[Inference] Batch: {len(image_names)} images -> {args.output_dir}/real/ and .../gen/ ({mode_label})")
    else:
        if not os.path.isfile(args.input):
            print(f"Input not found: {args.input}")
            return
        mode_label = "Stage1 (no ckpt)" if args.checkpoint is None else f"Stage2/3 (ckpt: {args.checkpoint})"
        print(f"[Inference] Single image ({mode_label}).")

    ckpt = None
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        print(f"Loading model from checkpoint: {args.checkpoint}")
    else:
        print("Loading pretrained model only (NO checkpoint).")
    mode = ckpt.get("mode", "full") if isinstance(ckpt, dict) else "full"
    model = FluxI2ITrainable(mode=mode).to(device)
    model.setup_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    if ckpt is not None:
        if isinstance(ckpt, dict):
            # 優先載入 LoRA 權重（新格式）
            if "lora" in ckpt:
                res = model.unet.load_state_dict(ckpt["lora"], strict=False)
                print("[LoRA load] missing:", len(res.missing_keys), "unexpected:", len(res.unexpected_keys))
            # 向後相容舊格式：整個 model_state_dict
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "vlm_proj" in ckpt:
                model.vlm_proj.load_state_dict(ckpt["vlm_proj"])
            if "pooled_proj" in ckpt:
                model.pooled_proj.load_state_dict(ckpt["pooled_proj"])
        else:
            # 最後退路：當成完整 state_dict 載入
            model.load_state_dict(ckpt, strict=False)
    model.vae.to(torch.float32)
    model.eval()
    print(f"[Inference] mode={mode}, UNet accepts guidance: {getattr(model, '_accepts_guidance', False)}")

    if batch_mode:
        for name in tqdm(image_names, desc="Inference"):
            inp_path = os.path.join(args.input_dir, name)
            stem = Path(name).stem
            ext = Path(name).suffix.lower()
            img_tensor = transform(Image.open(inp_path).convert("RGB")).unsqueeze(0).to(device)
            out_img = _run_one(model, device, img_tensor, args)
            gen_path = os.path.join(gen_dir, stem + ".png")
            vutils.save_image(out_img.unsqueeze(0), gen_path)
            shutil.copy2(inp_path, os.path.join(real_dir, stem + ext))
        print(f"Inference: {len(image_names)} images saved under {args.output_dir}/real/ and .../gen/")
        if args.run_eval:
            print("\nRunning evaluation (Face-Sim, CLIP Score, FID)...")
            from evaluate import run_evaluation
            run_evaluation(
                real_dir=real_dir,
                gen_dir=gen_dir,
                reference_dir=args.reference_dir,
                output_path=args.eval_output,
            )
        return

    # 单张模式
    img_tensor = transform(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)
    print("Refining trajectory...")
    out_img = _run_one(model, device, img_tensor, args)
    out_path = args.output
    if not Path(out_path).suffix:
        out_path = out_path.rstrip("/") + ".png"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(out_img.unsqueeze(0), out_path)
    print(f"Saved to: {out_path}")

    if args.organize:
        real_sub = os.path.join(out_dir or ".", "real")
        gen_sub = os.path.join(out_dir or ".", "gen")
        os.makedirs(real_sub, exist_ok=True)
        os.makedirs(gen_sub, exist_ok=True)
        out_stem = Path(out_path).stem
        in_ext = Path(args.input).suffix.lower()
        shutil.copy2(args.input, os.path.join(real_sub, out_stem + in_ext))
        shutil.copy2(out_path, os.path.join(gen_sub, Path(out_path).name))
        print(f"Organized: {real_sub}/ and {gen_sub}/")

    if args.run_eval:
        if args.organize and out_dir:
            real_dir = os.path.join(out_dir, "real")
            gen_dir = os.path.join(out_dir, "gen")
            print("\nRunning evaluation (Face-Sim, CLIP Score, FID)...")
            from evaluate import run_evaluation
            run_evaluation(
                real_dir=real_dir,
                gen_dir=gen_dir,
                reference_dir=args.reference_dir,
                output_path=args.eval_output,
            )
        elif args.real_dir and os.path.isdir(args.real_dir):
            out_basename = os.path.basename(out_path)
            print("\nRunning evaluation (Face-Sim, CLIP Score, FID)...")
            from evaluate import run_evaluation
            with tempfile.TemporaryDirectory() as tmp:
                shutil.copy2(out_path, os.path.join(tmp, out_basename))
                run_evaluation(
                    real_dir=args.real_dir,
                    gen_dir=tmp,
                    reference_dir=args.reference_dir,
                    output_path=args.eval_output,
                )
        else:
            print("Skipping evaluation: use --organize or --real_dir (and ensure output is in a dir) for --run_eval.")


if __name__ == "__main__":
    main()
