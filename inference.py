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
    p.add_argument("--t_stop", type=float, default=0.0, help="多步推理从 t=1 积分到 t_stop。越小越卡通，越大越保守。")
    p.add_argument("--steps", type=int, default=20, help="ODE 积分步数。")
    p.add_argument("--single_step", action="store_true", help="source-end 单步 probe，不跑多步。")
    p.add_argument("--single_step_t", type=float, default=0.15, help="单步 probe 的 step_scale，控制步长。")
    p.add_argument("--noise_factor", type=float, default=0.02, help="起点 latent 加的噪声比例，与训练一致。")
    p.add_argument("--guidance", type=float, default=1.0, help="Guidance scale when UNet supports it (older diffusers/FLUX). For diffusers>=0.30 + FLUX.1-schnell, 此值會被自動忽略。")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--run_eval", action="store_true", help="推理并整理后跑 Face-Sim / CLIP Score / FID。")
    p.add_argument("--real_dir", type=str, default=None, help="单张且未 --organize 时评估用原图目录；批量/organize 时无需传。")
    p.add_argument("--reference_dir", type=str, default=None, help="FID 参考图目录；不传则跳过 FID。")
    p.add_argument("--eval_output", type=str, default=None, help="评估结果 JSON 路径（如 stage1_metrics.json）。")
    return p.parse_args()


def _run_one(model, device, img_tensor, t_stop, steps, guidance, single_step=False, single_step_t=0.2, noise_factor=0.02):
    """对单张 img_tensor (1,C,H,W) 做一次推理，返回 (C,H,W) 在 [0,1]。
    I2I Rectified Flow：起点 = 真人 latent + 微量噪声 (t=1)，积分到 t=t_stop。
    控制风格强度靠 t_stop：0=强卡通，0.5=轻风格。
    """
    steps = max(1, int(steps))
    with torch.no_grad():
        latents = model.vae.encode(img_tensor.to(torch.float32)).latent_dist.mode()
        latents = (latents - model.vae.config.shift_factor) * model.scaling_factor
        h, w = latents.shape[-2], latents.shape[-1]
        packed_real = model._pack(latents).to(torch.bfloat16)
        noise = torch.randn_like(packed_real)
        packed_source = packed_real + noise_factor * noise
        x_t = packed_source.clone()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            clip_in = F.interpolate((img_tensor + 1.0) / 2.0, size=(224, 224), mode="bilinear", align_corners=False)
            clip_in = (clip_in - model.clip_mean.to(img_tensor.device)) / model.clip_std.to(img_tensor.device)
            clip_outputs = model.clip_vision(clip_in.to(torch.bfloat16))
            vlm_feats_raw = clip_outputs.last_hidden_state
            vlm_proj_out = model.vlm_proj(vlm_feats_raw)
            vlm_proj_out = F.layer_norm(vlm_proj_out, (vlm_proj_out.shape[-1],))
            vlm_proj_out = vlm_proj_out * 0.1
            pooled_raw = clip_outputs.pooler_output
            pooled_projections = model.pooled_proj(pooled_raw)
            pooled_projections = F.layer_norm(pooled_projections, (pooled_projections.shape[-1],))
            pooled_projections = pooled_projections * 0.05

        encoder_hidden_states = torch.zeros((1, 512, 4096), device=device, dtype=torch.bfloat16)
        encoder_hidden_states[:, :257, :] = vlm_proj_out.to(torch.bfloat16)
        img_ids, txt_ids = model._get_ids(h, w, device)
        guidance_t = torch.full((1,), 0.0, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        if single_step:
            # source-end probe：状态是 source，时间标签必须是 1.0
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
            step_scale = max(1e-4, min(1.0, float(single_step_t)))
            x_t = x_t - step_scale * v_pred
        else:
            # 从 t=1.0 走到 t=t_stop，控制风格强度靠提前停下
            t_stop = max(0.0, min(1.0, float(t_stop)))
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

    if args.checkpoint is None:
        print("Loading pretrained model only (NO checkpoint).")
    else:
        print(f"Loading model from checkpoint: {args.checkpoint}")
    model = FluxI2ITrainable().to(device)
    model.setup_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
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
    print(f"[Inference] UNet accepts guidance: {getattr(model, '_accepts_guidance', False)}")

    if batch_mode:
        for name in tqdm(image_names, desc="Inference"):
            inp_path = os.path.join(args.input_dir, name)
            stem = Path(name).stem
            ext = Path(name).suffix.lower()
            img_tensor = transform(Image.open(inp_path).convert("RGB")).unsqueeze(0).to(device)
            out_img = _run_one(model, device, img_tensor, args.t_stop, args.steps, args.guidance, args.single_step, args.single_step_t, noise_factor=args.noise_factor)
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
    out_img = _run_one(model, device, img_tensor, args.t_stop, args.steps, args.guidance, args.single_step, args.single_step_t, noise_factor=args.noise_factor)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(out_img.unsqueeze(0), args.output)
    print(f"Saved to: {args.output}")

    if args.organize:
        real_sub = os.path.join(out_dir or ".", "real")
        gen_sub = os.path.join(out_dir or ".", "gen")
        os.makedirs(real_sub, exist_ok=True)
        os.makedirs(gen_sub, exist_ok=True)
        out_stem = Path(args.output).stem
        in_ext = Path(args.input).suffix.lower()
        shutil.copy2(args.input, os.path.join(real_sub, out_stem + in_ext))
        shutil.copy2(args.output, os.path.join(gen_sub, Path(args.output).name))
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
            out_basename = os.path.basename(args.output)
            print("\nRunning evaluation (Face-Sim, CLIP Score, FID)...")
            from evaluate import run_evaluation
            with tempfile.TemporaryDirectory() as tmp:
                shutil.copy2(args.output, os.path.join(tmp, out_basename))
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
