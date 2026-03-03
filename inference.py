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
    p.add_argument("--t0", type=float, default=0.5, help="Start timestep for ODE.")
    p.add_argument("--steps", type=int, default=15, help="ODE refinement steps.")
    p.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--run_eval", action="store_true", help="推理并整理后跑 Face-Sim / CLIP Score / FID。")
    p.add_argument("--real_dir", type=str, default=None, help="单张且未 --organize 时评估用原图目录；批量/organize 时无需传。")
    p.add_argument("--reference_dir", type=str, default=None, help="FID 参考图目录；不传则跳过 FID。")
    p.add_argument("--eval_output", type=str, default=None, help="评估结果 JSON 路径（如 stage1_metrics.json）。")
    return p.parse_args()


def _run_one(model, device, img_tensor, t0, steps, guidance):
    """对单张 img_tensor (1,C,H,W) 做一次推理，返回 (1,C,H,W) 在 [0,1]。"""
    with torch.no_grad():
        latents = model.vae.encode(img_tensor.to(torch.float32)).latent_dist.mode()
        latents = (latents - model.vae.config.shift_factor) * model.scaling_factor
        h, w = latents.shape[-2], latents.shape[-1]
        x_start = model._pack(latents).to(torch.bfloat16)
        noise = torch.randn_like(x_start)
        x_t = (1.0 - t0) * x_start + t0 * noise

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
        guidance_t = torch.full((1,), float(guidance), device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        dt = t0 / steps
        for i in range(steps):
            current_t = t0 * (1 - i / steps)
            t_tensor = torch.full((1,), current_t, device=device, dtype=torch.bfloat16)
            v_pred = model.unet(
                hidden_states=x_t,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=t_tensor * 1000,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance_t,
                return_dict=False,
            )[0]
            v_pred = torch.nan_to_num(v_pred, nan=0.0).clamp(-3.0, 3.0)
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
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    model.vae.to(torch.float32)
    model.eval()

    if batch_mode:
        for name in tqdm(image_names, desc="Inference"):
            inp_path = os.path.join(args.input_dir, name)
            stem = Path(name).stem
            ext = Path(name).suffix.lower()
            img_tensor = transform(Image.open(inp_path).convert("RGB")).unsqueeze(0).to(device)
            out_img = _run_one(model, device, img_tensor, args.t0, args.steps, args.guidance)
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
    out_img = _run_one(model, device, img_tensor, args.t0, args.steps, args.guidance)
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
