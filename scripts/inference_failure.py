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
    p = argparse.ArgumentParser(description="Unified inference: Optimized for Stage3 Style vs Identity.")
    p.add_argument("--checkpoint", type=str, default=None, help="可选。训好的权重路径。")
    p.add_argument("--input", type=str, default="./data/inference/photo4.jpg", help="单张输入路径。")
    p.add_argument("--output", type=str, default="./inference_output.png", help="单张输出路径。")
    p.add_argument("--input_dir", type=str, default=None, help="批量：原图目录。")
    p.add_argument("--output_dir", type=str, default=None, help="批量：结果目录。")
    p.add_argument("--organize", action="store_true", help="自动整理 real/ 与 gen/ 目录。")
    
    # --- 核心推理参数 ---
    p.add_argument("--t0", type=float, default=0.35, help="ODE 起始步。0.5风格强但易崩，0.25保真但像重建。建议 0.3-0.4。")
    p.add_argument("--steps", type=int, default=25, help="ODE 步数。多步 Euler 积分更平滑。")
    p.add_argument("--style_strength", type=float, default=0.6, help="风格强度缩放。减小此值（如0.5）可有效解决‘变红’和‘噪点’。")
    p.add_argument("--clamp_val", type=float, default=1.2, help="v_pred 的截断值。收紧此值（如0.8-1.2）可防止色彩溢出。")
    
    p.add_argument("--single_step", action="store_true", help="单步模式（类似训练 Preview）。")
    p.add_argument("--single_step_t", type=float, default=0.2)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--run_eval", action="store_true")
    p.add_argument("--real_dir", type=str, default=None)
    p.add_argument("--reference_dir", type=str, default=None)
    p.add_argument("--eval_output", type=str, default=None)
    return p.parse_args()

def _run_one(model, device, img_tensor, t0, steps, guidance, style_strength=1.0, clamp_val=3.0, single_step=False, single_step_t=0.2):
    """优化后的推理核心：引入强度缩放与动态截断"""
    steps = max(1, int(steps))
    t_use = float(single_step_t) if single_step else t0
    
    with torch.no_grad():
        # VAE 编码
        latents = model.vae.encode(img_tensor.to(torch.float32)).latent_dist.mode()
        latents = (latents - model.vae.config.shift_factor) * model.scaling_factor
        h, w = latents.shape[-2], latents.shape[-1]
        x_start = model._pack(latents).to(torch.bfloat16)
        noise = torch.randn_like(x_start)
        x_t = (1.0 - t_use) * x_start + t_use * noise

        # CLIP Conditioning 路径
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device == "cuda")):
            clip_in = F.interpolate((img_tensor + 1.0) / 2.0, size=(224, 224), mode="bilinear", align_corners=False)
            clip_in = (clip_in - model.clip_mean.to(img_tensor.device)) / model.clip_std.to(img_tensor.device)
            clip_outputs = model.clip_vision(clip_in.to(torch.bfloat16))

            vlm_proj_out = model.vlm_proj(clip_outputs.last_hidden_state)
            vlm_proj_out = F.layer_norm(vlm_proj_out, (vlm_proj_out.shape[-1],)) * 0.01

            pooled_projections = model.pooled_proj(clip_outputs.pooler_output)
            pooled_projections = F.layer_norm(pooled_projections, (pooled_projections.shape[-1],)) * 0.01

        encoder_hidden_states = torch.zeros((1, 512, 4096), device=device, dtype=torch.bfloat16)
        encoder_hidden_states[:, :257, :] = vlm_proj_out.to(torch.bfloat16)
        img_ids, txt_ids = model._get_ids(h, w, device)
        guidance_t = torch.full((1,), float(guidance), device=device, dtype=torch.bfloat16)

        # 积分循环
        if single_step:
            t_tensor = torch.full((1,), t_use, device=device, dtype=torch.bfloat16)
            unet_kw = dict(hidden_states=x_t, encoder_hidden_states=encoder_hidden_states, 
                           pooled_projections=pooled_projections, timestep=t_tensor * 1000,
                           img_ids=img_ids, txt_ids=txt_ids, return_dict=False)
            if getattr(model, "_accepts_guidance", False): unet_kw["guidance"] = guidance_t
            
            v_pred = model.unet(**unet_kw)[0]
            # 应用强度缩放与截断
            v_pred = v_pred * style_strength
            v_pred = torch.nan_to_num(v_pred, nan=0.0).clamp(-clamp_val, clamp_val)
            x_t = x_t - t_tensor.view(-1, 1, 1) * v_pred
        else:
            dt = t0 / steps
            for i in range(steps):
                current_t = t0 * (1 - i / steps)
                t_tensor = torch.full((1,), current_t, device=device, dtype=torch.bfloat16)
                unet_kw = dict(hidden_states=x_t, encoder_hidden_states=encoder_hidden_states,
                               pooled_projections=pooled_projections, timestep=t_tensor * 1000,
                               img_ids=img_ids, txt_ids=txt_ids, return_dict=False)
                if getattr(model, "_accepts_guidance", False): unet_kw["guidance"] = guidance_t
                
                v_pred = model.unet(**unet_kw)[0]
                # 核心修正：缩放 v_pred 抑制累积误差导致的变红
                v_pred = v_pred * style_strength
                v_pred = torch.nan_to_num(v_pred, nan=0.0).clamp(-clamp_val, clamp_val)
                x_t = x_t - v_pred * dt

        # VAE 解码
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

    # 加载模型
    model = FluxI2ITrainable().to(device)
    model.setup_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict):
            if "lora" in ckpt: model.unet.load_state_dict(ckpt["lora"], strict=False)
            if "vlm_proj" in ckpt: model.vlm_proj.load_state_dict(ckpt["vlm_proj"])
            if "pooled_proj" in ckpt: model.pooled_proj.load_state_dict(ckpt["pooled_proj"])
        else:
            model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 处理逻辑
    batch_mode = args.input_dir and args.output_dir
    if batch_mode:
        os.makedirs(os.path.join(args.output_dir, "gen"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "real"), exist_ok=True)
        image_names = _list_images(args.input_dir)
        for name in tqdm(image_names):
            inp_path = os.path.join(args.input_dir, name)
            img_tensor = transform(Image.open(inp_path).convert("RGB")).unsqueeze(0).to(device)
            out_img = _run_one(model, device, img_tensor, args.t0, args.steps, args.guidance, 
                              args.style_strength, args.clamp_val, args.single_step, args.single_step_t)
            vutils.save_image(out_img, os.path.join(args.output_dir, "gen", Path(name).stem + ".png"))
            shutil.copy2(inp_path, os.path.join(args.output_dir, "real", name))
    else:
        img_tensor = transform(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)
        out_img = _run_one(model, device, img_tensor, args.t0, args.steps, args.guidance, 
                          args.style_strength, args.clamp_val, args.single_step, args.single_step_t)
        vutils.save_image(out_img, args.output)

if __name__ == "__main__":
    main()