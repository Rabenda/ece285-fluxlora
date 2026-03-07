"""
第一步排查：VAE + pack/unpack 往返是否正常。

用法（在项目根目录 ece_285_deep_generate_model/project 下运行）：
  python scripts/check_pack_unpack.py --input ./data/real/xxx.jpg --output recon_pack_unpack.png

作用：对输入图做  encode -> pack -> unpack -> decode，保存重建图。
- 若 recon 基本像原图（顶多略糊）：pack/unpack 和 VAE 链没问题，可继续做「单步 x0」实验。
- 若 recon 出现栅格/条纹/色块：问题在 _pack/_unpack 或 latent 尺度，需先修这里。
"""
import argparse
import os
import sys

# 保证能 import 到 project 下的 models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

from models.flux_i2i_trainable import FluxI2ITrainable


def main():
    p = argparse.ArgumentParser(description="Check VAE + pack/unpack round-trip.")
    p.add_argument("--input", type=str, default="./data/real/photo4.jpg", help="输入图片路径。")
    p.add_argument("--output", type=str, default="./recon_pack_unpack.png", help="重建图保存路径。")
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

    print("Loading model (VAE + pack/unpack only, no LoRA needed) ...")
    model = FluxI2ITrainable().to(device)
    model.eval()

    img = transform(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        # 与 inference 里一致的 encode 尺度
        latents = model.vae.encode(img.float()).latent_dist.mode()
        latents = (latents - model.vae.config.shift_factor) * model.scaling_factor
        # pack -> unpack
        packed = model._pack(latents)
        unpacked = model._unpack(packed, latents.shape[-2], latents.shape[-1])
        # 还原回 VAE 解码用的尺度
        out_latents = (unpacked.float() / model.scaling_factor) + model.vae.config.shift_factor
        recon = model.vae.decode(out_latents).sample

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    vutils.save_image((recon.clamp(-1, 1) + 1) / 2, out_path)
    print(f"Saved: {out_path}")
    print("  -> 若图基本像原图（顶多略糊）：pack/unpack 正常，可做单步 x0 实验。")
    print("  -> 若有栅格/条纹/色块：问题在 _pack/_unpack 或 latent 尺度。")


if __name__ == "__main__":
    main()
