"""
First diagnostic: check VAE + pack/unpack round-trip.

Usage (from project root):
  python scripts/check_pack_unpack.py --input ./data/real/xxx.jpg --output recon_pack_unpack.png

Encodes input -> pack -> unpack -> decode, saves reconstruction.
- If recon looks like the original (maybe slightly blurry): pack/unpack and VAE are fine; try single-step x0 next.
- If recon has grid/stripes/artifacts: fix _pack/_unpack or latent scale first.
"""
import argparse
import os
import sys

# Ensure project models can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

from models.flux_i2i_trainable import FluxI2ITrainable


def main():
    p = argparse.ArgumentParser(description="Check VAE + pack/unpack round-trip.")
    p.add_argument("--input", type=str, default="./data/real/photo4.jpg", help="Input image path.")
    p.add_argument("--output", type=str, default="./recon_pack_unpack.png", help="Reconstruction output path.")
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
        # Same encode scale as in inference
        latents = model.vae.encode(img.float()).latent_dist.mode()
        latents = (latents - model.vae.config.shift_factor) * model.scaling_factor
        # pack -> unpack
        packed = model._pack(latents)
        unpacked = model._unpack(packed, latents.shape[-2], latents.shape[-1])
        # Back to VAE decode scale
        out_latents = (unpacked.float() / model.scaling_factor) + model.vae.config.shift_factor
        recon = model.vae.decode(out_latents).sample

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    vutils.save_image((recon.clamp(-1, 1) + 1) / 2, out_path)
    print(f"Saved: {out_path}")
    print("  -> If recon looks like original (maybe slightly blurry): pack/unpack OK; try single-step x0.")
    print("  -> If grid/stripes/artifacts: fix _pack/_unpack or latent scale.")


if __name__ == "__main__":
    main()
