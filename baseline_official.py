"""
官方 FLUX.1-schnell Img2Img 基线：用于对比 Stage2/Stage3 自训模型效果。
"""
import argparse
import os

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Official FLUX.1-schnell Img2Img baseline.")
    p.add_argument("--input", type=str, required=True, help="输入图像路径。")
    p.add_argument("--output", type=str, required=True, help="输出图像路径。")
    p.add_argument("--prompt", type=str, default="A cartoon character, anime style, high quality, digital art", help="引导文本。")
    p.add_argument("--strength", type=float, default=0.6, help="对原图的修改程度 (0.0 完全不变, 1.0 全新生成)。")
    p.add_argument("--steps", type=int, default=8, help="推理步数。")
    return p.parse_args()


def run_official_baseline(input_path, output_path, prompt="A cartoon character, anime style, high quality, digital art", strength=0.6, num_inference_steps=8):
    model_id = "black-forest-labs/FLUX.1-schnell"

    pipe = FluxImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    init_image = Image.open(input_path).convert("RGB").resize((512, 512))

    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0
    ).images[0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)
    print(f"Official Baseline saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_official_baseline(
        args.input,
        args.output,
        prompt=args.prompt,
        strength=args.strength,
        num_inference_steps=args.steps,
    )
