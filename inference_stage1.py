"""
Official FLUX.1-schnell Img2Img baseline for comparing with Stage2/Stage3 trained models.
Supports single image (--input / --output) and batch (--input_dir / --output_dir).
After batch run, use --run_eval for Face-Sim / CLIP Score / FID (same as inference.py).
"""
import argparse
import os
import shutil
from pathlib import Path

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _list_images(dir_path: str):
    return sorted(
        [f for f in os.listdir(dir_path) if f.lower().endswith(IMG_EXTENSIONS)],
        key=lambda x: Path(x).stem,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Official FLUX.1-schnell Img2Img baseline.")
    p.add_argument("--input", type=str, default=None, help="Single: input image path.")
    p.add_argument("--output", type=str, default=None, help="Single: output image path.")
    p.add_argument("--input_dir", type=str, default=None, help="Batch: source image dir (use with --output_dir).")
    p.add_argument("--output_dir", type=str, default=None, help="Batch: output dir; creates real/ and gen/.")
    p.add_argument("--prompt", type=str, default="", help="Prompt text.")
    p.add_argument("--strength", type=float, default=0.6, help="Edit strength (0.0 unchanged, 1.0 full generation).")
    p.add_argument("--steps", type=int, default=8, help="Inference steps.")
    p.add_argument("--run_eval", action="store_true", help="Run Face-Sim / CLIP Score / FID after batch inference.")
    p.add_argument("--reference_dir", type=str, default=None, help="FID reference dir; omit to skip FID.")
    p.add_argument("--eval_output", type=str, default=None, help="Evaluation JSON path (e.g. baseline_metrics.json).")
    return p.parse_args()


def run_one(pipe, input_path, output_path, prompt, strength, num_inference_steps):
    init_image = Image.open(input_path).convert("RGB").resize((512, 512))
    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
    ).images[0]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path)


def main():
    args = parse_args()
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
        print(f"[Baseline] Loading FLUX.1-schnell, batch: {len(image_names)} images -> {args.output_dir}/real/ and .../gen/")

        pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        for name in tqdm(image_names, desc="Baseline"):
            inp_path = os.path.join(args.input_dir, name)
            stem = Path(name).stem
            ext = Path(name).suffix.lower()
            gen_path = os.path.join(gen_dir, stem + ".png")
            run_one(pipe, inp_path, gen_path, args.prompt, args.strength, args.steps)
            shutil.copy2(inp_path, os.path.join(real_dir, stem + ext))

        print(f"Baseline: {len(image_names)} images saved under {args.output_dir}/real/ and .../gen/")
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

    # Single image
    if not args.input or not args.output:
        print("Single-image mode: pass --input and --output. Batch mode: pass --input_dir and --output_dir.")
        return
    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}")
        return

    print("[Baseline] Loading FLUX.1-schnell (single image).")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    run_one(pipe, args.input, args.output, args.prompt, args.strength, args.steps)
    print(f"Official Baseline saved to: {args.output}")


if __name__ == "__main__":
    main()
