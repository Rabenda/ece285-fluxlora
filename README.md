# ECE 285: Real-to-Cartoon Image-to-Image with FLUX and LoRA

Image-only, identity-aware cartoonization using FLUX.1-schnell with LoRA. No text prompts at inference; the model is conditioned solely on the input image and learns the cartoon style via flow matching (Rectified Flow or noise-conditioned flow).

## Overview

- **Goal**: Real portrait → cartoon portrait using only the input image (no captions or style prompts at test time).
- **Pipeline**: Train LoRA on FLUX with flow loss (and optional identity loss); inference runs the same image-conditioned ODE without any text.
- **Stages**:
  - **Stage 1 (baseline)**: Pretrained FLUX + zero-initialized LoRA, no training. Image-only inference does not produce cartoon style.
  - **Stage 2**: LoRA trained with flow loss only. Strong style, possible identity drift.
  - **Stage 3**: LoRA + identity loss (FaceNet). Balances style and face preservation.

## Project structure

```
project/
├── train_cartoon.py       # Training entry (Stage2/Stage3)
├── inference.py           # Unified inference (Stage1 without ckpt, Stage2/3 with ckpt)
├── evaluate.py            # Metrics: Face-Sim, CLIP Score, FID
├── dataset.py             # CartoonDataset (real/ + cartoon/ under --train_dir)
├── identity_loss.py       # Stage3 face identity loss (FaceNet)
├── models/
│   └── flux_i2i_trainable.py   # FLUX + LoRA, full/lite flow modes
├── scripts/
│   ├── download_stable_faces.py
│   ├── preprocess_real_cartoon.py
│   └── ...
└── README.md
```

## Data

Under `--train_dir` you need:

- `real/` — real face images
- `cartoon/` — cartoon/style reference images

Optionally `pairs.csv` (columns `real`, `cartoon`) for paired data; otherwise the dataset uses unpaired sampling. You can use `scripts/download_stable_faces.py` (or similar) to prepare data.

## Training

Two flow modes:

- **`--mode full`** (default): I2I Rectified Flow (source↔target, endpoint loss). Inference from t=1 to `--t_stop` with condition scales a1/a2.
- **`--mode lite`**: Noise-conditioned flow (x_t = (1-t)*target + t*noise). Inference from `--t0` with fixed 0.01 conditioning.

**Examples:**

```bash
# Stage2 (LoRA only), FULL mode
python train_cartoon.py --mode full --stage stage2 --train_dir ./data --checkpoint_root ./checkpoints --samples_dir ./samples

# Stage3 (LoRA + identity loss), FULL mode
python train_cartoon.py --mode full --stage stage3 --train_dir ./data --checkpoint_root ./checkpoints --samples_dir ./samples

# Stage3, LITE mode
python train_cartoon.py --mode lite --stage stage3 --train_dir ./data --checkpoint_root ./checkpoints --samples_dir ./samples

# Resume (mode is read from checkpoint if present)
python train_cartoon.py --stage stage3 --train_dir ./data --resume

# Sanity check before training
python train_cartoon.py --stage stage3 --train_dir ./data --sanity_check
```

Checkpoints are saved under `checkpoint_root` as `full_step_*.pt` (and `*_final.pt`). Each includes `lora`, `vlm_proj`, `pooled_proj`, `mode`, and optimizer state.

## Inference

- **Stage 1**: Run without `--checkpoint` to use pretrained FLUX + zero LoRA (baseline; output is not cartoon).
- **Stage 2/3**: Pass `--checkpoint path/to/full_step_*.pt`. Mode (full/lite) is taken from the checkpoint; if missing, it is inferred from the path (e.g. `lite` in path → LITE).

**Single image:**

```bash
python inference.py --checkpoint ./checkpoints/full_step_000500.pt --input ./data/real/photo.jpg --output ./out.png
```

**Batch + organize + evaluation:**

```bash
python inference.py --checkpoint ./checkpoints/full_step_000500.pt \
  --input_dir ./data/real --output_dir ./results \
  --organize --run_eval --reference_dir ./data/cartoon --eval_output ./metrics.json
```

FULL mode uses `--t_stop`, `--steps`, `--cond_scale_vlm`/`--cond_scale_pooled`. LITE mode uses `--t0`, `--steps`. Single-image eval skips FID (needs ≥2 generated images).

## Evaluation

Face-Sim (identity), CLIP Score (style vs. text prompt), FID (distribution vs. reference set, requires ≥2 generated images and `--reference_dir`).

**Standalone:**

```bash
python evaluate.py --real_dir ./data/real --gen_dir ./results/gen --reference_dir ./data/cartoon --output metrics.json
```

**From code:** `from evaluate import run_evaluation; run_evaluation(real_dir=..., gen_dir=..., reference_dir=..., output_path=...)`

Optional deps: `pip install facenet-pytorch` (Face-Sim), `pip install torchmetrics[image]` (FID).

## Requirements

- PyTorch, diffusers, transformers, peft
- Optional: facenet-pytorch, torchmetrics[image]

Check `environment.yml` or project docs for versions.

## License

See repository root.
