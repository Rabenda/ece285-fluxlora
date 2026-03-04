"""
Stage2 / Stage3 训练脚本。Stage1 为仅推理基线，见 inference.py（不传 --checkpoint）。

数据：--train_dir 下需有 real/ 与 cartoon/ 两个子目录（可用 scripts/download_stable_faces.py 准备）。
训练完成后用 inference.py 加载 checkpoint 做推理；其他同学只需跑本脚本 + inference.py 即可复现。

# Stage2：仅 LoRA（流损失）
python train_cartoon.py --stage stage2 --train_dir ./data --checkpoint_root ./checkpoints --samples_dir ./samples

# Stage3：LoRA + 身份损失（推荐）
python train_cartoon.py --stage stage3 --train_dir ./data --checkpoint_root ./checkpoints --samples_dir ./samples

# 从已有 checkpoint 续训
python train_cartoon.py --stage stage3 --train_dir ./data --resume

# 训练前做环境自检（可选）
python train_cartoon.py --stage stage3 --train_dir ./data --sanity_check
"""
import argparse
import glob
import os

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model_state_dict

from dataset import CartoonDataset
from identity_loss import FaceIdentityLoss
from models.flux_i2i_trainable import FluxI2ITrainable


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def latest_full_ckpt_path(root):
    paths = sorted(glob.glob(os.path.join(root, "full_step_*.pt")))
    return paths[-1] if paths else None


def lambda_schedule(t_scalar, lambda0, p):
    return float(lambda0 * ((1.0 - t_scalar) ** p))


def gamma_schedule(global_step, start, end, warmup_steps):
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return end
    return start + (end - start) * (global_step / float(warmup_steps))


@torch.no_grad()
def save_preview_images(generated_images, save_path):
    img = generated_images[0].detach().float().cpu()
    img = torch.nan_to_num(img, nan=0.0).clamp(-1, 1)
    vutils.save_image((img + 1) / 2, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Identity-aware LoRA training on FLUX.")
    parser.add_argument("--train_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--samples_dir", type=str, default="./samples")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="每 N 個 micro-batch 累積梯度後更新一次，等效 batch_size = batch_size * N。")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_interval_step", type=int, default=500)
    parser.add_argument("--sample_interval_step", type=int, default=20)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--gamma_start", type=float, default=2.0)
    parser.add_argument("--gamma_end", type=float, default=2.0)
    parser.add_argument("--gamma_warmup_steps", type=int, default=0)
    parser.add_argument("--stage", type=str, default="stage3", choices=["stage2", "stage3"], help="stage2: LoRA only; stage3: LoRA + identity loss.")
    parser.add_argument("--lambda0", type=float, default=0.6, help="Identity loss base weight in stage3.")
    parser.add_argument("--lambda_p", type=float, default=2.0, help="Time-dependent exponent in stage3.")
    parser.add_argument("--use_autocast", action="store_true")
    parser.add_argument("--sanity_check", action="store_true", help="训练前跑一次环境/模型自检；未通过则退出。默认关闭。")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ensure_dir(args.samples_dir)
    ensure_dir(args.checkpoint_root)

    print(f"Loading model for {args.stage} ...")
    model = FluxI2ITrainable().to(device)
    model.setup_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    # 可选：环境自检，未通过则直接退出
    if args.sanity_check:
        from sanity_check import run_sanity_check
        if not run_sanity_check(device, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, verbose=True):
            print("Training aborted: sanity check failed. Please fix environment (e.g. CUDA, VRAM, dependencies).")
            return
        print()

    id_criterion = None
    if args.stage == "stage3":
        id_criterion = FaceIdentityLoss(device=device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    dataset = CartoonDataset(args.train_dir)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Ensure real/ and cartoon/ each contain at least one image.")
    if len(dataset) < args.batch_size:
        raise ValueError(
            f"Not enough samples for one batch (len(dataset)={len(dataset)}, batch_size={args.batch_size}). "
            "Use a larger dataset or smaller batch_size."
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    fixed_batch = next(iter(dataloader))
    fixed_real = fixed_batch["real"].to(device)
    fixed_cartoon = fixed_batch["cartoon"].to(device)

    global_step = 0
    if args.resume:
        ckpt_path = latest_full_ckpt_path(args.checkpoint_root)
        if ckpt_path:
            print(f"Resuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                # 新格式：僅保存 LoRA 權重在 "lora" key 下
                if "lora" in ckpt:
                    model.unet.load_state_dict(ckpt["lora"], strict=False)
                # 向後相容舊格式："model_state_dict"
                elif "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                global_step = ckpt.get("global_step", 0)

    accum_steps = max(1, args.gradient_accumulation_steps)
    print(f"Training on {device}, autocast={args.use_autocast}, gradient_accumulation_steps={accum_steps} (effective batch = {args.batch_size * accum_steps})")
    last_rf, last_id, last_lam, last_total = "0.0000", "0.0000", "0.0000", "0.0000"
    for epoch in range(args.epochs):
        model.train()
        step_in_accum = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            if step_in_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            real_imgs = batch["real"].to(device, non_blocking=True)
            cartoon_imgs = batch["cartoon"].to(device, non_blocking=True)
            gamma = gamma_schedule(global_step, args.gamma_start, args.gamma_end, args.gamma_warmup_steps)

            # Stage2/3: condition on real, target latent toward cartoon
            cond_image = real_imgs
            target_image = cartoon_imgs

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_autocast):
                if args.stage == "stage3":
                    rf_loss, preview, aux = model(
                        target_image=target_image,
                        cond_image=cond_image,
                        gamma=gamma,
                        return_aux=True,
                    )
                else:
                    rf_loss, preview = model(
                        target_image=target_image,
                        cond_image=cond_image,
                        gamma=gamma,
                        return_aux=False,
                    )
                    aux = None

                if rf_loss is None or preview is None or (not torch.isfinite(rf_loss)):
                    continue

                total_loss = rf_loss
                id_loss = torch.zeros((), device=device)
                lam = 0.0
                if args.stage == "stage3":
                    sampled_t = float(aux["sampled_t"].mean().detach().item())
                    lam = lambda_schedule(sampled_t, args.lambda0, args.lambda_p)
                    with torch.cuda.amp.autocast(enabled=False):
                        id_loss = id_criterion(real_imgs.float(), preview.float())
                    total_loss = rf_loss + lam * id_loss

            if not torch.isfinite(total_loss):
                continue

            (total_loss / accum_steps).backward()
            step_in_accum += 1

            if step_in_accum == accum_steps:
                if args.stage == "stage3" and global_step < 3:
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            print(f"[Stage3 grad check] {name} grad mean: {param.grad.abs().mean().item():.6f}")
                            break
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                step_in_accum = 0
                global_step += 1

                if global_step % args.sample_interval_step == 0:
                    model.eval()
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_autocast):
                        _, preview = model(target_image=fixed_cartoon, cond_image=fixed_real, gamma=gamma)
                        if preview is not None:
                            save_preview_images(preview, os.path.join(args.samples_dir, f"step_{global_step:06d}.png"))
                    model.train()

                if global_step > 0 and global_step % args.save_interval_step == 0:
                    ckpt_path = os.path.join(args.checkpoint_root, f"full_step_{global_step:06d}.pt")
                    lora_state = get_peft_model_state_dict(model.unet)
                    torch.save(
                        {
                            "global_step": global_step,
                            "lora": lora_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "stage": args.stage,
                            "lambda0": args.lambda0,
                            "lambda_p": args.lambda_p,
                        },
                        ckpt_path,
                    )

                last_rf = f"{rf_loss.item():.4f}"
                last_id = f"{id_loss.item():.4f}"
                last_lam = f"{lam:.4f}"
                last_total = f"{total_loss.item():.4f}"
                pbar.set_postfix(rf=last_rf, id=last_id, lam=last_lam, total=last_total, step=global_step, accum=f"0/{accum_steps}")
            else:
                pbar.set_postfix(rf=last_rf, id=last_id, lam=last_lam, total=last_total, step=global_step, accum=f"{step_in_accum}/{accum_steps}")

        if step_in_accum > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    print("Training finished.")


if __name__ == "__main__":
    main()