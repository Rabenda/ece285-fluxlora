"""
Stage2 / Stage3 训练脚本。Stage1 为仅推理基线，见 inference.py（不传 --checkpoint）。

支持两种模型架构（--mode）：
- full：I2I Rectified Flow（source/target、endpoint loss、t_stop 推理）
- lite：noise-conditioned flow（x_t = (1-t)*target + t*noise，t0 推理）

数据：--train_dir 下需有 real/ 与 cartoon/ 两个子目录。

# FULL 模式（推荐）
python train_cartoon.py --mode full --stage stage3 --train_dir ./data --checkpoint_root ./checkpoints --samples_dir ./samples

# LITE 模式
python train_cartoon.py --mode lite --stage stage3 --train_dir ./data --checkpoint_root ./checkpoints --samples_dir ./samples

# Stage2：仅 LoRA
python train_cartoon.py --mode full --stage stage2 --train_dir ./data --checkpoint_root ./checkpoints --samples_dir ./samples

# 续训 / 自检
python train_cartoon.py --stage stage3 --train_dir ./data --resume
python train_cartoon.py --stage stage3 --train_dir ./data --sanity_check
"""
import argparse
import csv
import glob
import os

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model_state_dict

from dataset import CartoonDataset
from identity_loss import FaceIdentityLoss
from inference import _run_one as run_inference_one
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
    parser.add_argument("--lr", type=float, default=5e-6, help="学习率；4bit+LoRA+accum 建议 5e-6，仍不稳可试 2e-6。")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_interval_step", type=int, default=50)
    parser.add_argument("--preview_interval_step", type=int, default=50, help="每 N 步存 interp_preview 与 source_preview。")
    parser.add_argument("--infer_interval_step", type=int, default=50, help="每 N 步存 compare（含 single/multi-step infer）与 infer_step_*.png。")
    parser.add_argument("--sample_t_stop", type=float, default=0.0, help="[FULL] multi-step 从 t=1 积分到 t_stop。")
    parser.add_argument("--sample_t0", type=float, default=0.5, help="[LITE] multi-step 起始 t0。")
    parser.add_argument("--sample_steps", type=int, default=20, help="infer 的 ODE 步数。")
    parser.add_argument("--sample_single_step_t", type=float, default=0.15, help="single-step probe 的步长，与 source_preview 语义一致。")
    parser.add_argument("--source_preview_step", type=float, default=0.15, help="source-end preview 步长，独立于 interpolation preview_t。")
    parser.add_argument("--noise_factor", type=float, default=0.02, help="起点 latent 噪声比例，与训练一致。")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--gamma_start", type=float, default=0.0, help="gamma 调度起点；新实验 A 建议 0.0。")
    parser.add_argument("--gamma_end", type=float, default=0.3, help="gamma 调度终点；新实验 A 建议 0.3。")
    parser.add_argument("--gamma_warmup_steps", type=int, default=300, help="gamma warmup 步数。")
    parser.add_argument("--mix_cartoon_latent", type=float, default=0.6, help="target 中 cartoon 混合比例；0.6 让 target 更明确，利于稳定。")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "lite"], help="FluxI2ITrainable: full=I2I Rectified Flow, lite=noise-conditioned flow.")
    parser.add_argument("--a1", type=float, default=0.1, help="[FULL] 条件缩放 a1（vlm_proj），LITE 忽略。")
    parser.add_argument("--a2", type=float, default=0.05, help="[FULL] 条件缩放 a2（pooled_proj），LITE 忽略。")
    parser.add_argument("--stage", type=str, default="stage3", choices=["stage2", "stage3"], help="stage2: LoRA only; stage3: LoRA + identity loss.")
    parser.add_argument("--lambda0", type=float, default=0.6, help="Identity loss base weight in stage3.")
    parser.add_argument("--lambda_p", type=float, default=2.0, help="Time-dependent exponent in stage3.")
    parser.add_argument("--use_autocast", action="store_true")
    parser.add_argument("--sanity_check", action="store_true", help="训练前跑一次环境/模型自检；未通过则退出。默认关闭。")
    parser.add_argument("--train_projection", action="store_true", help="若指定则同时训练 vlm_proj/pooled_proj；默认仅训 LoRA 更稳。")
    parser.add_argument("--no_pbar", action="store_true", help="关闭 tqdm 进度条，减少 I/O 阻塞（后台/长时间跑时建议加）。")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ensure_dir(args.samples_dir)
    ensure_dir(args.checkpoint_root)

    csv_path = os.path.join(args.checkpoint_root, "train_metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "rf_loss", "id_loss", "total_loss", "lam", "gamma", "lr", "grad_norm", "v_pred_absmax", "target_v_absmax", "v_src_absmax", "lora_diff_mean", "lora_grad_mean"])

    spike_log_path = os.path.join(args.checkpoint_root, "spike_log.csv")
    if not os.path.exists(spike_log_path):
        with open(spike_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "batch_idx", "rf_loss", "v_pred_absmax", "target_v_absmax", "t_mean", "trigger", "grad_norm", "vlm_proj_out_absmax", "enc_hidden_absmax", "pooled_proj_absmax"])

    def _write_spike_row(info, step, batch_idx, grad_norm_str=""):
        rf = info.get("rf_loss")
        rf_str = "nan" if (isinstance(rf, float) and not (rf == rf)) else (round(rf, 6) if isinstance(rf, (int, float)) else rf)
        row = [
            step, batch_idx,
            rf_str,
            round(info.get("v_pred_absmax", 0), 6),
            round(info.get("target_v_absmax", 0), 6),
            round(info.get("t_mean", 0), 6),
            info.get("trigger", "loss"),
            grad_norm_str,
        ]
        for k in ("vlm_proj_out_absmax", "enc_hidden_absmax", "pooled_proj_absmax"):
            v = info.get(k)
            row.append(round(v, 6) if isinstance(v, (int, float)) and (v == v) else ("" if v is None else "nan"))
        with open(spike_log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    print(f"Loading model for {args.stage} (mode={args.mode}) ...")
    model = FluxI2ITrainable(mode=args.mode).to(device)
    model.setup_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    if not args.train_projection:
        model.vlm_proj.requires_grad_(False)
        model.pooled_proj.requires_grad_(False)
        print("Frozen vlm_proj and pooled_proj (train LoRA only). Use --train_projection to unfreeze.")

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
                if "vlm_proj" in ckpt:
                    model.vlm_proj.load_state_dict(ckpt["vlm_proj"])
                if "pooled_proj" in ckpt:
                    model.pooled_proj.load_state_dict(ckpt["pooled_proj"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                global_step = ckpt.get("global_step", 0)
                ckpt_mode = ckpt.get("mode", None)
                if ckpt_mode is not None and ckpt_mode != args.mode:
                    print(f"[Resume] Checkpoint mode={ckpt_mode} overrides --mode={args.mode}. Using mode={ckpt_mode} for training and saves.")
                    args.mode = ckpt_mode
                    model._mode = str(ckpt_mode).lower()

    # 保存当前 LoRA 状态作为“初始”参考（resume 时则为“本次运行起点”），用于 diff 诊断
    init_lora_state = {}
    for name, p in model.named_parameters():
        if p.requires_grad and "lora" in name.lower():
            init_lora_state[name] = p.detach().float().cpu().clone()
    probe_lora_name = next(iter(init_lora_state), None) if init_lora_state else None
    if probe_lora_name is not None:
        print(f"[LoRA debug] Saved init state for {len(init_lora_state)} LoRA params; probe param: {probe_lora_name}")

    accum_steps = max(1, args.gradient_accumulation_steps)
    print(f"Training on {device}, autocast={args.use_autocast}, gradient_accumulation_steps={accum_steps} (effective batch = {args.batch_size * accum_steps})")
    last_rf, last_id, last_lam, last_total = "0.0000", "0.0000", "0.0000", "0.0000"
    rf_acc = id_acc = total_acc = lam_acc = gamma_acc = count_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        step_in_accum = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", mininterval=10.0, disable=args.no_pbar)
        for batch in pbar:
            if step_in_accum == 0:
                optimizer.zero_grad(set_to_none=True)
                rf_acc = id_acc = total_acc = lam_acc = gamma_acc = count_acc = 0.0

            real_imgs = batch["real"].to(device, non_blocking=True)
            cartoon_imgs = batch["cartoon"].to(device, non_blocking=True)
            gamma = gamma_schedule(global_step, args.gamma_start, args.gamma_end, args.gamma_warmup_steps)

            # Stage2/3: condition on real, target latent toward cartoon
            cond_image = real_imgs
            target_image = cartoon_imgs

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.use_autocast):
                if args.stage == "stage3":
                    rf_loss, preview, aux = model(
                        target_image=target_image,
                        cond_image=cond_image,
                        gamma=gamma,
                        mix_cartoon_latent=args.mix_cartoon_latent,
                        return_aux=True,
                        noise_factor=args.noise_factor,
                        source_preview_step=args.source_preview_step,
                        a1=args.a1,
                        a2=args.a2,
                    )
                else:
                    rf_loss, preview, aux = model(
                        target_image=target_image,
                        cond_image=cond_image,
                        gamma=gamma,
                        mix_cartoon_latent=args.mix_cartoon_latent,
                        return_aux=True,
                        noise_factor=args.noise_factor,
                        source_preview_step=args.source_preview_step,
                        a1=args.a1,
                        a2=args.a2,
                    )

                if rf_loss is None or preview is None or (not torch.isfinite(rf_loss)):
                    spike_info = getattr(model, "_last_spike_info", None)
                    if spike_info is not None:
                        _write_spike_row(spike_info, global_step, step_in_accum, "")
                        model._last_spike_info = None
                    continue

                # 用于 CSV 与每 50 step 打印：输出侧诊断（interpolation + source-end 两支）
                last_v_pred_absmax = aux.get("v_pred_absmax")
                last_target_v_absmax = aux.get("target_v_absmax")
                last_v_src_absmax = aux.get("v_src_absmax")

                total_loss = rf_loss
                id_loss = torch.zeros((), device=device)
                lam = 0.0
                if args.stage == "stage3":
                    sampled_t = float(aux["sampled_t"].mean().detach().item())
                    lam = lambda_schedule(sampled_t, args.lambda0, args.lambda_p)
                    with torch.amp.autocast("cuda", enabled=False):
                        # 关键：ID Loss 必须算在 source_preview 上，强迫模型在真实推理起点保持五官
                        id_loss = id_criterion(real_imgs.float(), aux["source_preview"].float())
                    total_loss = rf_loss + lam * id_loss

            if not torch.isfinite(total_loss):
                continue

            spike_info = getattr(model, "_last_spike_info", None)
            if spike_info is not None:
                _write_spike_row(spike_info, global_step, step_in_accum, "")
                model._last_spike_info = None

            (total_loss / accum_steps).backward()
            step_in_accum += 1
            rf_acc += rf_loss.item()
            id_acc += id_loss.item()
            total_acc += total_loss.item()
            lam_acc += lam
            gamma_acc += gamma
            count_acc += 1.0

            if step_in_accum == accum_steps:
                if args.stage == "stage3" and global_step < 3:
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            print(f"[Stage3 grad check] {name} grad mean: {param.grad.abs().mean().item():.6f}")
                            break
                n = max(1, int(count_acc))
                rf_avg = rf_acc / n
                id_avg = id_acc / n
                total_avg = total_acc / n
                lam_avg = lam_acc / n
                gamma_avg = gamma_acc / n
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                if grad_norm.item() > 10000.0:
                    with open(spike_log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step + 1,
                            accum_steps - 1,
                            round(rf_avg, 6),
                            -1, -1, -1,
                            "grad_norm",
                            round(grad_norm.item(), 6),
                            "", "", "",
                        ])

                # LoRA 诊断：在 optimizer.step() 之前读取 grad，保证是“本步更新前”的梯度
                lora_diff_mean = ""
                lora_grad_mean = ""
                if probe_lora_name is not None:
                    p = dict(model.named_parameters())[probe_lora_name]
                    cur = p.detach().float().cpu()
                    init = init_lora_state[probe_lora_name]
                    lora_diff_mean = (cur - init).abs().mean().item()
                    lora_grad_mean = p.grad.detach().float().abs().mean().item() if p.grad is not None else 0.0

                optimizer.step()
                step_in_accum = 0
                global_step += 1

                if global_step % 50 == 0:
                    vp = (last_v_pred_absmax if last_v_pred_absmax is not None else 0)
                    tv = (last_target_v_absmax if last_target_v_absmax is not None else 0)
                    vs = (last_v_src_absmax if last_v_src_absmax is not None else 0)
                    if probe_lora_name is not None:
                        print(f"[debug] step={global_step} v_pred_absmax={vp:.4f} v_src_absmax={vs:.4f} target_v_absmax={tv:.4f} | [LoRA] {probe_lora_name} diff_mean={lora_diff_mean:.6e} grad_mean={lora_grad_mean:.6e}")
                    else:
                        print(f"[debug] step={global_step} v_pred_absmax={vp:.4f} v_src_absmax={vs:.4f} target_v_absmax={tv:.4f}")

                v_pred_str = round(last_v_pred_absmax, 6) if last_v_pred_absmax is not None else ""
                target_str = round(last_target_v_absmax, 6) if last_target_v_absmax is not None else ""
                v_src_str = round(last_v_src_absmax, 6) if last_v_src_absmax is not None else ""
                lora_diff_str = round(lora_diff_mean, 8) if isinstance(lora_diff_mean, (int, float)) else lora_diff_mean
                lora_grad_str = round(lora_grad_mean, 8) if isinstance(lora_grad_mean, (int, float)) else lora_grad_mean

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        global_step,
                        epoch + 1,
                        round(rf_avg, 6),
                        round(id_avg, 6),
                        round(total_avg, 6),
                        round(lam_avg, 6),
                        round(gamma_avg, 6),
                        optimizer.param_groups[0]["lr"],
                        round(grad_norm.item(), 6),
                        v_pred_str,
                        target_str,
                        v_src_str,
                        lora_diff_str,
                        lora_grad_str,
                    ])

                # 1) Training preview：interp_preview 与 source_preview 分开存，文件名明确
                if global_step % args.preview_interval_step == 0:
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.use_autocast):
                        _, preview, aux = model(
                            target_image=fixed_cartoon, cond_image=fixed_real, gamma=gamma,
                            mix_cartoon_latent=args.mix_cartoon_latent, return_aux=True,
                            noise_factor=args.noise_factor, source_preview_step=args.source_preview_step,
                            a1=args.a1, a2=args.a2,
                        )
                        if preview is not None:
                            save_preview_images(preview, os.path.join(args.samples_dir, f"interp_preview_step_{global_step:06d}.png"))
                        if "source_preview" in aux:
                            save_preview_images(aux["source_preview"], os.path.join(args.samples_dir, f"source_preview_step_{global_step:06d}.png"))
                    model.train()

                # 2) 真实推理 + 对比图（real | interp_preview | source_preview | infer_single | infer_multi）
                if global_step > 0 and global_step % args.infer_interval_step == 0:
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.use_autocast):
                        _, preview, aux = model(
                            target_image=fixed_cartoon, cond_image=fixed_real, gamma=gamma,
                            mix_cartoon_latent=args.mix_cartoon_latent, return_aux=True,
                            noise_factor=args.noise_factor, source_preview_step=args.source_preview_step,
                            a1=args.a1, a2=args.a2,
                        )
                        _infer_args = argparse.Namespace(
                            steps=args.sample_steps,
                            t_stop=args.sample_t_stop,
                            t0=args.sample_t0,
                            noise_factor=args.noise_factor,
                            cond_scale_vlm=args.a1,
                            cond_scale_pooled=args.a2,
                            single_step_t=args.sample_single_step_t,
                        )
                        _infer_args.single_step = True
                        infer_single = run_inference_one(model, device, fixed_real[0:1], _infer_args)
                        _infer_args.single_step = False
                        infer_multi = run_inference_one(model, device, fixed_real[0:1], _infer_args)
                    real_01 = (fixed_real[0].float().cpu() + 1.0) / 2.0
                    interp_preview_01 = (preview[0].float().cpu().clamp(-1, 1) + 1.0) / 2.0 if preview is not None else real_01
                    source_preview_01 = (aux["source_preview"][0].float().cpu().clamp(-1, 1) + 1.0) / 2.0 if "source_preview" in aux else interp_preview_01
                    infer_single_01 = infer_single  # (C,H,W) already [0,1]
                    infer_multi_01 = infer_multi
                    compare = torch.stack([real_01, interp_preview_01, source_preview_01, infer_single_01, infer_multi_01], dim=0)
                    vutils.save_image(compare, os.path.join(args.samples_dir, f"compare_step_{global_step:06d}.png"), nrow=5)
                    vutils.save_image(infer_multi.unsqueeze(0), os.path.join(args.samples_dir, f"infer_step_{global_step:06d}.png"))
                    vutils.save_image(infer_single.unsqueeze(0), os.path.join(args.samples_dir, f"infer_single_step_{global_step:06d}.png"))
                    model.train()

                if global_step > 0 and global_step % args.save_interval_step == 0:
                    ckpt_path = os.path.join(args.checkpoint_root, f"full_step_{global_step:06d}.pt")
                    lora_state = get_peft_model_state_dict(model.unet)
                    torch.save(
                        {
                            "global_step": global_step,
                            "lora": lora_state,
                            "vlm_proj": model.vlm_proj.state_dict(),
                            "pooled_proj": model.pooled_proj.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "stage": args.stage,
                            "mode": args.mode,
                            "lambda0": args.lambda0,
                            "lambda_p": args.lambda_p,
                        },
                        ckpt_path,
                    )

                last_rf = f"{rf_avg:.4f}"
                last_id = f"{id_avg:.4f}"
                last_lam = f"{lam_avg:.4f}"
                last_total = f"{total_avg:.4f}"
                pbar.set_postfix(rf=last_rf, id=last_id, lam=last_lam, total=last_total, step=global_step, accum=f"0/{accum_steps}")
            else:
                pbar.set_postfix(rf=last_rf, id=last_id, lam=last_lam, total=last_total, step=global_step, accum=f"{step_in_accum}/{accum_steps}")

        if step_in_accum > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            n = max(1, int(count_acc))
            rf_avg = rf_acc / n
            id_avg = id_acc / n
            total_avg = total_acc / n
            lam_avg = lam_acc / n
            gamma_avg = gamma_acc / n
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step,
                    epoch + 1,
                    round(rf_avg, 6),
                    round(id_avg, 6),
                    round(total_avg, 6),
                    round(lam_avg, 6),
                    round(gamma_avg, 6),
                    optimizer.param_groups[0]["lr"],
                    round(grad_norm.item(), 6),
                    "",
                    "",
                    "",
                    "",
                    "",
                ])

    # 训练结束保存最终 checkpoint，避免最后一轮未对齐 save_interval 而漏存
    final_ckpt_path = os.path.join(args.checkpoint_root, f"full_step_{global_step:06d}_final.pt")
    lora_state = get_peft_model_state_dict(model.unet)
    torch.save(
        {
            "global_step": global_step,
            "lora": lora_state,
            "vlm_proj": model.vlm_proj.state_dict(),
            "pooled_proj": model.pooled_proj.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stage": args.stage,
            "mode": args.mode,
            "lambda0": args.lambda0,
            "lambda_p": args.lambda_p,
        },
        final_ckpt_path,
    )
    print(f"Final checkpoint saved to: {final_ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()