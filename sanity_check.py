"""
环境/模型自检：在训练前快速跑一次 forward，确认能跑通且无 NaN。
供 train_cartoon.py 通过 --sanity_check 调用，默认关闭。
"""
import gc
import torch

from models.flux_i2i_trainable import FluxI2ITrainable


def _print_vram(label: str) -> None:
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        print(f"  [{label}] VRAM Allocated: {a:.2f} GB | Reserved: {r:.2f} GB")


def run_sanity_check(
    device: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    size: int = 512,
    verbose: bool = True,
) -> bool:
    """
    加载模型、跑一次 dummy forward，检查 loss/preview 是否有限。
    返回 True 表示通过，False 表示不通过（应终止训练并提示用户）。
    """
    if verbose:
        print("Running sanity check (model load + one forward pass)...")
    if device == "cpu":
        if verbose:
            print("  Warning: device is CPU; check may be slow.")
    else:
        torch.cuda.empty_cache()
        gc.collect()
        if verbose:
            _print_vram("Before model")

    try:
        model = FluxI2ITrainable().to(device)
        model.setup_lora(rank=lora_rank, alpha=lora_alpha)
        model.vae.to(torch.float32)
        model.eval()
    except Exception as e:
        if verbose:
            print(f"  Sanity check failed: model init error: {e}")
        return False

    if verbose and device != "cpu":
        _print_vram("Model loaded")

    # Dummy data: (1, 3, H, W) in [-1, 1], 与训练时一致
    dummy_real = torch.rand(1, 3, size, size, device=device, dtype=torch.float32) * 2.0 - 1.0
    dummy_cartoon = torch.rand(1, 3, size, size, device=device, dtype=torch.float32) * 2.0 - 1.0

    try:
        with torch.no_grad():
            loss, preview, aux = model(
                target_image=dummy_cartoon,
                cond_image=dummy_real,
                gamma=2.0,
                return_aux=True,
            )
    except Exception as e:
        if verbose:
            print(f"  Sanity check failed: forward error: {e}")
        return False

    ok = True
    if loss is None or not torch.isfinite(loss).all():
        if verbose:
            print(f"  Sanity check failed: loss is None or not finite (got {loss})")
        ok = False
    if preview is None or torch.isnan(preview).any() or torch.isinf(preview).any():
        if verbose:
            print("  Sanity check failed: preview is None or contains NaN/Inf.")
        ok = False

    del model, dummy_real, dummy_cartoon, loss, preview, aux
    if device != "cpu":
        torch.cuda.empty_cache()
        gc.collect()

    if ok and verbose:
        print("  Sanity check passed. Environment is OK for training.")
    return ok
