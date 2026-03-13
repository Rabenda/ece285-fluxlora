"""
Read train_metrics.csv and plot loss / parameter curves.

Usage (from project root):
  python scripts/plot_train_metrics.py
  python scripts/plot_train_metrics.py --csv ./checkpoints/train_metrics.csv --output ./checkpoints/train_curves.png
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Plot training metrics from train_metrics.csv")
    p.add_argument("--csv", type=str, default="./checkpoints/train_metrics.csv", help="Path to CSV.")
    p.add_argument("--output", type=str, default=None, help="Path to save figure; omit to show only.")
    p.add_argument("--no-show", action="store_true", help="Do not show window; save only (for headless).")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if df.empty:
        print("CSV is empty.")
        sys.exit(0)

    step = df["step"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Losses
    ax = axes[0, 0]
    ax.plot(step, df["rf_loss"], label="rf_loss", alpha=0.9)
    ax.plot(step, df["id_loss"], label="id_loss", alpha=0.9)
    ax.plot(step, df["total_loss"], label="total_loss", alpha=0.9)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Loss")

    # 2) lam, gamma
    ax = axes[0, 1]
    ax.plot(step, df["lam"], label="lam", alpha=0.9)
    ax.plot(step, df["gamma"], label="gamma", alpha=0.9)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("lam / gamma")

    # 3) lr
    ax = axes[1, 0]
    ax.plot(step, df["lr"], color="green", alpha=0.9)
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.grid(True, alpha=0.3)
    ax.set_title("Learning rate")

    # 4) grad_norm
    ax = axes[1, 1]
    ax.plot(step, df["grad_norm"], color="purple", alpha=0.9)
    ax.set_xlabel("step")
    ax.set_ylabel("grad_norm")
    ax.grid(True, alpha=0.3)
    ax.set_title("Grad norm (before clip)")

    plt.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
