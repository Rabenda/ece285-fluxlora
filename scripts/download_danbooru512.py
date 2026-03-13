"""
Download Danbooru 512x512 anime face dataset from Kaggle and organize as cartoon/ for training.

Dataset: lukexng/animefaces-512x512
- ~140k 512x512 anime faces from Danbooru, portrait-cropped; suitable for StyleGAN/NovelAI/FLUX training.

Requires: Kaggle API configured (~/.kaggle/kaggle.json) and dataset license accepted on Kaggle.

Usage:
  python download_danbooru512.py --out_dir ./danbooru512
  python download_danbooru512.py --out_dir ./danbooru512 --num_images 2000
  python download_danbooru512.py --out_dir ./danbooru512 --no_download   # extract/organize only
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

KAGGLE_DATASET = "lukexng/animefaces-512x512"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Download Kaggle Danbooru 512×512 anime faces -> cartoon/.")
    p.add_argument("--out_dir", type=str, default="./danbooru512", help="Output root; creates cartoon/")
    p.add_argument("--num_images", type=int, default=None, help="Max images to keep; omit for all")
    p.add_argument("--size", type=int, default=512, help="Output size (default 512)")
    p.add_argument("--seed", type=int, default=42, help="Random seed when sampling num_images")
    p.add_argument("--no_download", action="store_true", help="Do not download; only extract and organize existing zip")
    return p.parse_args()


def main():
    args = parse_args()
    import random
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    cartoon_dir = out_dir / "cartoon"
    cartoon_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        print("Downloading from Kaggle (requires ~/.kaggle/kaggle.json and license accepted)...")
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(out_dir),
            ],
            check=True,
        )

    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No zip found in {out_dir}; run without --no_download to download first.")
    zip_path = zips[0]
    print(f"Using archive: {zip_path}")

    extract_marker = raw_dir / ".done"
    if not extract_marker.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        extract_marker.touch()

    def list_images(root: Path):
        return [
            f for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in IMG_EXT and not f.name.startswith("._")
            and "__MACOSX" not in f.parts
        ]

    files = list_images(raw_dir)
    if not files:
        raise ValueError(f"No images found under {raw_dir} after extract.")

    if args.num_images is not None and len(files) > args.num_images:
        random.shuffle(files)
        files = files[: args.num_images]

    for f in cartoon_dir.glob("*.png"):
        f.unlink()
    from PIL import Image
    from tqdm import tqdm
    success = 0
    for src in tqdm(files, desc="Organize cartoon"):
        try:
            img = Image.open(src).convert("RGB")
            if args.size != 512 or img.size != (512, 512):
                img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
            dst = cartoon_dir / f"{success:06d}.png"
            img.save(dst)
            success += 1
        except Exception as e:
            print(f"Skip {src}: {e}")

    n = success
    print(f"Done. cartoon: {n} images at {cartoon_dir.absolute()}")
    print("This is anime-face-only data; pair with real images (e.g. FFHQ) under the same train_dir:")
    print("  train_dir/real/   <- real faces")
    print("  train_dir/cartoon/ <- this script's cartoon/ (or symlink/copy danbooru512/cartoon there)")


if __name__ == "__main__":
    main()
