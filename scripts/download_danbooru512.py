"""
从 Kaggle 下载 Danbooru 512×512 动漫脸数据集，整理为 cartoon/ 供训练使用。

数据集：lukexng/animefaces-512x512
- 约 14 万张 512×512 动漫脸，来源 Danbooru，已 portrait crop，适合 StyleGAN/NovelAI/FLUX 等训练。

前置：已配置 Kaggle API（~/.kaggle/kaggle.json），且需在 Kaggle 页面接受该数据集许可。

用法：
  python download_danbooru512.py --out_dir ./danbooru512
  python download_danbooru512.py --out_dir ./danbooru512 --num_images 2000
  python download_danbooru512.py --out_dir ./danbooru512 --no_download   # 仅解压整理
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

KAGGLE_DATASET = "lukexng/animefaces-512x512"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Download Kaggle Danbooru 512×512 anime faces -> cartoon/.")
    p.add_argument("--out_dir", type=str, default="./danbooru512", help="输出根目录，将创建 cartoon/")
    p.add_argument("--num_images", type=int, default=None, help="最多保留张数，不指定则全部保留")
    p.add_argument("--size", type=int, default=512, help="输出边长（默认 512，保持原样）")
    p.add_argument("--seed", type=int, default=42, help="随机种子，用于 num_images 时抽样")
    p.add_argument("--no_download", action="store_true", help="不下载，仅对已有 zip 解压并整理")
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
        print("正在从 Kaggle 下载（需已配置 ~/.kaggle/kaggle.json 并在网页接受许可）...")
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
        raise FileNotFoundError(f"在 {out_dir} 下未找到 zip，请先下载（去掉 --no_download）")
    zip_path = zips[0]
    print(f"使用压缩包: {zip_path}")

    extract_marker = raw_dir / ".done"
    if not extract_marker.exists():
        print("解压中...")
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
        raise ValueError(f"解压后未在 {raw_dir} 下找到图片")

    if args.num_images is not None and len(files) > args.num_images:
        random.shuffle(files)
        files = files[: args.num_images]

    for f in cartoon_dir.glob("*.png"):
        f.unlink()
    from PIL import Image
    from tqdm import tqdm
    success = 0
    for src in tqdm(files, desc="整理 cartoon"):
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
    print(f"完成。cartoon: {n} 张，路径: {cartoon_dir.absolute()}")
    print("说明：此为纯动漫脸数据，训练时需搭配 real 图（如 FFHQ / face2anime 的 real），将本目录与 real 目录放在同一 train_dir 下，例如：")
    print("  train_dir/real/   <- 真人图")
    print("  train_dir/cartoon/ <- 本脚本输出的 cartoon/（或把 danbooru512/cartoon 软链/复制到 train_dir/cartoon）")


if __name__ == "__main__":
    main()
