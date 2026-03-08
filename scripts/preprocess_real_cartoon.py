"""
FFHQ + 动漫脸数据预处理：统一尺寸、过滤、重命名，输出 real/、cartoon/、pairs.csv。
训练时 --train_dir 指向 --out_dir 即可。

动漫侧增强（可选）：
- --cascade：lbpcascade_animeface.xml 路径，启用后人脸检测，以脸为中心裁剪，未检测到脸则丢弃。
  wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
- --min_saturation：饱和度下限，过滤线稿/灰度图（默认 20，0 表示不启用）。
- --min_laplacian_var：Laplacian 方差下限，过滤模糊/强行放大的图（默认 0 表示不启用）。
使用 --cascade 或 --min_laplacian_var 时需安装：pip install numpy opencv-python
"""
import argparse
import csv
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess FFHQ + Danbooru face datasets.")
    p.add_argument("--real_src", type=str, required=True, help="Source dir for real faces (e.g. FFHQ subset)")
    p.add_argument("--cartoon_src", type=str, required=True, help="Source dir for cartoon/anime faces")
    p.add_argument("--out_dir", type=str, default="./data", help="Output root dir")
    p.add_argument("--size", type=int, default=512, help="Output image size")
    p.add_argument("--num_real", type=int, default=2000, help="Max number of real images to keep")
    p.add_argument("--num_cartoon", type=int, default=2000, help="Max number of cartoon images to keep")
    p.add_argument("--threads", type=int, default=8, help="Worker threads")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--min_side_real", type=int, default=256, help="Minimum short side for real images")
    p.add_argument("--min_side_cartoon", type=int, default=256, help="Minimum short side for cartoon images")
    p.add_argument("--max_aspect_real", type=float, default=1.25, help="Maximum aspect ratio for real images")
    p.add_argument("--max_aspect_cartoon", type=float, default=1.35, help="Maximum aspect ratio for cartoon images")
    p.add_argument("--resume", action="store_true", help="Skip existing outputs")
    # 动漫脸质量增强
    p.add_argument("--cascade", type=str, default=None, help="Path to lbpcascade_animeface.xml; enable face-based crop for cartoon")
    p.add_argument("--min_saturation", type=float, default=20.0, help="Min mean saturation for cartoon (filter sketch/gray); 0=off")
    p.add_argument("--min_laplacian_var", type=float, default=0.0, help="Min Laplacian variance for cartoon (filter blur); 0=off")
    p.add_argument("--face_pad", type=float, default=1.5, help="Crop size = face_box * face_pad when using cascade")
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(root: Path):
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
        and "__MACOSX" not in p.parts and not p.name.startswith("._")
    ]


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def basic_filter(img: Image.Image, min_side: int, max_aspect: float) -> bool:
    w, h = img.size
    if min(w, h) < min_side:
        return False
    aspect = max(w / h, h / w)
    if aspect > max_aspect:
        return False
    return True


def cartoon_filter_extra(img: Image.Image, min_saturation: float, min_laplacian_var: float) -> bool:
    """过滤线稿/灰度（饱和度）、模糊图（Laplacian 方差）。阈值 0 表示不检查。"""
    if min_saturation > 0:
        # PIL HSV: H 0-255, S 0-255, V 0-255
        img_hsv = np.array(img.convert("HSV"))
        saturation = float(img_hsv[:, :, 1].mean())
        if saturation < min_saturation:
            return False
    if min_laplacian_var > 0:
        import cv2
        gray = np.array(img.convert("L"))
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if var < min_laplacian_var:
            return False
    return True


def detect_anime_face_crop(img: Image.Image, cascade, face_pad: float):
    """
    用 lbpcascade_animeface 检测动漫脸，取最大脸，以脸为中心按 face_pad 倍扩展裁剪为正方形。
    返回 PIL.Image 或 None（未检测到脸或出错）。
    """
    import cv2
    arr = np.array(img)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape[:2]
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    if len(faces) == 0:
        return None
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    cx = x + w / 2.0
    cy = y + h / 2.0
    s = max(w, h) * face_pad
    left = int(cx - s / 2)
    top = int(cy - s / 2)
    right = int(left + s)
    bottom = int(top + s)
    left = max(0, min(left, w_img - 1))
    top = max(0, min(top, h_img - 1))
    right = max(left + 1, min(right, w_img))
    bottom = max(top + 1, min(bottom, h_img))
    crop = img.crop((left, top, right, bottom))
    # 若裁剪不是正方形，按短边中心裁成正方形
    cw, ch = crop.size
    if cw != ch:
        crop = center_crop_square(crop)
    return crop


def process_one(
    src: Path,
    dst: Path,
    size: int,
    min_side: int,
    max_aspect: float,
    resume: bool,
    cascade_classifier=None,
    min_saturation: float = 0.0,
    min_laplacian_var: float = 0.0,
    face_pad: float = 1.5,
):
    if resume and dst.exists():
        return True, str(dst)

    try:
        img = Image.open(src).convert("RGB")
        if not basic_filter(img, min_side=min_side, max_aspect=max_aspect):
            return False, f"filtered: {src}"

        use_face = cascade_classifier is not None
        if use_face:
            img_crop = detect_anime_face_crop(img, cascade_classifier, face_pad)
            if img_crop is None:
                return False, f"no_face: {src}"
            img = img_crop
        else:
            img = center_crop_square(img)

        if min_saturation > 0 or min_laplacian_var > 0:
            if not cartoon_filter_extra(img, min_saturation, min_laplacian_var):
                return False, f"quality: {src}"

        img = img.resize((size, size), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="PNG")
        return True, str(dst)
    except Exception as e:
        return False, f"{src}: {e}"


def prepare_domain(
    src_dir: Path,
    out_dir: Path,
    num_keep: int,
    size: int,
    min_side: int,
    max_aspect: float,
    threads: int,
    seed: int,
    resume: bool,
    desc: str,
    cascade_classifier=None,
    min_saturation: float = 0.0,
    min_laplacian_var: float = 0.0,
    face_pad: float = 1.5,
):
    ensure_dir(out_dir)

    files = list_images(src_dir)
    if not files:
        raise ValueError(f"No images found under {src_dir}")

    random.Random(seed).shuffle(files)
    # 启用人脸检测时过滤更严，多取候选
    mult = 10 if cascade_classifier is not None else 5
    candidates = files[: max(num_keep * mult, num_keep)]

    futures = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for i, src in enumerate(candidates):
            dst = out_dir / f"{i:05d}.png"
            futures.append(
                ex.submit(
                    process_one,
                    src,
                    dst,
                    size,
                    min_side,
                    max_aspect,
                    resume,
                    cascade_classifier=cascade_classifier,
                    min_saturation=min_saturation,
                    min_laplacian_var=min_laplacian_var,
                    face_pad=face_pad,
                )
            )

        kept = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            ok, _ = fut.result()
            if ok:
                kept += 1
            if kept >= num_keep:
                break

    valid_files = sorted(out_dir.glob("*.png"))
    if len(valid_files) < num_keep:
        print(f"[{desc}] Warning: only kept {len(valid_files)} images")
    valid_files = valid_files[:num_keep]

    for p in sorted(out_dir.glob("*.png"))[num_keep:]:
        p.unlink()

    temp_files = []
    for i, p in enumerate(valid_files):
        tmp = out_dir / f"tmp_{i:05d}.png"
        p.rename(tmp)
        temp_files.append(tmp)

    for i, p in enumerate(temp_files):
        p.rename(out_dir / f"{i:05d}.png")

    final_count = len(list(out_dir.glob("*.png")))
    print(f"[{desc}] Final count: {final_count}")


def write_pairs_csv(real_dir: Path, cartoon_dir: Path, csv_path: Path, seed: int):
    real_files = sorted([p.name for p in real_dir.glob("*.png")])
    cartoon_files = sorted([p.name for p in cartoon_dir.glob("*.png")])

    n = min(len(real_files), len(cartoon_files))
    cartoon_files = cartoon_files[:n]
    random.Random(seed).shuffle(cartoon_files)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["real", "cartoon"])
        for r, c in zip(real_files[:n], cartoon_files):
            writer.writerow([r, c])

    print(f"[pairs] wrote {n} pairs -> {csv_path}")


def main():
    args = parse_args()
    random.seed(args.seed)

    real_src = Path(args.real_src)
    cartoon_src = Path(args.cartoon_src)
    out_dir = Path(args.out_dir)

    real_out = out_dir / "real"
    cartoon_out = out_dir / "cartoon"
    pairs_csv = out_dir / "pairs.csv"

    real_src = real_src.resolve()
    cartoon_src = cartoon_src.resolve()
    if not real_src.exists():
        raise FileNotFoundError(
            f"real_src not found: {real_src}\n"
            f"当前工作目录: {Path.cwd()}\n请确认路径存在，或使用已有目录如 data_sucess/real"
        )
    if not cartoon_src.exists():
        raise FileNotFoundError(
            f"cartoon_src not found: {cartoon_src}\n"
            f"当前工作目录: {Path.cwd()}\n请确认路径存在，或使用已有目录如 data_sucess/cartoon"
        )

    ensure_dir(out_dir)

    prepare_domain(
        src_dir=real_src,
        out_dir=real_out,
        num_keep=args.num_real,
        size=args.size,
        min_side=args.min_side_real,
        max_aspect=args.max_aspect_real,
        threads=args.threads,
        seed=args.seed,
        resume=args.resume,
        desc="Preparing real",
    )

    cascade = None
    if args.cascade:
        cascade_path = Path(args.cascade)
        if cascade_path.exists():
            import cv2
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if cascade.empty():
                raise RuntimeError(f"Failed to load cascade: {args.cascade}")
            print(f"[cartoon] Using anime face cascade: {args.cascade}")
        else:
            raise FileNotFoundError(f"--cascade not found: {args.cascade}")

    prepare_domain(
        src_dir=cartoon_src,
        out_dir=cartoon_out,
        num_keep=args.num_cartoon,
        size=args.size,
        min_side=args.min_side_cartoon,
        max_aspect=args.max_aspect_cartoon,
        threads=args.threads,
        seed=args.seed,
        resume=args.resume,
        desc="Preparing cartoon",
        cascade_classifier=cascade,
        min_saturation=args.min_saturation,
        min_laplacian_var=args.min_laplacian_var,
        face_pad=args.face_pad,
    )

    write_pairs_csv(real_out, cartoon_out, pairs_csv, args.seed)

    print("\nDone.")
    print(f"real   : {real_out}")
    print(f"cartoon: {cartoon_out}")
    print(f"pairs  : {pairs_csv}")


if __name__ == "__main__":
    main()
