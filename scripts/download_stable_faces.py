import os, io, random, hashlib
from typing import Any, Dict, Optional, List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def _to_pil(x):
    """Convert HF dataset image object (PIL / dict{bytes,path} / datasets.Image) to PIL.Image."""
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict):
        if x.get("bytes") is not None:
            return Image.open(io.BytesIO(x["bytes"]))
        if x.get("path") is not None:
            return Image.open(x["path"])
    # datasets.Image / other wrappers
    try:
        return x.convert("RGB")
    except Exception as e:
        raise TypeError(f"Unsupported image object type: {type(x)}") from e


def _pick_image_key(sample: Dict[str, Any]) -> Optional[str]:
    """Pick image column key, robust to case and dataset-specific naming."""
    # common keys (case included)
    candidates = ["image", "Image", "img", "Img", "images", "Images", "picture", "Picture", "pixel_values"]
    for k in candidates:
        if k in sample:
            return k

    # fallback: find any key containing image/img/pic (case-insensitive)
    keys = list(sample.keys())
    for k in keys:
        lk = k.lower()
        if ("image" in lk) or ("img" in lk) or ("pic" in lk):
            return k

    return None


def _center_crop_resize(img: Image.Image, size: int = 512) -> Image.Image:
    """Center-crop to square then resize."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.BICUBIC)


def _hash_name(img: Image.Image, prefix: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    h = hashlib.sha1(buf.getvalue()).hexdigest()[:12]
    return f"{prefix}_{h}.jpg"


def _count_images(out_dir: str) -> int:
    if not os.path.exists(out_dir):
        return 0
    n = 0
    for f in os.listdir(out_dir):
        lf = f.lower()
        if lf.endswith((".jpg", ".jpeg", ".png", ".webp")):
            n += 1
    return n


def dump_images(
    dataset_name: str,
    split: str,
    out_dir: str,
    num: int,
    streaming: bool = True,
    seed: int = 0,
    size: int = 512,
    buffer_shuffle: int = 2000,
    max_tries_multiplier: int = 80,
):
    os.makedirs(out_dir, exist_ok=True)

    # resume: only download the missing amount
    existing = _count_images(out_dir)
    if existing >= num:
        print(f"✅ Skip {dataset_name}: {out_dir} already has {existing} images (>= {num})")
        return

    need = num - existing
    print(f"\n📡 Loading {dataset_name} [{split}] streaming={streaming}")
    print(f"📁 {out_dir}: existing={existing}, need={need}")

    ds = load_dataset(dataset_name, split=split, streaming=streaming)

    if streaming:
        ds = ds.shuffle(buffer_size=buffer_shuffle, seed=seed)

    it = iter(ds)
    first = next(it)
    img_key = _pick_image_key(first)
    if img_key is None:
        raise RuntimeError(f"Cannot find image field in keys: {list(first.keys())}")

    def chain_first():
        yield first
        for s in it:
            yield s

    saved = 0
    tries = 0
    max_tries = need * max_tries_multiplier

    pbar = tqdm(total=need)
    for sample in chain_first():
        if saved >= need or tries >= max_tries:
            break
        tries += 1

        try:
            img = _to_pil(sample[img_key]).convert("RGB")
            img = _center_crop_resize(img, size=size)

            fname = _hash_name(img, prefix=dataset_name.split("/")[-1])
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                continue

            img.save(fpath, "JPEG", quality=95)
            saved += 1
            pbar.update(1)

        except Exception:
            continue

    pbar.close()
    final_count = _count_images(out_dir)
    print(f"✅ Saved +{saved} images to {out_dir} (now {final_count} total, tries={tries})")

    if saved < need:
        print("⚠️ Warning: did not reach requested amount. Try increasing max_tries_multiplier or use non-streaming.")


def try_dump_from_candidates(
    candidates: List[str],
    split: str,
    out_dir: str,
    num: int,
    seed: int,
    size: int,
):
    # resume check before trying anything
    existing = _count_images(out_dir)
    if existing >= num:
        print(f"✅ Skip REAL: {out_dir} already has {existing} images (>= {num})")
        return

    last_err = None
    for name in candidates:
        try:
            dump_images(
                dataset_name=name,
                split=split,
                out_dir=out_dir,
                num=num,
                streaming=True,
                seed=seed,
                size=size,
            )
            print(f"✅ Using REAL dataset: {name}")
            return
        except Exception as e:
            print(f"❌ Failed REAL candidate: {name} -> {e}")
            last_err = e
            continue
    raise RuntimeError(f"All REAL dataset candidates failed. Last error: {last_err}")


if __name__ == "__main__":
    random.seed(0)

    # ------------ Config ------------
    N_REAL = 1000
    N_CARTOON = 1000
    SIZE = 512

    REAL_DIR = "./data/real"
    CARTOON_DIR = "./data/cartoon"

    # REAL: FFHQ mirrors (fallback)
    ffhq_candidates = [
        "marcosv/ffhq-dataset",
        "students/ffhq",
        "huggingface/ffhq",
        "KBlueLeaf/ffhq",
    ]

    # CARTOON: AnimeStyle (field name is often "Image")
    cartoon_dataset = "Dhiraj45/AnimeStyle"

    # ------------ Download ------------
    try_dump_from_candidates(
        candidates=ffhq_candidates,
        split="train",
        out_dir=REAL_DIR,
        num=N_REAL,
        seed=0,
        size=SIZE,
    )

    dump_images(
        dataset_name=cartoon_dataset,
        split="train",
        out_dir=CARTOON_DIR,
        num=N_CARTOON,
        streaming=True,
        seed=1,
        size=SIZE,
    )

    print("\n🎉 Done.")
    print(f" - REAL   : {REAL_DIR} ({_count_images(REAL_DIR)} images)")
    print(f" - CARTOON: {CARTOON_DIR} ({_count_images(CARTOON_DIR)} images)")
