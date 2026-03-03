"""
统一评估脚本：Face-Sim、CLIP Score、FID。
可供 Stage1 / Stage2 / Stage3 的 inference 输出共用。

用法示例：
  命令行（任一 stage 的 inference 输出都可评估）：
    python evaluate.py --real_dir ./data/real --gen_dir ./samples
    python evaluate.py --real_dir ./data/real --gen_dir ./samples --reference_dir ./data/anime_ref --output results.json

  在代码中调用（例如 inference.py 跑完后）：
    from evaluate import run_evaluation
    run_evaluation(real_dir="./data/real", gen_dir="./samples", reference_dir="./data/anime_ref", output_path="stage1_metrics.json")
"""
import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# CLIP
from transformers import CLIPModel, CLIPProcessor

# Face-Sim: facenet_pytorch (需安装: pip install facenet-pytorch)
try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False

# FID: torchmetrics (需安装: pip install torchmetrics[image])
try:
    from torchmetrics.image import FrechetInceptionDistance
    HAS_FID = True
except ImportError:
    HAS_FID = False


# ---------- 配置 ----------
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
STYLE_PROMPT = "a high-quality Japanese anime portrait illustration"
FACENET_PRETRAINED = "vggface2"
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _find_real_for_gen(gen_name: str, real_dir: str) -> Optional[str]:
    """按文件名 stem 在 real_dir 中找对应原图。"""
    stem = Path(gen_name).stem
    for ext in IMG_EXTENSIONS:
        p = os.path.join(real_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _list_images(dir_path: str):
    return sorted(
        [f for f in os.listdir(dir_path) if f.lower().endswith(IMG_EXTENSIONS)],
        key=lambda x: Path(x).stem,
    )


def compute_clip_score(clip_model, clip_processor, image_paths, device, prompt=STYLE_PROMPT):
    """CLIP Score: 生成图与风格描述之间的余弦相似度（与论文一致），取平均。"""
    clip_model.eval()
    scores = []
    # 预编码文本（一次）
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = clip_model.get_text_features(**text_inputs)
        text_feats = F.normalize(text_feats, dim=-1)
    for path in tqdm(image_paths, desc="CLIP Score"):
        img = Image.open(path).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_feats = clip_model.get_image_features(**inputs)
            image_feats = F.normalize(image_feats, dim=-1)
            cos_sim = (image_feats * text_feats).sum(dim=-1).item()
            scores.append(cos_sim)
    return sum(scores) / len(scores) if scores else 0.0


def compute_face_sim(mtcnn, resnet, real_dir, gen_paths, device):
    """Face-Sim: 每对 (real, gen) 用 FaceNet 提特征后算余弦相似度，取平均。"""
    if not HAS_FACENET:
        return None
    resnet.eval()
    sims = []
    for gen_path in tqdm(gen_paths, desc="Face-Sim"):
        real_path = _find_real_for_gen(os.path.basename(gen_path), real_dir)
        if real_path is None:
            continue
        real_img = Image.open(real_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")
        real_face = mtcnn(real_img)
        gen_face = mtcnn(gen_img)
        if real_face is None or gen_face is None:
            continue
        with torch.no_grad():
            real_emb = resnet(real_face.unsqueeze(0).to(device))
            gen_emb = resnet(gen_face.unsqueeze(0).to(device))
            real_emb = F.normalize(real_emb, dim=-1)
            gen_emb = F.normalize(gen_emb, dim=-1)
            sim = F.cosine_similarity(real_emb, gen_emb, dim=-1).item()
            sims.append(sim)
    return sum(sims) / len(sims) if sims else None


def compute_fid(gen_dir, reference_dir, device, max_ref=1000, max_gen=1000):
    """FID: 生成图分布 vs 参考艺术图分布。"""
    if not HAS_FID:
        return None
    fid = FrechetInceptionDistance(normalize=True).to(device)
    ref_list = _list_images(reference_dir)[:max_ref]
    gen_list = _list_images(gen_dir)[:max_gen]
    if not ref_list or not gen_list:
        return None
    to_tensor = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    for p in tqdm(ref_list, desc="FID (ref)"):
        img = Image.open(os.path.join(reference_dir, p)).convert("RGB")
        t = to_tensor(img).unsqueeze(0).to(device)
        fid.update(t, real=True)
    for p in tqdm(gen_list, desc="FID (gen)"):
        img = Image.open(os.path.join(gen_dir, p)).convert("RGB")
        t = to_tensor(img).unsqueeze(0).to(device)
        fid.update(t, real=False)
    return fid.compute().item()


def run_evaluation(
    real_dir: str,
    gen_dir: str,
    reference_dir: Optional[str] = None,
    style_prompt: str = STYLE_PROMPT,
    output_path: Optional[str] = None,
    max_fid_ref: int = 500,
    max_fid_gen: int = 500,
    device: Optional[str] = None,
) -> dict:
    """
    供其他 inference 脚本调用：传入目录路径，返回包含 clip_score / face_sim / fid 的字典。
    若某指标未计算则对应键为 None。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_names = _list_images(gen_dir)
    if not gen_names:
        return {"n_images": 0, "clip_score": None, "face_sim": None, "fid": None}
    gen_paths = [os.path.join(gen_dir, n) for n in gen_names]
    results = {"n_images": len(gen_paths), "real_dir": real_dir, "gen_dir": gen_dir}

    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    results["clip_score"] = compute_clip_score(clip_model, clip_processor, gen_paths, device, prompt=style_prompt)
    del clip_model, clip_processor
    if device == "cuda":
        torch.cuda.empty_cache()

    if HAS_FACENET:
        print("Loading FaceNet (MTCNN + InceptionResnetV1)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mtcnn = MTCNN(image_size=160, margin=0, device=device)
        resnet = InceptionResnetV1(pretrained=FACENET_PRETRAINED).eval().to(device)
        results["face_sim"] = compute_face_sim(mtcnn, resnet, real_dir, gen_paths, device)
    else:
        results["face_sim"] = None

    if reference_dir and os.path.isdir(reference_dir) and HAS_FID:
        results["fid"] = compute_fid(gen_dir, reference_dir, device, max_ref=max_fid_ref, max_gen=max_fid_gen)
    else:
        results["fid"] = None

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Face-Sim, CLIP Score, FID for any inference output.")
    parser.add_argument("--real_dir", type=str, required=True, help="原图目录（与 gen 按文件名 stem 对应）")
    parser.add_argument("--gen_dir", type=str, required=True, help="生成图目录（如 samples/ 或某 stage 输出）")
    parser.add_argument("--reference_dir", type=str, default=None, help="FID 参考集（高质量艺术/动漫图）；不传则跳过 FID")
    parser.add_argument("--output", type=str, default=None, help="结果写入 JSON 的路径")
    parser.add_argument("--style_prompt", type=str, default=STYLE_PROMPT, help="CLIP 风格描述")
    parser.add_argument("--max_fid_ref", type=int, default=500)
    parser.add_argument("--max_fid_gen", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if not os.path.isdir(args.gen_dir):
        print(f"gen_dir not found or not a directory: {args.gen_dir}")
        return
    if not _list_images(args.gen_dir):
        print(f"No images in {args.gen_dir}. Run inference first.")
        return
    if not os.path.isdir(args.real_dir):
        print(f"real_dir not found: {args.real_dir}")
        return
    if not HAS_FACENET:
        print("Face-Sim will be skipped (install facenet-pytorch).")
    if args.reference_dir and not HAS_FID:
        print("FID will be skipped (install torchmetrics[image]).")
    if not args.reference_dir:
        print("FID will be skipped (no --reference_dir).")

    results = run_evaluation(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        reference_dir=args.reference_dir,
        style_prompt=args.style_prompt,
        output_path=args.output,
        max_fid_ref=args.max_fid_ref,
        max_fid_gen=args.max_fid_gen,
        device=device,
    )

    print("-" * 40)
    print("Evaluation Results")
    print("-" * 40)
    print(f"  N images:     {results['n_images']}")
    print(f"  CLIP Score:   {results['clip_score']:.4f}" if results.get('clip_score') is not None else "  CLIP Score:   N/A")
    print(f"  Face-Sim:     {results['face_sim']:.4f}" if results.get('face_sim') is not None else "  Face-Sim:     N/A")
    print(f"  FID:          {results['fid']:.2f}" if results.get('fid') is not None else "  FID:          N/A")
    print("-" * 40)
    if args.output:
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
