import os
import random
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = (".jpg", ".png", ".jpeg", ".webp", ".bmp")


class CartoonDataset(Dataset):
    def __init__(self, root_dir, size=512, use_pairs_csv=True):
        self.real_dir = os.path.join(root_dir, "real")
        self.cartoon_dir = os.path.join(root_dir, "cartoon")
        self.pairs_csv = os.path.join(root_dir, "pairs.csv")

        if not os.path.exists(self.real_dir) or not os.path.exists(self.cartoon_dir):
            raise FileNotFoundError(
                f"⚠️ 路径错误！请确保 {root_dir} 下有 real 和 cartoon 文件夹"
            )

        self.real_names = sorted([
            f for f in os.listdir(self.real_dir)
            if f.lower().endswith(IMG_EXTS)
        ])
        self.cartoon_names = sorted([
            f for f in os.listdir(self.cartoon_dir)
            if f.lower().endswith(IMG_EXTS)
        ])

        if not self.real_names or not self.cartoon_names:
            raise FileNotFoundError(
                f"⚠️ 目录为空！请确保 {self.real_dir} 与 {self.cartoon_dir} 内各有至少一张图片。"
            )

        # 更保守的数据增强：不再用 RandomResizedCrop
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.use_pairs_csv = False
        self.pairs = []

        if use_pairs_csv and os.path.isfile(self.pairs_csv):
            try:
                with open(self.pairs_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        r = row["real"]
                        c = row["cartoon"]
                        if (
                            os.path.isfile(os.path.join(self.real_dir, r))
                            and os.path.isfile(os.path.join(self.cartoon_dir, c))
                        ):
                            self.pairs.append((r, c))
                if len(self.pairs) > 0:
                    self.use_pairs_csv = True
                    print(f"[CartoonDataset] Using pairs.csv with {len(self.pairs)} pairs.")
            except Exception as e:
                print(f"[CartoonDataset] Failed to read pairs.csv, fallback to random unpaired. Error: {e}")

        if not self.use_pairs_csv:
            print(
                f"[CartoonDataset] No valid pairs.csv found, using fallback unpaired mode. "
                f"real={len(self.real_names)}, cartoon={len(self.cartoon_names)}"
            )

    def __len__(self):
        if self.use_pairs_csv:
            return len(self.pairs)
        return max(len(self.real_names), len(self.cartoon_names))

    def __getitem__(self, idx):
        n = self.__len__()
        if n <= 0:
            raise RuntimeError("Dataset length is zero.")

        try:
            if self.use_pairs_csv:
                real_name, cartoon_name = self.pairs[idx % len(self.pairs)]
            else:
                # fallback：real 按 index，cartoon 随机抽，适合 unpaired style transfer
                real_name = self.real_names[idx % len(self.real_names)]
                cartoon_name = random.choice(self.cartoon_names)

            real_img = Image.open(os.path.join(self.real_dir, real_name)).convert("RGB")
            cartoon_img = Image.open(os.path.join(self.cartoon_dir, cartoon_name)).convert("RGB")

            return {
                "real": self.transform(real_img),
                "cartoon": self.transform(cartoon_img),
            }

        except Exception as e:
            # 最多重试一次，避免坏图导致无限递归
            retry_idx = random.randint(0, n - 1)
            if retry_idx == idx:
                retry_idx = (idx + 1) % n
            try:
                if self.use_pairs_csv:
                    real_name, cartoon_name = self.pairs[retry_idx % len(self.pairs)]
                else:
                    real_name = self.real_names[retry_idx % len(self.real_names)]
                    cartoon_name = random.choice(self.cartoon_names)
                real_img = Image.open(os.path.join(self.real_dir, real_name)).convert("RGB")
                cartoon_img = Image.open(os.path.join(self.cartoon_dir, cartoon_name)).convert("RGB")
                return {"real": self.transform(real_img), "cartoon": self.transform(cartoon_img)}
            except Exception:
                raise RuntimeError(f"Failed to load image (idx={idx}, retry={retry_idx}): {e}") from e