import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CartoonDataset(Dataset):
    def __init__(self, root_dir, size=512):
        self.real_dir = os.path.join(root_dir, "real")
        self.cartoon_dir = os.path.join(root_dir, "cartoon")
        
        if not os.path.exists(self.real_dir) or not os.path.exists(self.cartoon_dir):
            raise FileNotFoundError(f"⚠️ 路径错误！请确保 {root_dir} 下有 real 和 cartoon 文件夹")

        self.real_names = [f for f in os.listdir(self.real_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.cartoon_names = [f for f in os.listdir(self.cartoon_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not self.real_names or not self.cartoon_names:
            raise FileNotFoundError(
                f"⚠️ 目錄為空！請確保 {self.real_dir} 與 {self.cartoon_dir} 內各有至少一張 .jpg/.png/.jpeg 圖片。"
            )

        self.transform = transforms.Compose([
            # 保持 512 分辨率，FLUX 对 16 整数倍的分辨率支持最好
            transforms.RandomResizedCrop(
                size=(size, size), 
                scale=(0.85, 1.0), # 稍微放宽缩放，增加多样性
                ratio=(0.95, 1.05)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 明确指定三个通道
        ])

    def __len__(self):
        # 使用真实的样本总量
        return max(len(self.real_names), len(self.cartoon_names))

    def __getitem__(self, idx):
        # 🌟 改进：real 尽量按顺序遍历，cartoon 保持随机
        real_name = self.real_names[idx % len(self.real_names)]
        cartoon_name = random.choice(self.cartoon_names)
        
        try:
            # 强制转换为 RGB，防止灰度图或带 Alpha 通道的图导致维度错误
            real_img = Image.open(os.path.join(self.real_dir, real_name)).convert("RGB")
            cartoon_img = Image.open(os.path.join(self.cartoon_dir, cartoon_name)).convert("RGB")
            
            return {
                "real": self.transform(real_img),
                "cartoon": self.transform(cartoon_img)
            }
        except Exception as e:
            # 如果某张图坏了，打印一下文件名方便你清理数据集
            n = self.__len__()
            if n <= 0:
                raise
            print(f"❌ 损坏的图像: {real_name} 或 {cartoon_name}, 正在重试...")
            return self.__getitem__(random.randint(0, n - 1))