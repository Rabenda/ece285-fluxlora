"""
Identity loss (L_ID) for Stage III: 1 - cos(phi(x_real), phi(x_gen)).
Uses FaceNet (InceptionResnetV1, VGGFace2) as backbone, aligned with evaluation Face-Sim
and with common practice in face stylization (VToonify, DualStyleGAN, etc.).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from facenet_pytorch import InceptionResnetV1
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False

# FaceNet 输入：160x160，值域 [0, 1]（与 facenet_pytorch / VGGFace2 常见用法一致）
FACE_SIZE = 160


class FaceIdentityLoss(nn.Module):
    """
    Identity-aware loss for Stage III with a face recognition backbone.

    L_ID = 1 - cos(phi(x_real), phi(x_gen))
    Backbone: InceptionResnetV1 (VGGFace2), same as evaluation Face-Sim.
    Uses 160x160 resized crops (differentiable) so gradients flow to x_gen and thus to LoRA.
    """

    def __init__(self, device="cuda", pretrained="vggface2"):
        super().__init__()
        if not HAS_FACENET:
            raise ImportError(
                "FaceNet is required for FaceIdentityLoss. Install with: pip install facenet-pytorch"
            )
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False).to(device).eval()
        self.backbone.requires_grad_(False)
        self._device = device

    def _prep(self, x):
        """[B,3,H,W] in [-1,1] -> [B,3,160,160] in [0,1], differentiable."""
        x = x.float()
        x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        x = F.interpolate(x, size=(FACE_SIZE, FACE_SIZE), mode="bilinear", align_corners=False)
        return x

    def _embed(self, x):
        """x: [B,3,160,160] in [0,1]. Returns [B, 512]."""
        return self.backbone(x.to(self._device)).float()

    def forward(self, x_real, x_gen):
        # 原图作为锚点，不反传
        with torch.no_grad():
            real_160 = self._prep(x_real)
            emb_real = F.normalize(self._embed(real_160), dim=-1)
        # 生成图需要梯度，经 backbone 得到 emb_gen，梯度回传到 x_gen -> LoRA
        gen_160 = self._prep(x_gen)
        emb_gen = F.normalize(self._embed(gen_160), dim=-1)
        cos_sim = (emb_real * emb_gen).sum(dim=-1)
        return 1.0 - cos_sim.mean()
