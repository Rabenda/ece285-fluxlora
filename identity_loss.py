import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


class FaceIdentityLoss(nn.Module):
    """
    Identity-aware loss used in Stage III.

    We compute cosine distance between image embeddings from a frozen vision
    backbone and optimize:
        L_ID = 1 - cos(phi(x_real), phi(x_gen))
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16).eval()
        self.backbone.requires_grad_(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _prep(self, x):
        x = x.float()
        x = ((x + 1.0) / 2.0).clamp(0, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x.to(dtype=torch.bfloat16)

    def _embed(self, x):
        return self.backbone(self._prep(x)).pooler_output.float()

    def forward(self, x_real, x_gen):
        with torch.no_grad():
            emb_real = F.normalize(self._embed(x_real), dim=-1)
        emb_gen = F.normalize(self._embed(x_gen), dim=-1)
        cos_sim = (emb_real * emb_gen).sum(dim=-1)
        return 1.0 - cos_sim.mean()
