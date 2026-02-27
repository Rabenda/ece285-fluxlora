import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

from models.flux_i2i_trainable import FluxI2ITrainable


# --- Config ---
CHECKPOINT_PATH = "./checkpoints/full_step_000500.pt"
INPUT_IMAGE_PATH = "./data/inference/photo4.jpg"
OUTPUT_PATH = "./inference_refinement_final.png"
T0 = 0.5
STEPS = 15
GUIDANCE = 3.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_inference_refinement():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = FluxI2ITrainable().to(DEVICE)
    model.setup_lora(rank=16, alpha=16)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    model.vae.to(torch.float32)
    model.eval()

    # 1) input preprocessing
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(Image.open(INPUT_IMAGE_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 2) encode and pack
        latents = model.vae.encode(img_tensor.to(torch.float32)).latent_dist.mode()
        latents = (latents - model.vae.config.shift_factor) * model.scaling_factor
        h, w = latents.shape[-2], latents.shape[-1]
        x_start = model._pack(latents).to(torch.bfloat16)

        # z_t = (1-t) * x0 + t * noise
        noise = torch.randn_like(x_start)
        x_t = (1.0 - T0) * x_start + T0 * noise

        # 3) conditioning
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            vlm_feats = model.get_vlm_embedding(img_tensor)
            pooled = model.clip_vision(
                ((F.interpolate((img_tensor + 1.0) / 2.0, size=(224, 224), mode="bilinear", align_corners=False)
                  - model.clip_mean.to(img_tensor.device)) / model.clip_std.to(img_tensor.device)).to(torch.bfloat16)
            ).pooler_output
            pooled_projections = model.pooled_proj(pooled).to(torch.bfloat16)

        encoder_hidden_states = torch.zeros((1, 512, 4096), device=DEVICE, dtype=torch.bfloat16)
        encoder_hidden_states[:, :257, :] = vlm_feats
        img_ids, txt_ids = model._get_ids(h, w, DEVICE)
        guidance = torch.full((1,), float(GUIDANCE), device=DEVICE, dtype=torch.bfloat16)

    # 4) ODE refinement
    print("Refining trajectory...")
    with torch.no_grad():
        dt = T0 / STEPS

        for i in range(STEPS):
            current_t = T0 * (1 - i / STEPS)
            t_tensor = torch.full((1,), current_t, device=DEVICE, dtype=torch.bfloat16)

            v_pred = model.unet(
                hidden_states=x_t,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=t_tensor * 1000,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

            v_pred = torch.nan_to_num(v_pred, nan=0.0).clamp(-3.0, 3.0)
            x_t = x_t - v_pred * dt
            print(f"Step {i + 1}/{STEPS} | t: {current_t:.3f}", end="\r")

    print("\nDecoding...")
    with torch.no_grad():
        out_latents = model._unpack(x_t, h, w).to(torch.float32)
        out_latents = out_latents.clamp(-5.0, 5.0)
        out_latents = (out_latents / model.scaling_factor) + model.vae.config.shift_factor
        final_image = model.vae.decode(out_latents).sample
        vutils.save_image((final_image[0].cpu().clamp(-1, 1) + 1) / 2, OUTPUT_PATH)
        print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_inference_refinement()