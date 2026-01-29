import os
import sys
import uuid
import asyncio
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

# -------------------------
# 0) Torch performance knobs
# -------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1) ComfyUI setup
# -------------------------
# Adjust this if your ComfyUI folder is different
COMFY_PATH = os.path.abspath("./ComfyUI")
# if not os.path.isdir(COMFY_PATH):
#     raise RuntimeError(f"COMFY_PATH not found: {COMFY_PATH}")

sys.path.append(COMFY_PATH)

# Import ComfyUI internals
from nodes import (
    CheckpointLoaderSimple,
    VAEEncodeForInpaint,
    KSampler,
    VAEDecode,
    CLIPTextEncode
)

# -------------------------
# 2) Global model load (ONCE)
# -------------------------
MODEL_NAME = "v1-5-pruned-emaonly-fp16.safetensors"
# MODEL_NAME = "lustifySDXLNSFW_v20-inpainting.safetensors"
# Expected location:
# ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors

print("Loading ComfyUI checkpoint:", MODEL_NAME)
ckpt_loader = CheckpointLoaderSimple()
MODEL_LOADED, CLIP_LOADED, VAE_LOADED = ckpt_loader.load_checkpoint(ckpt_name=MODEL_NAME)
print("Model loaded OK.")

# Optional: show devices (may be CPU at this moment; Comfy nodes often move internally)
try:
    print("CUDA available:", torch.cuda.is_available(), "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception:
    pass

# -------------------------
# 3) Inpainting function
# -------------------------
def run_inpainting_comfy(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    prompt_text: str = "high quality",
    negative_text: str = "blur, low quality, distortion, ugly, bad anatomy, text, watermark",
    max_side: int = 1024,
    steps: int = 20,
    cfg: float = 8.0,
    grow_mask_by: int = 6,
    denoise: float = 1.0,
):
    """
    Returns a PIL RGB image.
    """

    # Defensive resize to avoid huge VRAM usage (8GB VRAM friendly)
    def limit_size(pil_img: Image.Image, max_side: int):
        w, h = pil_img.size
        m = max(w, h)
        if m <= max_side:
            return pil_img
        scale = max_side / float(m)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return pil_img.resize((nw, nh), Image.LANCZOS)

    image_pil = limit_size(image_pil.convert("RGB"), max_side=max_side)
    mask_pil = limit_size(mask_pil.convert("L"), max_side=max_side)

    # PIL -> tensor [1,H,W,C] float 0..1
    def pil_to_tensor_hwc(pil_img: Image.Image, channels: int):
        arr = np.array(pil_img, copy=False)
        if channels == 3 and arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if channels == 1 and arr.ndim == 3:
            arr = arr[..., :1]
        arr = arr.astype(np.float32) / 255.0
        t = torch.from_numpy(arr)[None, ...]  # [1,H,W,C]
        return t

    image_tensor = pil_to_tensor_hwc(image_pil, 3)
    mask_tensor = pil_to_tensor_hwc(mask_pil, 1)

    # In many Comfy builds, nodes can accept GPU tensors; if you see device errors,
    # comment out these .to(...) lines and keep tensors on CPU.
    # if DEVICE.type == "cuda":
    #     image_tensor = image_tensor.to(DEVICE, non_blocking=True)
    #     mask_tensor = mask_tensor.to(DEVICE, non_blocking=True)

    clip_encoder = CLIPTextEncode()
    vae_inpaint = VAEEncodeForInpaint()
    ksampler = KSampler()
    vae_decoder = VAEDecode()

    # Inference mode + autocast reduces VRAM and speeds up
    with torch.inference_mode():
        # autocast ON for CLIP text encode
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            cond_pos = clip_encoder.encode(clip=CLIP_LOADED, text=prompt_text)[0]
            cond_neg = clip_encoder.encode(clip=CLIP_LOADED, text=negative_text)[0]

        # autocast OFF for inpaint encode (mask conv2d wants float32)
        with torch.autocast(device_type="cuda", enabled=False):
            latent = vae_inpaint.encode(
                pixels=image_tensor,  # CPU float32
                mask=mask_tensor,  # CPU float32
                vae=VAE_LOADED,
                grow_mask_by=int(grow_mask_by),
            )[0]

        # autocast ON for sampler + decode
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            seed = torch.randint(0, 10_000_000, (1,), device="cpu").item()
            latent_out = ksampler.sample(
                model=MODEL_LOADED,
                seed=int(seed),
                steps=int(steps),
                cfg=float(cfg),
                sampler_name="euler",
                scheduler="normal",
                positive=cond_pos,
                negative=cond_neg,
                latent_image=latent,
                denoise=float(denoise),
            )[0]

            decoded = vae_decoder.decode(samples=latent_out, vae=VAE_LOADED)[0]
            img_u8 = (decoded[0].clamp(0, 1) * 255.0).to(torch.uint8)
            out_np = img_u8.cpu().numpy()

    # Cleanup (important for long-running API servers)
    del image_tensor, mask_tensor, cond_pos, cond_neg, latent, latent_out, decoded, img_u8
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return Image.fromarray(out_np)

