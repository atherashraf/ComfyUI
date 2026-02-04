import uuid
import sys
import os
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pathlib import Path

# --- 1. COMPIFY SETUP (Global) ---
# Point this to your ComfyUI root folder
COMFY_PATH = os.path.abspath("./ComfyUI")
sys.path.append(COMFY_PATH)

# Import ComfyUI internals
from nodes import (
    CheckpointLoaderSimple,
    VAEEncodeForInpaint,
    KSampler,
    VAEDecode,
    CLIPTextEncode
)

# Initialize Loaders globally so we don't reload the 2GB+ model on every API call
# (Optional optimization: if you want to load fresh every time, move this inside the function)
print("Loading ComfyUI Model into Memory...")
ckpt_loader = CheckpointLoaderSimple()
# Ensure this file exists in ComfyUI/models/checkpoints/
# MODEL_NAME = "v1-5-pruned-emaonly-fp16.safetensors"
MODEL_NAME = "lustifySDXLNSFW_v20-inpainting"
MODEL_LOADED, CLIP_LOADED, VAE_LOADED = ckpt_loader.load_checkpoint(ckpt_name=MODEL_NAME)
print("Model Loaded Successfully.")


# --- 2. THE INPAINTING HELPER ---

def run_inpainting_comfy(image_pil, mask_pil, prompt_text="high quality"):
    print("--- Starting ComfyUI Inpainting Task ---")

    # --- A. Helper: Convert PIL to Tensor ---
    # ComfyUI expects [Batch, Height, Width, Channels] (Float 0..1)
    def pil_to_tensor(pil_img):
        # Ensure we are in RGB for images
        if pil_img.mode != "RGB" and pil_img.mode != "L":
            pil_img = pil_img.convert("RGB")

        img_arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_arr)[None,]  # Add batch dimension

    # Convert Inputs
    image_tensor = pil_to_tensor(image_pil)
    # Mask must be grayscale [1, H, W, 1]
    mask_tensor = pil_to_tensor(mask_pil.convert("L"))

    # --- B. Encode Prompts (Conditioning) ---
    # Uses the globally loaded CLIP model
    clip_encoder = CLIPTextEncode()

    # Positive Prompt
    cond_pos = clip_encoder.encode(
        clip=CLIP_LOADED,
        text=prompt_text
    )[0]

    # Negative Prompt
    cond_neg = clip_encoder.encode(
        clip=CLIP_LOADED,
        text="blur, low quality, distortion, ugly, bad anatomy, text, watermark"
    )[0]

    # --- C. VAE Encode (The Inpainting Setup) ---
    # Uses the globally loaded VAE model
    vae_inpaint = VAEEncodeForInpaint()

    # 'grow_mask_by' expands the mask slightly to ensure seamless edges
    latent = vae_inpaint.encode(
        pixels=image_tensor,
        mask=mask_tensor,
        vae=VAE_LOADED,
        grow_mask_by=6
    )[0]

    # --- D. KSampler (The Generation) ---
    # Uses the globally loaded UNET Model
    ksampler = KSampler()

    # Generate a random seed if you want variety
    seed = torch.randint(0, 10000000, (1,)).item()

    latent_out = ksampler.sample(
        model=MODEL_LOADED,
        seed=seed,
        steps=20,
        cfg=8.0,
        sampler_name="euler",
        scheduler="normal",
        positive=cond_pos,
        negative=cond_neg,
        latent_image=latent,
        denoise=1.0  # 1.0 = completely regenerate masked area
    )[0]

    # --- E. Decode & Return ---
    vae_decoder = VAEDecode()
    decoded = vae_decoder.decode(samples=latent_out, vae=VAE_LOADED)[0]

    # --- F. The Fix: .detach() ---
    # We must detach from the gradient graph before converting to numpy
    result_array = 255. * decoded[0].detach().cpu().numpy()

    # Clip values to safe 0-255 range and convert to uint8
    return Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
