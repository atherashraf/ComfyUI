# scripts/inpainting.py
import numpy as np
import torch
from PIL import Image

# Import after COMFY_PATH is added to sys.path in smart_model_cache
from nodes import (
    VAEEncodeForInpaint,
    KSampler,
    VAEDecode,
    CLIPTextEncode
)

def run_inpainting_comfy(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    model,  # Pass model from cache
    clip,   # Pass clip from cache
    vae,    # Pass vae from cache
    prompt_text: str = "high quality",
    negative_text: str = "",
    max_side: int = 1024,
    steps: int = 20,
    cfg: float = 8.0,
    grow_mask_by: int = 6,
    denoise: float = 1.0,
):
    """
    Returns a PIL RGB image using pre-loaded model components.
    """
    orig_w, orig_h = image_pil.size

    # Defensive resize to avoid huge VRAM usage
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

    clip_encoder = CLIPTextEncode()
    vae_inpaint = VAEEncodeForInpaint()
    ksampler = KSampler()
    vae_decoder = VAEDecode()

    # Inference mode + autocast reduces VRAM and speeds up
    with torch.inference_mode():
        # autocast ON for CLIP text encode
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(torch.cuda.is_available())):
            cond_pos = clip_encoder.encode(clip=clip, text=prompt_text)[0]
            cond_neg = clip_encoder.encode(clip=clip, text=negative_text)[0]

        # autocast OFF for inpaint encode (mask conv2d wants float32)
        with torch.autocast(device_type="cuda", enabled=False):
            latent = vae_inpaint.encode(
                pixels=image_tensor,
                mask=mask_tensor,
                vae=vae,
                grow_mask_by=int(grow_mask_by),
            )[0]

        # autocast ON for sampler + decode
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(torch.cuda.is_available())):
            seed = torch.randint(0, 10_000_000, (1,), device="cpu").item()
            latent_out = ksampler.sample(
                model=model,
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

            decoded = vae_decoder.decode(samples=latent_out, vae=vae)[0]
            img_u8 = (decoded[0].clamp(0, 1) * 255.0).to(torch.uint8)
            out_np = img_u8.cpu().numpy()

    # Cleanup intermediate tensors (but keep cached models)
    del image_tensor, mask_tensor, cond_pos, cond_neg, latent, latent_out, decoded, img_u8
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_pil = Image.fromarray(out_np)

    # Resize back to original resolution
    if out_pil.size != (orig_w, orig_h):
        out_pil = out_pil.resize((orig_w, orig_h), Image.LANCZOS)

    return out_pil