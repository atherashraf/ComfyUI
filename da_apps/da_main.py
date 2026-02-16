# da_main.py
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import base64
import traceback
from PIL import Image
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from da_apps.scripts.inpainting import run_inpainting_comfy
from da_apps.scripts.smart_model_cache import SmartModelCache, DEVICE

# Import from our modules


# Initialize cache
model_cache = SmartModelCache(max_cache_size=2)


# Pydantic Models
class ImageMaskRequest(BaseModel):
    image: str  # data:image/png;base64,...
    mask: str  # data:image/png;base64,...
    positive_prompt: str
    negative_prompt: Optional[str] = ""
    checkpoint_file: str = "512-inpainting-ema.safetensors"
    checkpoint_name: Optional[str] = "512 Inpainting"


# Helper Functions
def _split_data_url(data_url: str):
    """Split data:image/png;base64,... into mime and base64"""
    if not data_url.startswith("data:"):
        return None, data_url
    header, data = data_url.split(",", 1)
    mime = header.split(":")[1].split(";")[0]
    return mime, data


def _b64_to_bytes(b64: str) -> bytes:
    """Base64 to bytes"""
    # Handle URL-safe base64
    b64 = b64.replace('-', '+').replace('_', '/')
    # Add padding if needed
    missing_padding = len(b64) % 4
    if missing_padding:
        b64 += '=' * (4 - missing_padding)
    return base64.b64decode(b64)


def _bytes_to_data_url(data: bytes, mime: str) -> str:
    """Bytes to data URL"""
    b64 = base64.b64encode(data).decode('utf-8')
    return f"data:{mime};base64,{b64}"


# FastAPI App
app = FastAPI(title="Inpainting API with Smart Cache")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
input_dir = Path("tmp_input")
output_dir = Path("tmp_output")
input_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# GPU lock for sequential processing
gpu_lock = asyncio.Lock()


@app.post("/api/image-mask")
async def image_mask(payload: ImageMaskRequest):
    try:
        job_id = uuid.uuid4().hex[:8]
        # print(f"\n[{job_id}] Starting job with checkpoint: {payload.checkpoint_file}")
        # print(f"[{job_id}] Full payload: {payload.model_dump_json()}")  # ADD THIS LINE

        img_path = input_dir / f"{job_id}_image.png"
        mask_path = input_dir / f"{job_id}_mask.png"
        out_path = output_dir / f"{job_id}_result.jpg"

        # 1) Decode & Save raw files
        img_mime, img_b64 = _split_data_url(payload.image)
        mask_mime, mask_b64 = _split_data_url(payload.mask)

        img_path.write_bytes(_b64_to_bytes(img_b64))
        mask_path.write_bytes(_b64_to_bytes(mask_b64))

        # 2) Load IMAGE safely
        with Image.open(img_path) as im:
            im.load()
            # print(f"[{job_id}] Input Image: {im.mode} {im.size}")

            if im.mode == "RGBA":
                r, g, b, a = im.split()
                img = Image.merge("RGB", (r, g, b))
                final_img_path = input_dir / f"{job_id}_image_fixed.jpg"
                img.save(final_img_path, format="JPEG", quality=100)
                # print(f"[{job_id}] Converted RGBA -> RGB")
            else:
                img = im.convert("RGB")
                final_img_path = img_path

        # 3) Load MASK safely
        with Image.open(mask_path) as mk:
            mk.load()
            mask = mk.convert("L")

        mask.save(mask_path)

        # Optional: reject insanely large inputs early
        w, h = img.size
        if max(w, h) > 4096:
            raise HTTPException(status_code=413, detail="Image too large. Please upload <= 4K.")

        # 4) Run Inpainting (locked to 1 GPU job at a time)
        print(f"[{job_id}] Starting Inpainting")
        async with gpu_lock:
            # Load model from cache
            model, clip, vae = model_cache.load_checkpoint(payload.checkpoint_file)

            # Run inpainting
            result_pil = run_inpainting_comfy(
                image_pil=img,
                mask_pil=mask,
                model=model,
                clip=clip,
                vae=vae,
                prompt_text=payload.positive_prompt,
                negative_text=payload.negative_prompt or "",
                max_side=1024,
                steps=20,
                cfg=8.0,
            )

        # 5) Save & Return
        result_pil.save(out_path, format="JPEG", quality=95)
        print(f"[{job_id}] ✓ Finished. Saved to {out_path}")

        out_bytes = out_path.read_bytes()

        return {
            "job_id": job_id,
            "checkpoint_used": payload.checkpoint_file,
            "image": _bytes_to_data_url(out_bytes, "image/jpeg"),
            "cache_info": model_cache.get_cache_info(),
            "saved": {
                "image_path": str(final_img_path),
                "result_path": str(out_path)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/available-checkpoints")
async def get_available_checkpoints():
    """Return list of available checkpoints"""
    checkpoints = model_cache.list_available_checkpoints()

    return {
        "checkpoints_dir": str(model_cache.checkpoints_dir),
        "checkpoints": checkpoints,
        "count": len(checkpoints),
        "cache_info": model_cache.get_cache_info()
    }


@app.post("/api/clear-cache")
async def clear_cache():
    """Clear model cache to free VRAM"""
    model_cache.clear_cache()
    return {"message": "Model cache cleared", "cache_info": model_cache.get_cache_info()}


@app.get("/api/cache-info")
async def cache_info():
    """Get cache information"""
    return model_cache.get_cache_info()


@app.post("/api/preload-checkpoint")
async def preload_checkpoint(checkpoint_file: str):
    """Preload a specific checkpoint into cache"""
    try:
        model_cache.load_checkpoint(checkpoint_file)
        return {
            "message": f"Checkpoint {checkpoint_file} loaded into cache",
            "cache_info": model_cache.get_cache_info()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cache_info": model_cache.get_cache_info()
    }

# ----------------------------
# Serve PhotoAI Studio build
# ----------------------------
PS_DIST_DIR = Path(__file__).resolve().parent / "photoai-studio" / "dist"

if PS_DIST_DIR.exists():
    # Serve assets (js/css/images) under /api/ps
    app.mount("/api/ps", StaticFiles(directory=str(PS_DIST_DIR), html=True), name="photoai-studio")

    # SPA fallback: deep links like /api/ps/settings should serve index.html
    @app.get("/api/ps/{full_path:path}")
    async def ps_spa_fallback(full_path: str):
        index_file = PS_DIST_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="PhotoAI Studio build not found (index.html missing).")
else:
    print(f"[WARN] PhotoAI Studio dist not found at: {PS_DIST_DIR}")


if __name__ == "__main__":
    import uvicorn
    import torch

    print("\n" + "=" * 50)
    print("Starting Inpainting API Server")
    print("=" * 50)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    print(f"Cache size: {model_cache.max_cache_size} models")
    print(f"Checkpoints dir: {model_cache.checkpoints_dir}")

    # List available checkpoints on startup
    checkpoints = model_cache.list_available_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")
    for cp in checkpoints[:5]:  # Show first 5
        print(f"  - {cp['file']} ({cp['size_mb']:.1f} MB)")
    if len(checkpoints) > 5:
        print(f"  ... and {len(checkpoints) - 5} more")

    print("=" * 50)
    print("Server ready on http://0.0.0.0:8000")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)