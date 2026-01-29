from typing import Optional

from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from da_apps.da_settings import input_dir, static_dir, output_dir, BASE_DIR
import base64
import uuid
import asyncio

from da_apps.scripts.inpainting import run_inpainting_comfy

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for testing; later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRITICAL: prevent concurrent GPU runs (8GB VRAM will OOM under load)
gpu_lock = asyncio.Lock()


class ImageMaskRequest(BaseModel):
    image: str  # data:image/png;base64,...
    mask: str   # data:image/png;base64,...
    positive_prompt:str
    negative_prompt:Optional[str]


def _split_data_url(data_url: str):
    if not data_url.startswith("data:") or "base64," not in data_url:
        raise ValueError("Expected a data URL with base64, e.g. data:image/png;base64,...")
    header, b64 = data_url.split("base64,", 1)
    mime = header[5:].split(";", 1)[0]  # "image/png"
    return mime, b64


def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"), validate=False)


def _bytes_to_data_url(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@app.post("/api/image-mask")
async def image_mask(payload: ImageMaskRequest):
    try:
        job_id = uuid.uuid4().hex

        img_path = input_dir / f"{job_id}_image.png"
        mask_path = input_dir / f"{job_id}_mask.png"
        out_path = output_dir / f"{job_id}_result.jpg"

        # 1) Decode & Save raw files
        img_mime, img_b64 = _split_data_url(payload.image)
        mask_mime, mask_b64 = _split_data_url(payload.mask)

        img_path.write_bytes(_b64_to_bytes(img_b64))
        mask_path.write_bytes(_b64_to_bytes(mask_b64))

        # 2) Load IMAGE safely (and close file handle)
        with Image.open(img_path) as im:
            im.load()
            print(f"[{job_id}] Input Image Mode: {im.mode}")

            if im.mode == "RGBA":
                r, g, b, a = im.split()
                img = Image.merge("RGB", (r, g, b))
                final_img_path = input_dir / f"{job_id}_image_fixed.jpg"
                img.save(final_img_path, format="JPEG", quality=100)
                print(f"[{job_id}] Converted RGBA -> RGB (Alpha dropped)")
            else:
                img = im.convert("RGB")
                final_img_path = img_path

        # 3) Load MASK safely (and close file handle)
        with Image.open(mask_path) as mk:
            mk.load()
            mask = mk.convert("L")

        # Optional but recommended: threshold mask to clean edges
        # mask = mask.point(lambda p: 255 if p > 128 else 0)

        mask.save(mask_path)

        # Optional: reject insanely large inputs early (prevents VRAM blowups)
        w, h = img.size
        if max(w, h) > 4096:
            raise HTTPException(status_code=413, detail="Image too large. Please upload <= 4K.")

        # 4) Run Inpainting (locked to 1 GPU job at a time)
        print(f"[{job_id}] Starting Inpainting...")
        async with gpu_lock:
            result_pil = run_inpainting_comfy(
                image_pil=img,
                mask_pil=mask,
                # prompt_text="high quality, seamless blend",
                prompt_text=payload.positive_prompt,
                # below parameters are for memory management
                # max_side=1024,   # on 8GB VRAM: 768 if you still see OOM
                # steps=16,        # safer than 20 on 8GB
                # cfg=7.0
            )

        # 5) Save & Return
        result_pil.save(out_path, format="JPEG", quality=95)
        print(f"[{job_id}] Finished. Saved to {out_path}")

        out_bytes = out_path.read_bytes()

        return {
            "job_id": job_id,
            "image": _bytes_to_data_url(out_bytes, "image/jpeg"),
            "saved": {
                "image_path": str(final_img_path),
                "result_path": str(out_path)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
#
# BASE_DIR = Path(__file__).resolve().parent
REACT_BUILD = BASE_DIR / "photoai-studio" / "dist"
print(REACT_BUILD)
# Serve the React app
app.mount(
    "/",
    StaticFiles(directory=str(REACT_BUILD), html=True),
    name="react"
)
@app.get("/")
def root():
    return RedirectResponse(url="/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
