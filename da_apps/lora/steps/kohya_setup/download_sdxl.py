from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import requests
from tqdm import tqdm


SDXL_MODELS: Dict[str, str] = {
    "sd_xl_base_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "sd_xl_refiner_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
}


def download_file(url: str, out_path: Path, retries: int = 3, timeout: int = 30) -> None:
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with open(out_path, "wb") as f, tqdm(
                    desc=out_path.name, total=total, unit="iB", unit_scale=True, unit_divisor=1024
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))
            return
        except Exception as e:
            if attempt == retries:
                raise
            print(f"⚠ Download failed ({e}), retrying ({attempt}/{retries})...")
            time.sleep(2 * attempt)


def download_sdxl_models(model_dir: str = "models") -> Path:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    for filename, url in SDXL_MODELS.items():
        fp = model_path / filename
        if fp.exists():
            print(f"✓ {filename} already exists")
            continue
        print(f"Downloading {filename}...")
        download_file(url, fp)

    print(f"\nModels saved to: {model_path}")
    return model_path
