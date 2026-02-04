import requests
import os
from tqdm import tqdm
from pathlib import Path


def download_sdxl_models(model_dir="models"):
    """Download SDXL models."""
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)

    models = {
        "sd_xl_base_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "sd_xl_refiner_1.0.safetensors": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
    }

    for filename, url in models.items():
        filepath = model_path / filename

        if filepath.exists():
            print(f"✓ {filename} already exists")
            continue

        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

        print(f"✓ Downloaded {filename}")

    print(f"\nModels saved to: {model_path}")


if __name__ == "__main__":
    # Download models
    sdxl_dir = Path(r"D:\lora_training\models")
    sdxl_dir.mkdir(parents=True, exist_ok=True)
    download_sdxl_models(str(sdxl_dir))