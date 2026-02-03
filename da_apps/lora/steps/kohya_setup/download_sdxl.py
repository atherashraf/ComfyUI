import os
from huggingface_hub import hf_hub_download


def download_sdxl_models(comfy_root):
    """
    Checks if SDXL Base AND Refiner exist in the ComfyUI checkpoints folder.
    Downloads them automatically if missing.
    """
    checkpoint_dir = os.path.join(comfy_root, "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # List of models to check
    models_to_download = [
        {
            "name": "SDXL Base 1.0",
            "filename": "sd_xl_base_1.0.safetensors",
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0"
        },
        {
            "name": "SDXL Refiner 1.0",
            "filename": "sd_xl_refiner_1.0.safetensors",
            "repo_id": "stabilityai/stable-diffusion-xl-refiner-1.0"
        }
    ]

    for model in models_to_download:
        file_path = os.path.join(checkpoint_dir, model["filename"])

        if os.path.exists(file_path):
            print(f"✅ {model['name']} found at: {file_path}")
        else:
            print(f"⚠️ {model['name']} NOT found. Downloading now... (Please wait)")
            try:
                hf_hub_download(
                    repo_id=model["repo_id"],
                    filename=model["filename"],
                    local_dir=checkpoint_dir,
                    local_dir_use_symlinks=False
                )
                print(f"🎉 {model['name']} download complete!")
            except Exception as e:
                print(f"❌ Error downloading {model['name']}: {e}")


if __name__ == "__main__":
    # Change this path if your ComfyUI is somewhere else
    default_comfy_root = r"D:\DevProjects\PhotoAI\ComfyUI"
    download_sdxl_models(default_comfy_root)