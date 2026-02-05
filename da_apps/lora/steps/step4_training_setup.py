from pathlib import Path

from da_apps.lora.steps.kohya_setup.auto_training import train_sdxl_lora_auto
from da_apps.lora.steps.kohya_setup.download_sdxl import download_sdxl_models
from da_apps.lora.steps.kohya_setup.kohya_config import generate_kohya_config
from da_apps.lora.steps.kohya_setup.structure_setup import prepare_kohya_structure

"""
    For your first full-body person LoRA, use Kohya_SS GUI. It's the most stable and has the best documentation. Follow these steps:
    
        1. Install Kohya_SS (30 minutes)
        
        2. Download SDXL models (15 minutes, 13GB total)
        
        3. Prepare dataset (5 minutes with our scripts)
        
        4. Configure training (10 minutes with my config)
        
        5. Start training (2-6 hours depending on epochs)
        
         D:/lora_training/
            ├── judi_dataset/
            │   ├── 1024_judy_1/          # 1024x1024 images
            │   │   ├── image_1.jpg
            │   │   ├── image_1.txt       # Caption file
            │   │   ├── image_2.jpg
            │   │   └── image_2.txt
            │   └── 1024_judy_2/          # Another concept if needed
            │
            └── models/
                ├── sd_xl_base_1.0.safetensors
                └── sd_xl_refiner_1.0.safetensors
"""






def run_full_training_pipeline(
        input_images_folder,
        working_dir=r"D:\lora_training",
        concept_name="judi"
):
    print(f"=== Starting LoRA Pipeline for '{concept_name}' ===")

    # Setup Paths
    work_path = Path(working_dir)

    # !!! FIX: Point to where your models actually are !!!
    models_path = work_path / "models"
    base_model_file = models_path / "sd_xl_base_1.0.safetensors"

    # dataset_out = work_path / "dataset"
    output_out = work_path / f"output_{concept_name}"
    logs_out = work_path / "logs"
    config_file = work_path / f"{concept_name}_config.toml"

    # 1. Prepare Dataset
    print("\n--- Step 1: Preparing Dataset ---")
    prepare_kohya_structure(input_images_folder, work_path, concept_name)

    # 2. Check Models
    print(f"\n--- Step 2: Checking Model ---")
    if not base_model_file.exists():
        print(f"⚠️ Model not found at {base_model_file}")
        print("Downloading now...")
        models_path.mkdir(parents=True, exist_ok=True)
        download_sdxl_models(str(models_path))
    else:
        print(f"✓ Found base model: {base_model_file}")

    # 3. Generate Config
    print("\n--- Step 3: Generating Config ---")
    dataset_dir = work_path / f"{concept_name}_dataset"

    generate_kohya_config(
        output_file=str(config_file),
        train_data_dir=dataset_dir,
        output_dir=output_out,
        logging_dir=logs_out,
        model_path=base_model_file,  # Passing the specific file path
        concept_name=concept_name
    )

    # 4. Start Training
    print("\n--- Step 4: Launching Training ---")
    train_sdxl_lora_auto(config_file)


if __name__ == "__main__":
    # Point this to your images
    source_images = r"D:\Documents\general\downloads\pics\judi_cropped"

    run_full_training_pipeline(
        input_images_folder=source_images,
        # concept_name="judi"
        concept_name="ohwx"
    )

