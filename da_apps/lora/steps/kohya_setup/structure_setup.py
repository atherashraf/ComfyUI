from pathlib import Path
import shutil
import os


def prepare_kohya_structure(source_folder, output_folder, concept_name="judi"):
    """
    Prepare folder structure for Kohya_SS training.
    D:/lora_training/
        ├── judi_dataset/
        │   ├── 1_judy/          # 1024x1024 images
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
    source_path = Path(source_folder)
    # Ensure we point to the dataset parent folder
    dataset_parent = Path(output_folder) / f"{concept_name}_dataset"

    # !!! FIX: Changed '1024' to '1'.
    # The number at the start of the folder name determines repeats.
    # 1 repeat * 148 images * 30 epochs = ~4,500 steps (Perfect for LoRA)
    concept_folder = dataset_parent / f"1_{concept_name}"

    if concept_folder.exists():
        print(f"Cleaning up old folder: {concept_folder}")
        shutil.rmtree(concept_folder)
