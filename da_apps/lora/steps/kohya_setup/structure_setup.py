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

    concept_folder.mkdir(parents=True, exist_ok=True)

    # Get all images
    images = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        images.extend(source_path.glob(f'*{ext}'))
        images.extend(source_path.glob(f'*{ext.upper()}'))

    print(f"Preparing {len(images)} images for Kohya_SS...")

    # Copy files
    for i, img_path in enumerate(sorted(images)):
        # Image file
        img_ext = img_path.suffix.lower()
        new_img_name = f"{concept_name}_{i:04d}{img_ext}"
        new_img_path = concept_folder / new_img_name

        # Copy image
        shutil.copy2(img_path, new_img_path)

        # Caption file
        caption_path = source_path / f"{img_path.stem}.txt"
        if caption_path.exists():
            new_caption_name = f"{concept_name}_{i:04d}.txt"
            new_caption_path = concept_folder / new_caption_name
            shutil.copy2(caption_path, new_caption_path)
            print(f"✓ {img_path.name} -> {new_img_name}")
        else:
            # Create default caption
            default_caption = f"photo of {concept_name}, full body"
            new_caption_name = f"{concept_name}_{i:04d}.txt"
            new_caption_path = concept_folder / new_caption_name
            with open(new_caption_path, 'w', encoding='utf-8') as f:
                f.write(default_caption)
            print(f"✓ {img_path.name} -> {new_img_name} (created caption)")

    print(f"\nKohya_SS dataset prepared at: {concept_folder}")
    print(f"Total images: {len(images)}")

    return dataset_parent  # Return the PARENT folder