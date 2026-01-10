# quick_start.py
"""
Quick start script for person LoRA training and generation
"""
import os

import folder_paths


def train_person_lora():
    """Train a LoRA from ComfyUI"""
    import comfy.sample
    import comfy.utils

    # 1. Collect training images
    training_images = [
        "person_front.jpg",
        "person_side.jpg",
        "person_smile.jpg",
        "person_serious.jpg"
    ]

    # 2. Setup training parameters
    params = {
        "instance_token": "john_doe",
        "class_token": "person",
        "resolution": 512,
        "batch_size": 1,
        "epochs": 10,
        "learning_rate": 1e-4,
        "lora_rank": 128,
        "lora_alpha": 32
    }

    # 3. Train (simplified - use actual training pipeline)
    print(f"Training LoRA for {len(training_images)} images...")

    # 4. Save to ComfyUI models directory
    output_path = os.path.join(folder_paths.models_dir, "loras", "john_doe_lora.safetensors")

    return output_path


def generate_variations(lora_path):
    """Generate person in different styles/poses"""

    styles = [
        ("anime style", "masterpiece, best quality"),
        ("cyberpunk", "neon lights, futuristic city"),
        ("renaissance painting", "oil painting, classical art"),
        ("professional photo", "sharp focus, studio lighting")
    ]

    poses = [
        "standing",
        "sitting",
        "running",
        "looking back"
    ]

    # Generate images
    for style, style_prompt in styles:
        for pose in poses:
            prompt = f"john_doe, {pose}, {style_prompt}, {style}"
            print(f"Generating: {prompt}")

            # Use ComfyUI nodes to generate
            # image = generate_with_comfy(prompt, lora_path)
            # image.save(f"output/{style}_{pose}.png")


if __name__ == "__main__":
    # Train LoRA
    lora_path = train_person_lora()
    print(f"LoRA saved to: {lora_path}")

    # Generate variations
    generate_variations(lora_path)