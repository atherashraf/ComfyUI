# comfy_lora_trainer.py
import torch
import folder_paths
import comfy.utils
import nodes
from PIL import Image
import json
import os
from pathlib import Path
import numpy as np


class LoRATrainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_images": ("IMAGE",),
                "instance_token": ("STRING", {"default": "shs_person"}),
                "class_token": ("STRING", {"default": "person"}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "num_epochs": ("INT", {"default": 10, "min": 1, "max": 100}),
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-5, "max": 1e-3}),
                "lora_rank": ("INT", {"default": 128, "min": 4, "max": 256}),
                "lora_alpha": ("INT", {"default": 32, "min": 1, "max": 128}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "captions": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)  # Returns path to LoRA file
    RETURN_NAMES = ("lora_path",)
    FUNCTION = "train"
    CATEGORY = "training"

    def train(self, training_images, instance_token, class_token, resolution,
              batch_size, num_epochs, learning_rate, lora_rank, lora_alpha,
              model=None, clip=None, captions=""):

        # Create output directory
        output_dir = folder_paths.get_output_directory()
        lora_dir = os.path.join(output_dir, "loras")
        os.makedirs(lora_dir, exist_ok=True)

        # Prepare images
        images = self.prepare_images(training_images, resolution)

        # Get captions
        caption_list = self.parse_captions(captions, len(images), instance_token, class_token)

        # Train LoRA
        lora_path = self.train_lora(
            images, caption_list, instance_token,
            lora_rank, lora_alpha, num_epochs, learning_rate
        )

        return (lora_path,)

    def prepare_images(self, images, resolution):
        """Prepare images for training"""
        prepared = []
        for img in images:
            # Convert tensor to PIL
            img_np = 255. * img.cpu().numpy().squeeze()
            img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

            # Resize
            img_pil = img_pil.resize((resolution, resolution), Image.Resampling.LANCZOS)

            # Convert to tensor for training
            img_tensor = torch.from_numpy(np.array(img_pil)).float() / 127.5 - 1.0
            prepared.append(img_tensor.unsqueeze(0))

        return torch.cat(prepared, dim=0)

    def parse_captions(self, captions, num_images, instance_token, class_token):
        """Parse captions from string or generate default"""
        if captions:
            lines = captions.strip().split('\n')
            if len(lines) >= num_images:
                return lines[:num_images]

        # Generate default captions
        return [f"{instance_token} {class_token}" for _ in range(num_images)]

    def train_lora(self, images, captions, instance_token, rank, alpha, epochs, lr):
        """Simplified LoRA training (in production, use Kohya or similar)"""
        # Note: Full training requires Kohya SS integration
        # This is a simplified version for demonstration

        print(f"Training LoRA with {len(images)} images")
        print(f"Instance token: {instance_token}")
        print(f"Rank: {rank}, Alpha: {alpha}")
        print(f"Epochs: {epochs}, LR: {lr}")

        # For actual training, you'd integrate Kohya SS or diffusers training
        # For now, create a dummy LoRA file structure
        lora_path = self.create_dummy_lora(instance_token, rank, alpha)

        return lora_path

    def create_dummy_lora(self, name, rank, alpha):
        """Create a placeholder LoRA file"""
        import safetensors.torch

        # Create minimal LoRA structure
        lora_dict = {
            f"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": torch.randn(rank, 320),
            f"lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": torch.randn(320, rank),
            "alpha": torch.tensor([alpha])
        }

        # Save to file
        lora_name = f"{name}_lora_{rank}_{alpha}.safetensors"
        lora_path = os.path.join(folder_paths.get_output_directory(), "loras", lora_name)

        safetensors.torch.save_file(lora_dict, lora_path)

        print(f"Created dummy LoRA at: {lora_path}")
        return lora_path


# Register node
NODE_CLASS_MAPPINGS = {
    "LoRATrainer": LoRATrainer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRATrainer": "Train LoRA for Person"
}