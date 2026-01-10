# comfy_pose_control.py
import torch
import comfy.utils
import nodes
import folder_paths
from PIL import Image
import numpy as np
import cv2
import json
from comfy_lora_trainer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# from .controlnet import ControlNetLoader, ControlNetApplyAdvanced


class PoseExtractorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_hands": ("BOOLEAN", {"default": True}),
                "detect_face": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    RETURN_NAMES = ("pose_image", "pose_data")
    FUNCTION = "extract_pose"
    CATEGORY = "image/pose"

    def extract_pose(self, image, detect_hands, detect_face):
        # Convert tensor to numpy
        img_np = 255. * image[0].cpu().numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # Extract pose using OpenPose (simplified)
        pose_img, pose_data = self.extract_openpose(img_np, detect_hands, detect_face)

        # Convert back to tensor
        pose_tensor = torch.from_numpy(pose_img).float() / 255.0
        pose_tensor = pose_tensor.unsqueeze(0)

        return (pose_tensor, pose_data)

    def extract_openpose(self, image, detect_hands, detect_face):
        # Simplified pose extraction
        # In production, use actual OpenPose or MediaPipe

        # Create skeleton visualization
        height, width = image.shape[:2]
        pose_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Mock pose data
        pose_data = {
            "people": [{
                "pose_keypoints_2d": self.generate_mock_keypoints(width, height),
                "face_keypoints_2d": [] if not detect_face else self.generate_face_keypoints(width, height),
                "hand_left_keypoints_2d": [] if not detect_hands else self.generate_hand_keypoints(width, height,
                                                                                                   "left"),
                "hand_right_keypoints_2d": [] if not detect_hands else self.generate_hand_keypoints(width, height,
                                                                                                    "right"),
            }]
        }

        # Draw skeleton (simplified)
        self.draw_skeleton(pose_img, pose_data["people"][0]["pose_keypoints_2d"])

        return pose_img, pose_data

    def generate_mock_keypoints(self, width, height):
        # Generate mock keypoints for demonstration
        return [width // 2, height // 2, 1.0] * 25  # 25 COCO keypoints

    def draw_skeleton(self, image, keypoints):
        # Draw simple skeleton lines
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i + 3]
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)


class StyleTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "style_lora": ("LORA",),
                "style_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "preserve_identity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "person_lora": ("LORA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_style"
    CATEGORY = "image/style"

    def apply_style(self, image, style_lora, style_strength, preserve_identity, person_lora=None):
        # Apply style transfer using LoRAs
        # This is a simplified version - actual implementation would use diffusion

        # Convert to PIL
        img_np = 255. * image[0].cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

        # Apply style (simplified - in production, use actual model inference)
        styled_image = self.style_transfer_pipeline(
            img_pil,
            style_lora,
            style_strength,
            person_lora,
            preserve_identity
        )

        # Convert back to tensor
        styled_np = np.array(styled_image).astype(np.float32) / 255.0
        styled_tensor = torch.from_numpy(styled_np).unsqueeze(0)

        return (styled_tensor,)

    def style_transfer_pipeline(self, image, style_lora, style_strength, person_lora, identity_strength):
        # Placeholder for actual style transfer
        return image


# Register nodes
NODE_CLASS_MAPPINGS.update({
    "PoseExtractorNode": PoseExtractorNode,
    "StyleTransferNode": StyleTransferNode,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "PoseExtractorNode": "Extract Pose",
    "StyleTransferNode": "Style Transfer with LoRA",
})