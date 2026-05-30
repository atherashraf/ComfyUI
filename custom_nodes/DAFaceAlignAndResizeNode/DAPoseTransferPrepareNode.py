import torch
import cv2
import numpy as np


class DAPoseTransferPrepareNode:
    """
    Prepares source and target images for pose-transfer workflows.

    This node does not generate the final pose-changed image.
    It prepares:
    - source image resized to target size
    - target pose reference image
    - target object/person mask
    - positive prompt
    - negative prompt

    Use outputs with:
    - Flux Kontext
    - ControlNet OpenPose / DWPose
    - Inpaint nodes
    - IPAdapter
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "mode": (["human_pose", "object_pose"], {"default": "human_pose"}),
                "padding_color": (["black", "white", "edge_extend"], {"default": "black"}),
                "mask_method": (["full", "target_foreground", "center_soft"], {"default": "center_soft"}),
                "prompt_subject": ("STRING", {
                    "default": "the subject from the source image",
                    "multiline": False
                }),
                "target_description": ("STRING", {
                    "default": "matching the pose and composition of the target image",
                    "multiline": False
                }),
                "strength_hint": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = (
        "source_resized",
        "pose_reference_image",
        "pose_mask",
        "positive_prompt",
        "negative_prompt"
    )

    FUNCTION = "process"
    CATEGORY = "DA Nodes/Pose"

    def tensor_to_cv2(self, tensor_image):
        if tensor_image.dim() == 4:
            tensor_image = tensor_image[0]

        image = tensor_image.detach().cpu().numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        if image.shape[-1] == 4:
            image = image[:, :, :3]

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def cv2_to_tensor(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        return tensor.unsqueeze(0)

    def resize_with_padding(self, image, target_size, padding_color="black"):
        target_w, target_h = target_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top

        if padding_color == "edge_extend":
            return cv2.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE
            )

        if padding_color == "white":
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        else:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas

    def create_full_mask(self, h, w):
        return torch.ones((h, w), dtype=torch.float32)

    def create_center_soft_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.float32)

        x1 = int(w * 0.12)
        x2 = int(w * 0.88)
        y1 = int(h * 0.05)
        y2 = int(h * 0.95)

        mask[y1:y2, x1:x2] = 1.0

        blur = max(31, min(h, w) // 12)
        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        mask = np.clip(mask, 0.0, 1.0)

        return torch.from_numpy(mask).float()

    def create_foreground_mask(self, image):
        """
        Simple OpenCV foreground-ish mask.
        Works best when target subject/object differs from background.
        """
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros((h, w), dtype=np.float32)

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours[:5]:
                if cv2.contourArea(c) > (h * w * 0.01):
                    cv2.drawContours(mask, [c], -1, 1.0, thickness=-1)

        if mask.sum() == 0:
            return self.create_center_soft_mask(h, w)

        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        blur = max(21, min(h, w) // 20)
        if blur % 2 == 0:
            blur += 1

        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        mask = np.clip(mask, 0.0, 1.0)

        return torch.from_numpy(mask).float()

    def build_prompts(self, mode, prompt_subject, target_description, strength_hint):
        if mode == "human_pose":
            positive = (
                f"Transform {prompt_subject} to match the human pose, body position, "
                f"gesture, camera angle, and composition of the target reference. "
                f"Preserve the identity, clothing style, colors, and visual details of the source subject. "
                f"{target_description}. Pose transfer strength {strength_hint:.2f}."
            )

            negative = (
                "extra limbs, missing limbs, broken anatomy, distorted hands, distorted face, "
                "wrong body pose, duplicate person, bad proportions, blurry, low quality"
            )
        else:
            positive = (
                f"Transform {prompt_subject} to match the object pose, orientation, scale, "
                f"perspective, and composition of the target reference. "
                f"Preserve the source object's identity, material, color, texture, and details. "
                f"{target_description}. Pose transfer strength {strength_hint:.2f}."
            )

            negative = (
                "deformed object, wrong shape, broken geometry, duplicate object, distorted perspective, "
                "missing parts, extra parts, blurry, low quality"
            )

        return positive, negative

    def process(
        self,
        source_image,
        target_image,
        mode="human_pose",
        padding_color="black",
        mask_method="center_soft",
        prompt_subject="the subject from the source image",
        target_description="matching the pose and composition of the target image",
        strength_hint=0.75
    ):
        src_img = self.tensor_to_cv2(source_image)
        tgt_img = self.tensor_to_cv2(target_image)

        target_h, target_w = tgt_img.shape[:2]

        source_resized = self.resize_with_padding(
            src_img,
            (target_w, target_h),
            padding_color
        )

        pose_reference = tgt_img.copy()

        if mask_method == "full":
            mask = self.create_full_mask(target_h, target_w)
        elif mask_method == "target_foreground":
            mask = self.create_foreground_mask(tgt_img)
        else:
            mask = self.create_center_soft_mask(target_h, target_w)

        positive, negative = self.build_prompts(
            mode,
            prompt_subject,
            target_description,
            strength_hint
        )

        return (
            self.cv2_to_tensor(source_resized),
            self.cv2_to_tensor(pose_reference),
            mask,
            positive,
            negative
        )