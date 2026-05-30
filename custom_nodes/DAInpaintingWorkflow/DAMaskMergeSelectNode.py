import torch
import torch.nn.functional as F


class DAMaskMergeSelectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "mask_mode": ([
                    "manual_mask",
                    "person_mask",
                    "face_mask",
                    "outpaint_mask",
                    "manual_plus_person",
                    "manual_plus_face",
                    "person_plus_face",
                    "all_masks"
                ], {"default": "manual_mask"}),

                "invert_mask": ("BOOLEAN", {"default": False}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 100}),
                "grow": ("INT", {"default": 0, "min": -100, "max": 100}),

                "outpaint_left": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "outpaint_top": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "outpaint_right": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "outpaint_bottom": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "manual_mask": ("MASK",),
                "person_mask": ("MASK",),
                "face_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "DA Nodes/Masking"

    def empty_mask(self, image):
        _, h, w, _ = image.shape
        return torch.zeros((h, w), dtype=torch.float32, device=image.device)

    def resize_mask_to_image(self, mask, image):
        _, h, w, _ = image.shape

        if mask is None:
            return self.empty_mask(image)

        mask = mask.float().to(image.device)

        if mask.dim() == 4:
            mask = mask[0, :, :, 0]
        elif mask.dim() == 3:
            mask = mask[0]
        elif mask.dim() != 2:
            return self.empty_mask(image)

        mask = mask.unsqueeze(0).unsqueeze(0)

        mask = F.interpolate(
            mask,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        return mask.squeeze(0).squeeze(0).clamp(0, 1)

    def create_outpaint_mask(self, image, left, top, right, bottom):
        _, h, w, _ = image.shape
        mask = torch.zeros((h, w), dtype=torch.float32, device=image.device)

        left = max(0, min(left, w))
        right = max(0, min(right, w))
        top = max(0, min(top, h))
        bottom = max(0, min(bottom, h))

        if left > 0:
            mask[:, :left] = 1.0
        if right > 0:
            mask[:, w - right:] = 1.0
        if top > 0:
            mask[:top, :] = 1.0
        if bottom > 0:
            mask[h - bottom:, :] = 1.0

        return mask.clamp(0, 1)

    def combine_masks(self, masks):
        result = torch.zeros_like(masks[0])
        for mask in masks:
            result = torch.maximum(result, mask)
        return result.clamp(0, 1)

    def grow_or_erode(self, mask, amount):
        if amount == 0:
            return mask

        kernel_size = abs(amount) * 2 + 1
        padding = kernel_size // 2

        mask_4d = mask.unsqueeze(0).unsqueeze(0)

        if amount > 0:
            result = F.max_pool2d(
                mask_4d,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
        else:
            inverted = 1.0 - mask_4d
            result = F.max_pool2d(
                inverted,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
            result = 1.0 - result

        return result.squeeze(0).squeeze(0).clamp(0, 1)

    def blur_mask(self, mask, blur):
        if blur <= 0:
            return mask

        kernel_size = blur * 2 + 1
        padding = blur

        mask_4d = mask.unsqueeze(0).unsqueeze(0)

        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size),
            dtype=mask_4d.dtype,
            device=mask_4d.device
        ) / float(kernel_size * kernel_size)

        blurred = F.conv2d(mask_4d, kernel, padding=padding)

        return blurred.squeeze(0).squeeze(0).clamp(0, 1)

    def process(
        self,
        image,
        mask_mode,
        invert_mask=False,
        blur=0,
        grow=0,
        outpaint_left=0,
        outpaint_top=0,
        outpaint_right=0,
        outpaint_bottom=0,
        manual_mask=None,
        person_mask=None,
        face_mask=None,
    ):
        manual = self.resize_mask_to_image(manual_mask, image)
        person = self.resize_mask_to_image(person_mask, image)
        face = self.resize_mask_to_image(face_mask, image)

        outpaint = self.create_outpaint_mask(
            image,
            outpaint_left,
            outpaint_top,
            outpaint_right,
            outpaint_bottom
        )

        if mask_mode == "manual_mask":
            final_mask = manual
        elif mask_mode == "person_mask":
            final_mask = person
        elif mask_mode == "face_mask":
            final_mask = face
        elif mask_mode == "outpaint_mask":
            final_mask = outpaint
        elif mask_mode == "manual_plus_person":
            final_mask = self.combine_masks([manual, person])
        elif mask_mode == "manual_plus_face":
            final_mask = self.combine_masks([manual, face])
        elif mask_mode == "person_plus_face":
            final_mask = self.combine_masks([person, face])
        else:
            final_mask = self.combine_masks([manual, person, face, outpaint])

        final_mask = self.grow_or_erode(final_mask, grow)
        final_mask = self.blur_mask(final_mask, blur)

        if invert_mask:
            final_mask = 1.0 - final_mask

        return image, final_mask.clamp(0, 1)