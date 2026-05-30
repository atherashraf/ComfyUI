import torch
import numpy as np
import cv2


class DAMaskCropToTargetSizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "target_image": ("IMAGE",),

                "padding_color": (
                    ["black", "white"],
                    {"default": "black"}
                ),

                "margin_percent": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01
                    }
                ),

                "fit_mode": (
                    ["contain", "cover"],
                    {"default": "contain"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = (
        "cropped_image",
        "cropped_mask",
        "debug_image"
    )

    FUNCTION = "process"
    CATEGORY = "DA Nodes/Masking"

    def tensor_to_cv2(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]

        img = tensor.cpu().numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        if img.shape[-1] == 4:
            img = img[:, :, :3]

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def cv2_to_tensor(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        return tensor.unsqueeze(0)

    def mask_to_numpy(self, mask):
        if mask.dim() == 3:
            mask = mask[0]

        return mask.cpu().numpy()

    def numpy_to_mask(self, mask):
        return torch.from_numpy(mask.astype(np.float32))

    def get_mask_bbox(self, mask):
        ys, xs = np.where(mask > 0.05)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = xs.min()
        y1 = ys.min()
        x2 = xs.max()
        y2 = ys.max()

        return x1, y1, x2, y2

    def add_margin(self, bbox, w, h, margin_percent):
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        mx = int(bw * margin_percent)
        my = int(bh * margin_percent)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        return x1, y1, x2, y2

    def resize_and_pad(
        self,
        image,
        target_w,
        target_h,
        padding_color,
        fit_mode
    ):
        h, w = image.shape[:2]

        if fit_mode == "cover":
            scale = max(target_w / w, target_h / h)
        else:
            scale = min(target_w / w, target_h / h)

        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        resized = cv2.resize(
            image,
            (nw, nh),
            interpolation=cv2.INTER_LANCZOS4
        )

        if padding_color == "white":
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        else:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        x = (target_w - nw) // 2
        y = (target_h - nh) // 2

        x2 = x + nw
        y2 = y + nh

        canvas[y:y2, x:x2] = resized

        return canvas

    def resize_mask_and_pad(
        self,
        mask,
        target_w,
        target_h,
        fit_mode
    ):
        h, w = mask.shape[:2]

        if fit_mode == "cover":
            scale = max(target_w / w, target_h / h)
        else:
            scale = min(target_w / w, target_h / h)

        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        resized = cv2.resize(
            mask,
            (nw, nh),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.zeros((target_h, target_w), dtype=np.float32)

        x = (target_w - nw) // 2
        y = (target_h - nh) // 2

        x2 = x + nw
        y2 = y + nh

        canvas[y:y2, x:x2] = resized

        return np.clip(canvas, 0, 1)

    def process(
        self,
        image,
        mask,
        target_image,
        padding_color="black",
        margin_percent=0.15,
        fit_mode="contain"
    ):
        src = self.tensor_to_cv2(image)
        mask_np = self.mask_to_numpy(mask)

        target = self.tensor_to_cv2(target_image)

        th, tw = target.shape[:2]
        sh, sw = src.shape[:2]

        bbox = self.get_mask_bbox(mask_np)

        if bbox is None:
            blank = np.zeros((th, tw, 3), dtype=np.uint8)
            blank_mask = np.zeros((th, tw), dtype=np.float32)

            return (
                self.cv2_to_tensor(blank),
                self.numpy_to_mask(blank_mask),
                self.cv2_to_tensor(blank)
            )

        x1, y1, x2, y2 = self.add_margin(
            bbox,
            sw,
            sh,
            margin_percent
        )

        cropped_image = src[y1:y2, x1:x2]
        cropped_mask = mask_np[y1:y2, x1:x2]

        final_image = self.resize_and_pad(
            cropped_image,
            tw,
            th,
            padding_color,
            fit_mode
        )

        final_mask = self.resize_mask_and_pad(
            cropped_mask,
            tw,
            th,
            fit_mode
        )

        debug = final_image.copy()

        return (
            self.cv2_to_tensor(final_image),
            self.numpy_to_mask(final_mask),
            self.cv2_to_tensor(debug)
        )