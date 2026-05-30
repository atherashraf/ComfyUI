import torch
import cv2
import numpy as np


class DAPersonAlignToTargetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "padding_color": (["black", "white", "edge_extend"], {"default": "black"}),
                "background_threshold": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "margin_percent": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "fit_mode": (["contain", "cover"], {"default": "contain"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "debug_image")
    FUNCTION = "process"
    CATEGORY = "DA Nodes/Person"

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

    def numpy_to_mask(self, mask):
        return torch.from_numpy(mask.astype(np.float32)).clamp(0, 1)

    def detect_object_mask(self, image, threshold):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Assumes masked object is on black/near-black background.
        mask = (gray > threshold).astype(np.float32)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def get_bbox(self, mask):
        ys, xs = np.where(mask > 0.05)

        if len(xs) == 0 or len(ys) == 0:
            return None

        return xs.min(), ys.min(), xs.max(), ys.max()

    def add_margin(self, bbox, img_w, img_h, margin_percent):
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        mx = int(bw * margin_percent)
        my = int(bh * margin_percent)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(img_w, x2 + mx)
        y2 = min(img_h, y2 + my)

        return x1, y1, x2, y2

    def make_canvas(self, h, w, padding_color):
        if padding_color == "white":
            return np.ones((h, w, 3), dtype=np.uint8) * 255
        return np.zeros((h, w, 3), dtype=np.uint8)

    def align_source_to_target(self, source, source_bbox, target_bbox, target_size, padding_color, fit_mode):
        target_w, target_h = target_size

        sx1, sy1, sx2, sy2 = source_bbox
        tx1, ty1, tx2, ty2 = target_bbox

        source_crop = source[sy1:sy2, sx1:sx2]

        src_h, src_w = source_crop.shape[:2]
        tgt_w = max(1, tx2 - tx1)
        tgt_h = max(1, ty2 - ty1)

        if fit_mode == "cover":
            scale = max(tgt_w / src_w, tgt_h / src_h)
        else:
            scale = min(tgt_w / src_w, tgt_h / src_h)

        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))

        resized = cv2.resize(source_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = self.make_canvas(target_h, target_w, padding_color)

        target_cx = (tx1 + tx2) // 2
        target_cy = (ty1 + ty2) // 2

        paste_x = int(target_cx - new_w / 2)
        paste_y = int(target_cy - new_h / 2)

        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)

        src_x2 = min(new_w, target_w - paste_x)
        src_y2 = min(new_h, target_h - paste_y)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return canvas

        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

        return canvas

    def create_output_mask(self, image, threshold):
        mask = self.detect_object_mask(image, threshold)
        return self.numpy_to_mask(mask)

    def process(
        self,
        source_image,
        target_image,
        padding_color="black",
        background_threshold=10,
        margin_percent=0.10,
        fit_mode="contain"
    ):
        src = self.tensor_to_cv2(source_image)
        tgt = self.tensor_to_cv2(target_image)

        target_h, target_w = tgt.shape[:2]
        src_h, src_w = src.shape[:2]

        src_mask = self.detect_object_mask(src, background_threshold)
        tgt_mask = self.detect_object_mask(tgt, background_threshold)

        src_bbox = self.get_bbox(src_mask)
        tgt_bbox = self.get_bbox(tgt_mask)

        if src_bbox is None or tgt_bbox is None:
            blank = self.make_canvas(target_h, target_w, padding_color)
            mask = np.zeros((target_h, target_w), dtype=np.float32)
            return self.cv2_to_tensor(blank), self.numpy_to_mask(mask), self.cv2_to_tensor(blank)

        src_bbox = self.add_margin(src_bbox, src_w, src_h, margin_percent)
        tgt_bbox = self.add_margin(tgt_bbox, target_w, target_h, margin_percent)

        aligned = self.align_source_to_target(
            source=src,
            source_bbox=src_bbox,
            target_bbox=tgt_bbox,
            target_size=(target_w, target_h),
            padding_color=padding_color,
            fit_mode=fit_mode
        )

        mask = self.create_output_mask(aligned, background_threshold)

        debug = aligned.copy()
        tx1, ty1, tx2, ty2 = tgt_bbox
        cv2.rectangle(debug, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)

        return self.cv2_to_tensor(aligned), mask, self.cv2_to_tensor(debug)