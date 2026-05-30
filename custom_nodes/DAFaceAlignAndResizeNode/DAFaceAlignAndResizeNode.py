import torch
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

class DAFaceAlignAndResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "padding_color": (["black", "white", "edge_extend"], {"default": "black"}),
                "blend_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blur_factor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "aligned_source")
    FUNCTION = "process"
    CATEGORY = "DA Nodes/Face"

    def __init__(self):
        # Auto-download model if missing
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        if not os.path.exists(model_path):
            print(f"[DA] Downloading face_landmarker.task...")
            urllib.request.urlretrieve(model_url, model_path)

        # Configure Detector with Iris Refinement
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def tensor_to_cv2(self, tensor):
        if tensor.dim() == 4: tensor = tensor[0]
        img = np.clip(tensor.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def cv2_to_tensor(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)

    def get_landmarks(self, cv2_img):
        if cv2_img is None: return None

        # 1. Convert to RGB and ensure it's 3-channel
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        # 2. BRIGHTNESS/CONTRAST BOOST (Only for detection)
        # This helps if the target face is in shadow
        detect_img = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=10)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_img)
        res = self.detector.detect(mp_image)

        if not res.face_landmarks:
            # 3. FINAL ATTEMPT: Try resizing if the image is massive
            # Large images (4K+) can actually make small faces harder to find
            h, w = img_rgb.shape[:2]
            if w > 1280:
                small_img = cv2.resize(detect_img, (1280, int(h * (1280 / w))))
                mp_image_small = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_img)
                res = self.detector.detect(mp_image_small)
                if res.face_landmarks:
                    # Map coordinates back to original size
                    scale_x, scale_y = w / 1280, h / int(h * (1280 / w))
                    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in res.face_landmarks[0]])

            print("[DA Debug] Face mesh failed to find landmarks in image.")
            return None

        h, w = cv2_img.shape[:2]
        return np.array([(int(lm.x * w), int(lm.y * h)) for lm in res.face_landmarks[0]])

    def process(self, source_image, target_image, padding_color, blend_alpha, blur_factor):
        src_img = self.tensor_to_cv2(source_image)
        tgt_img = self.tensor_to_cv2(target_image)
        h, w = tgt_img.shape[:2]

        src_pts = self.get_landmarks(src_img)
        tgt_pts = self.get_landmarks(tgt_img)

        if src_pts is None or tgt_pts is None:
            return self.cv2_to_tensor(tgt_img), torch.ones((h, w)), self.cv2_to_tensor(src_img)

        # 1. Similarity Transform (Prevents Aspect Ratio Issues)
        # Using Iris Centers (468, 473), Nose (1), and Mouth (61, 291)
        # These are the most stable points across different facial expressions
        stable_idx = [468, 473, 1, 61, 291]
        src_subset = np.float32([src_pts[i] for i in stable_idx])
        tgt_subset = np.float32([tgt_pts[i] for i in stable_idx])

        # estimateAffinePartial2D restricts math to uniform scale, rotation, and translation
        matrix, _ = cv2.estimateAffinePartial2D(src_subset, tgt_subset)
        
        b_mode = cv2.BORDER_REPLICATE if padding_color == "edge_extend" else cv2.BORDER_CONSTANT
        b_val = (255, 255, 255) if padding_color == "white" else (0, 0, 0)
        
        aligned_src = cv2.warpAffine(src_img, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=b_mode, borderValue=b_val)

        # 2. Trace Silhouette for Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        # 36 indices for the outer edge of the face
        silhouette = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        poly_pts = np.array([tgt_pts[i] for i in silhouette], dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], 255)

        # 3. Poisson Blending (Handles lighting and orientation mismatches)
        x, y, mw, mh = cv2.boundingRect(mask)
        center = (x + mw // 2, y + mh // 2)
        
        try:
            # NORMAL_CLONE adapts the source pixels to match target lighting
            blended = cv2.seamlessClone(aligned_src, tgt_img, mask, center, cv2.NORMAL_CLONE)
            
            if blend_alpha < 1.0:
                blended = cv2.addWeighted(blended, blend_alpha, tgt_img, 1.0 - blend_alpha, 0)
        except Exception:
            # Fallback for masks touching image boundaries
            alpha_mask = mask.astype(np.float32) / 255.0
            k_size = int(mw * blur_factor) | 1
            alpha_mask = cv2.GaussianBlur(alpha_mask, (max(3, k_size), max(3, k_size)), 0)
            alpha_mask = np.expand_dims(alpha_mask * blend_alpha, -1)
            blended = (aligned_src.astype(np.float32) * alpha_mask + tgt_img.astype(np.float32) * (1.0 - alpha_mask)).astype(np.uint8)

        return self.cv2_to_tensor(blended), torch.from_numpy(mask.astype(np.float32)/255.0), self.cv2_to_tensor(aligned_src)

