import os
import sys
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import folder_paths
from .da_person_mask_generator_comfyui import DAPersonMaskGenerator


def get_da_face_landmarker_model_path() -> str:
    model_folder_name = "mediapipe"
    model_name = "face_landmarker.task"

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

    if not os.path.exists(model_file_path):
        import urllib.request

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            f"face_landmarker/float16/latest/{model_name}"
        )
        print(f"Downloading '{model_name}' model")
        os.makedirs(model_folder_path, exist_ok=True)
        urllib.request.urlretrieve(model_url, model_file_path)

    return model_file_path


class DAPersonFaceLandmarkMaskGenerator:
    # classic mesh indices, still usable with FaceLandmarker output
    FACEMESH_FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]

    FACEMESH_LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    FACEMESH_RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    FACEMESH_LEFT_EYE = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398,
    ]
    FACEMESH_RIGHT_EYE = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
    ]

    FACEMESH_LEFT_PUPIL = [474, 475, 476, 477]
    FACEMESH_RIGHT_PUPIL = [469, 470, 471, 472]

    FACEMESH_LIPS = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        415, 310, 311, 312, 13, 82, 81, 42, 183, 78,
    ]

    def __init__(self):
        get_da_face_landmarker_model_path()

    @classmethod
    def INPUT_TYPES(cls):
        false_widget = (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        )
        true_widget = (
            "BOOLEAN",
            {"default": True, "label_on": "enabled", "label_off": "disabled"},
        )

        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "face": false_widget,
                "left_eyebrow": false_widget,
                "right_eyebrow": false_widget,
                "left_eye": true_widget,
                "right_eye": true_widget,
                "left_pupil": false_widget,
                "right_pupil": false_widget,
                "lips": true_widget,
                "number_of_faces": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "refine_mask": true_widget,
            },
        }

    CATEGORY = "DA Masking"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "generate_mask"

    @staticmethod
    @lru_cache(maxsize=8)
    def _create_landmarker(model_path: str, number_of_faces: int, confidence: float):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=number_of_faces,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        arr = 255.0 * image_tensor.cpu().numpy()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _mask_pil_to_tensor(mask_image: Image.Image) -> torch.Tensor:
        gray = mask_image.convert("L")
        arr = np.array(gray).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    @staticmethod
    def _fill_region(mask: np.ndarray, coords: list[tuple[int, int]]):
        if len(coords) >= 3:
            cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], 255)

    @staticmethod
    def _landmarks_to_coords(landmarks, img_width: int, img_height: int):
        coords = []
        for point in landmarks:
            x = int(np.clip(point.x, 0.0, 1.0) * (img_width - 1))
            y = int(np.clip(point.y, 0.0, 1.0) * (img_height - 1))
            coords.append((x, y))
        return coords

    def _draw_face_parts(
        self,
        mask: np.ndarray,
        mesh_coords: list[tuple[int, int]],
        face: bool,
        left_eyebrow: bool,
        right_eyebrow: bool,
        left_eye: bool,
        right_eye: bool,
        left_pupil: bool,
        right_pupil: bool,
        lips: bool,
    ):
        if face:
            face_coords = [mesh_coords[p] for p in self.FACEMESH_FACE_OVAL if p < len(mesh_coords)]
            self._fill_region(mask, face_coords)
            return

        if left_eyebrow:
            coords = [mesh_coords[p] for p in self.FACEMESH_LEFT_EYEBROW if p < len(mesh_coords)]
            self._fill_region(mask, coords)

        if right_eyebrow:
            coords = [mesh_coords[p] for p in self.FACEMESH_RIGHT_EYEBROW if p < len(mesh_coords)]
            self._fill_region(mask, coords)

        if left_eye:
            coords = [mesh_coords[p] for p in self.FACEMESH_LEFT_EYE if p < len(mesh_coords)]
            self._fill_region(mask, coords)

        if right_eye:
            coords = [mesh_coords[p] for p in self.FACEMESH_RIGHT_EYE if p < len(mesh_coords)]
            self._fill_region(mask, coords)

        if left_pupil:
            coords = [mesh_coords[p] for p in self.FACEMESH_LEFT_PUPIL if p < len(mesh_coords)]
            self._fill_region(mask, coords)

        if right_pupil:
            coords = [mesh_coords[p] for p in self.FACEMESH_RIGHT_PUPIL if p < len(mesh_coords)]
            self._fill_region(mask, coords)

        if lips:
            coords = [mesh_coords[p] for p in self.FACEMESH_LIPS if p < len(mesh_coords)]
            self._fill_region(mask, coords)

    def generate_mask(
        self,
        images,
        face: bool = False,
        left_eyebrow: bool = False,
        right_eyebrow: bool = False,
        left_eye: bool = True,
        right_eye: bool = True,
        left_pupil: bool = False,
        right_pupil: bool = False,
        lips: bool = True,
        number_of_faces: int = 1,
        confidence: float = 0.40,
        refine_mask: bool = True,
    ):
        a_person_mask_generator = None
        face_masks = None

        if refine_mask:
            a_person_mask_generator = DAPersonMaskGenerator()
            face_masks = a_person_mask_generator.get_mask_images(
                images=images,
                face_mask=True,
                background_mask=False,
                hair_mask=False,
                body_mask=False,
                clothes_mask=False,
                confidence=0.4,
                refine_mask=True,
            )

        model_path = get_da_face_landmarker_model_path()
        landmarker = self._create_landmarker(model_path, number_of_faces, confidence)

        res_masks = []

        for index, image in enumerate(images):
            image_pil = self._tensor_to_pil(image)

            cropped_image_pil = image_pil
            do_uncrop = False
            bbox = None

            if refine_mask and face_masks is not None:
                face_mask_image = face_masks[index]
                bbox = a_person_mask_generator.get_bbox_for_mask(mask_image=face_mask_image)
                if bbox is not None:
                    cropped_image_pil = image_pil.crop(bbox)
                    do_uncrop = True

            img_width, img_height = cropped_image_pil.size
            mask = np.zeros((img_height, img_width), dtype=np.uint8)

            if any([face, left_eyebrow, right_eyebrow, left_eye, right_eye, left_pupil, right_pupil, lips]):
                rgb = np.asarray(cropped_image_pil.convert("RGB"))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)

                if results.face_landmarks:
                    for face_landmarks in results.face_landmarks[:number_of_faces]:
                        mesh_coords = self._landmarks_to_coords(face_landmarks, img_width, img_height)
                        self._draw_face_parts(
                            mask,
                            mesh_coords,
                            face,
                            left_eyebrow,
                            right_eyebrow,
                            left_eye,
                            right_eye,
                            left_pupil,
                            right_pupil,
                            lips,
                        )

            mask_image = Image.fromarray(mask, mode="L")

            if do_uncrop and bbox is not None:
                uncropped_mask_image = Image.new("L", image_pil.size, 0)
                uncropped_mask_image.paste(mask_image, bbox)
                mask_image = uncropped_mask_image

            res_masks.append(self._mask_pil_to_tensor(mask_image))

        return (torch.cat(res_masks, dim=0),)