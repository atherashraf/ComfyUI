from .da_person_mask_generator_comfyui import DAPersonMaskGenerator
from .da_person_face_landmark_mask_generator_comfyui import DAPersonFaceLandmarkMaskGenerator

NODE_CLASS_MAPPINGS = {
    "DAPersonMaskGenerator": DAPersonMaskGenerator,
    "DAPersonFaceLandmarkMaskGenerator": DAPersonFaceLandmarkMaskGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DAPersonMaskGenerator": "DA Person Mask Generator",
    "DAPersonFaceLandmarkMaskGenerator": "DA Person Face Landmark Mask Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]