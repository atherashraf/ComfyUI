from .da_person_mask_generator_comfyui import DAPersonMaskGenerator
from .da_person_face_landmark_mask_generator_comfyui import DAPersonFaceLandmarkMaskGenerator
from .DAMaskCropToTargetSizeNode import DAMaskCropToTargetSizeNode

NODE_CLASS_MAPPINGS = {
    "DAPersonMaskGenerator": DAPersonMaskGenerator,
    "DAPersonFaceLandmarkMaskGenerator": DAPersonFaceLandmarkMaskGenerator,
    "DAMaskCropToTargetSizeNode": DAMaskCropToTargetSizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DAPersonMaskGenerator": "DA Person Mask Generator",
    "DAPersonFaceLandmarkMaskGenerator": "DA Person Face Landmark Mask Generator",
    "DAMaskCropToTargetSizeNode": "DA Mask Crop To Target Size",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]