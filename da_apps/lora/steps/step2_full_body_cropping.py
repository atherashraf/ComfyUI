import cv2
# from PIL import Image
from pathlib import Path
# import numpy as np
# import os


def full_body_crop_for_sdxl(input_folder, output_folder, target_size=1024):
    """
    Smart full-body cropping for SDXL training.
    Detects person and crops to include entire body.
    """

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try to load OpenCV's full body detector
    # Note: OpenCV's full body detection is basic. For better results,
    # consider YOLO or MediaPipe Pose, but this is a good start.
    fullbody_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_fullbody.xml'
    )

    # Also try upper body as fallback
    upperbody_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_upperbody.xml'
    )

    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))

    print(f"Found {len(image_files)} images for full-body training")
