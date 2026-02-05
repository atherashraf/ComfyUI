from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class FaceCropConfig:
    target_size: int = 1024
    expansion_factor: float = 1.6
    min_confidence: float = 0.35
    suffix_face: str = "_face_cropped"
    suffix_fallback: str = "_cropped"


class FaceCenteredCropper:
    def __init__(self, cfg: FaceCropConfig):
        self.cfg = cfg
        self.detector_type = "opencv"
        self.face_detection = None
        self.mp_face_detection = None

        # Try MediaPipe, fallback to OpenCV
        try:
            import mediapipe as mp  # type: ignore
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.cfg.min_confidence,
            )
            self.detector_type = "mediapipe"
            print("✓ Using MediaPipe for face detection")
        except Exception:
            self._init_opencv_detector()
            print("✓ Using OpenCV Haar Cascades for face detection")

        print(f"Initialized FaceCenteredCropper ({self.detector_type}) -> {cfg.target_size}x{cfg.target_size}")

    def _init_opencv_detector(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

    def _pil_to_cv(self, img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def detect_face(self, image_cv: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.detector_type == "mediapipe" and self.face_detection is not None:
            rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb)
            if not results.detections:
                return None

            best = max(results.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box
            h, w = image_cv.shape[:2]
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            bw = min(w - x, bw)
            bh = min(h - y, bh)

            return {"x": x, "y": y, "width": bw, "height": bh, "confidence": float(best.score[0])}

        # OpenCV fallback
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(min(w, h) * 0.1), int(min(w, h) * 0.1))
        )
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(min(w, h) * 0.1), int(min(w, h) * 0.1))
            )
        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, bw, bh = faces[0]

        face_area = bw * bh
        img_area = w * h
        confidence = min(face_area / max(img_area * 0.1, 1), 1.0)
        return {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh), "confidence": float(confidence)}

    def _square_crop_box(self, img_w: int, img_h: int, cx: int, cy: int, size: int) -> Tuple[int, int, int, int]:
        half = size // 2
        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > img_w:
            x1 -= (x2 - img_w)
            x2 = img_w
        if y2 > img_h:
            y1 -= (y2 - img_h)
            y2 = img_h

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            # fallback to center square
            min_dim = min(img_w, img_h)
            x1 = (img_w - min_dim) // 2
            y1 = (img_h - min_dim) // 2
            x2 = x1 + min_dim
            y2 = y1 + min_dim

        return int(x1), int(y1), int(x2), int(y2)

    def process_image(self, image_path: Path) -> Tuple[Image.Image, bool]:
        img = Image.open(image_path).convert("RGB")
        cv_img = self._pil_to_cv(img)
        h, w = cv_img.shape[:2]

        face = self.detect_face(cv_img)
        if face and face["confidence"] >= self.cfg.min_confidence:
            cx = face["x"] + face["width"] // 2
            cy = face["y"] + face["height"] // 2
            base_size = int(face["height"] * self.cfg.expansion_factor)
            min_crop = int(self.cfg.target_size * 0.65)
            crop_size = min(max(base_size, min_crop), min(w, h))
            x1, y1, x2, y2 = self._square_crop_box(w, h, cx, cy, crop_size)
            cropped = img.crop((x1, y1, x2, y2))
            is_face = True
        else:
            # smart center crop
            min_dim = min(w, h)
            crop = int(min_dim * 0.85)
            x1 = (w - crop) // 2
            y1 = (h - crop) // 2
            cropped = img.crop((x1, y1, x1 + crop, y1 + crop))
            is_face = False

        out = cropped.resize((self.cfg.target_size, self.cfg.target_size), Image.Resampling.LANCZOS)
        return out, is_face

    def process_folder(self, input_folder: Path, output_folder: Path, overwrite: bool = False) -> Dict[str, int]:
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        images.sort()

        stats = {"total": len(images), "saved": 0, "face": 0, "fallback": 0, "skipped": 0, "errors": 0}

        for p in tqdm(images, desc="Face-centered crop", unit="img"):
            try:
                out_name = p.stem + self.cfg.suffix_fallback + p.suffix.lower()
                out_path = output_folder / out_name

                if out_path.exists() and not overwrite:
                    stats["skipped"] += 1
                    continue

                img, is_face = self.process_image(p)
                suffix = self.cfg.suffix_face if is_face else self.cfg.suffix_fallback
                out_path = output_folder / (p.stem + suffix + p.suffix.lower())
                img.save(out_path, quality=95, optimize=True)

                stats["saved"] += 1
                stats["face"] += int(is_face)
                stats["fallback"] += int(not is_face)
            except Exception as e:
                stats["errors"] += 1
                print(f"\n✗ Error processing {p.name}: {e}")

        print(f"Done. {stats}")
        return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--expand", type=float, default=1.6)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = FaceCropConfig(target_size=args.size, expansion_factor=args.expand, min_confidence=args.conf)
    FaceCenteredCropper(cfg).process_folder(Path(args.input), Path(args.output), overwrite=args.overwrite)
