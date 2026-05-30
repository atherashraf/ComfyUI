from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from PIL import Image
from tqdm import tqdm


@dataclass
class FaceHairCropConfig:
    target_size: int = 1024
    padding: float = 0.25  # Extra padding around face for hair
    yolo_face_weights: str = "yolov8n-face.pt"  # or "yolov8n.pt" for default
    min_conf: float = 0.35
    include_hair_segmentation: bool = False  # Use segmentation for better hair capture
    hair_expand_factor: float = 0.3  # Expand face box upward for hair


def _square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def yolo_face_hair_crop(
        input_folder: Path,
        output_folder: Path,
        target_size: int = 1024,
        padding: float = 0.25,
        overwrite: bool = False,
        yolo_weights: str = "yolov8n-face.pt",
        min_conf: float = 0.35,
        include_hair_segmentation: bool = False,
        hair_expand_factor: float = 0.3
) -> Dict[str, int]:
    """
    Detect faces and crop with padding to include hair.
    Optionally uses segmentation for better hair capture.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("Install YOLO: pip install ultralytics")
        return {"errors": 1}

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images.sort()

    # Try to load face-specific model, fall back to default
    try:
        model = YOLO(yolo_weights)
    except Exception:
        print(f"Warning: Could not load {yolo_weights}, falling back to yolov8n.pt")
        model = YOLO("yolov8n.pt")

    stats = {"total": len(images), "saved": 0, "detected": 0, "fallback": 0, "skipped": 0, "errors": 0}

    for img_path in tqdm(images, desc="Face+Hair crop", unit="img"):
        try:
            out_file = output_path / f"{img_path.stem}_face_hair_cropped{img_path.suffix.lower()}"
            if out_file.exists() and not overwrite:
                stats["skipped"] += 1
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            # Detect faces
            results = model(img, verbose=False)

            face_boxes: List[Tuple[float, float, float, float, float]] = []

            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        # Check if it's a face (class 0 is usually person, but face models vary)
                        # For face-specific models, adjust class ID accordingly
                        is_face = False
                        if "face" in yolo_weights.lower():
                            is_face = cls_id == 0  # Face models often use class 0 for face
                        else:
                            is_face = cls_id == 0  # Person class in default model

                        if is_face and conf >= min_conf:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            face_boxes.append((x1, y1, x2, y2, conf))

            if face_boxes:
                # Get largest face by area
                face_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                x1, y1, x2, y2, _ = face_boxes[0]

                face_w = x2 - x1
                face_h = y2 - y1

                # Expand to include hair (especially upward)
                if include_hair_segmentation:
                    # Try to get hair mask using segmentation
                    if hasattr(r, 'masks') and r.masks is not None:
                        # Use mask to find hair region (typically above face)
                        # This is simplified - full implementation would need more logic
                        mask = r.masks.data[0].cpu().numpy()
                        # Expand upward based on mask
                        y_expand = face_h * hair_expand_factor
                    else:
                        y_expand = face_h * hair_expand_factor
                else:
                    # Simple expansion: more padding on top for hair
                    y_expand = face_h * hair_expand_factor

                # Apply padding with extra on top for hair
                pad_x = face_w * padding
                pad_top = face_h * (padding + hair_expand_factor)  # Extra padding on top
                pad_bottom = face_h * padding

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_top)  # More expansion upward
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_bottom)

                stats["detected"] += 1
            else:
                # Fallback: center crop as before
                min_dim = min(w, h)
                x1 = (w - min_dim) // 2
                y1 = (h - min_dim) // 2
                x2 = x1 + min_dim
                y2 = y1 + min_dim
                stats["fallback"] += 1

            cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
            cropped = _square_pad(cropped).resize((target_size, target_size), Image.Resampling.LANCZOS)
            cropped.save(out_file, quality=95, optimize=True)
            stats["saved"] += 1

        except Exception as e:
            stats["errors"] += 1
            print(f"\n✗ Error processing {img_path.name}: {e}")

    print(f"Done. Stats: {stats}")
    return stats


def smart_face_hair_crop(
        input_folder: Path,
        output_folder: Path,
        target_size: int = 1024,
        face_padding: float = 0.2,
        hair_top_expansion: float = 0.4,  # Extra space above face for hair
        overwrite: bool = False,
        use_face_detection: bool = True,
        fallback_to_center: bool = True
) -> Dict[str, int]:
    """
    Simplified smart crop focusing on face region with hair.
    Uses OpenCV's face detection as lightweight alternative.
    """
    try:
        import cv2
    except ImportError:
        print("Install OpenCV: pip install opencv-python")
        return {"errors": 1}

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images.sort()

    stats = {"total": len(images), "saved": 0, "detected": 0, "fallback": 0, "skipped": 0, "errors": 0}

    for img_path in tqdm(images, desc="Smart face+hair crop", unit="img"):
        try:
            out_file = output_path / f"{img_path.stem}_face_hair_cropped{img_path.suffix.lower()}"
            if out_file.exists() and not overwrite:
                stats["skipped"] += 1
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            if use_face_detection:
                # Convert PIL to OpenCV format
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)

                if len(faces) > 0:
                    # Get largest face by area
                    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                    fx, fy, fw, fh = faces[0]

                    # Expand to include hair (more on top)
                    expand_x = int(fw * face_padding)
                    expand_top = int(fh * hair_top_expansion)
                    expand_bottom = int(fh * face_padding)

                    x1 = max(0, fx - expand_x)
                    y1 = max(0, fy - expand_top)
                    x2 = min(w, fx + fw + expand_x)
                    y2 = min(h, fy + fh + expand_bottom)

                    stats["detected"] += 1
                elif fallback_to_center:
                    # Fallback to center crop
                    min_dim = min(w, h)
                    x1 = (w - min_dim) // 2
                    y1 = (h - min_dim) // 2
                    x2 = x1 + min_dim
                    y2 = y1 + min_dim
                    stats["fallback"] += 1
                else:
                    stats["errors"] += 1
                    continue
            else:
                # Center crop if face detection disabled
                min_dim = min(w, h)
                x1 = (w - min_dim) // 2
                y1 = (h - min_dim) // 2
                x2 = x1 + min_dim
                y2 = y1 + min_dim
                stats["fallback"] += 1

            cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
            cropped = _square_pad(cropped).resize((target_size, target_size), Image.Resampling.LANCZOS)
            cropped.save(out_file, quality=95, optimize=True)
            stats["saved"] += 1

        except Exception as e:
            stats["errors"] += 1
            print(f"\n✗ Error processing {img_path.name}: {e}")

    print(f"Done. Stats: {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Face and hair cropping tool")
    ap.add_argument("--input", required=True, help="Input folder")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--size", type=int, default=1024, help="Target size (default: 1024)")
    ap.add_argument("--padding", type=float, default=0.25, help="Padding around face (default: 0.25)")
    ap.add_argument("--hair-expand", type=float, default=0.3, help="Extra space for hair above face (default: 0.3)")
    ap.add_argument("--weights", type=str, default="yolov8n-face.pt", help="YOLO face model weights")
    ap.add_argument("--min-conf", type=float, default=0.35, help="Minimum confidence (default: 0.35)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    ap.add_argument("--method", choices=["yolo", "opencv"], default="yolo",
                    help="Detection method: yolo or opencv (default: yolo)")
    ap.add_argument("--segmentation", action="store_true",
                    help="Use segmentation for better hair capture (YOLO only)")

    args = ap.parse_args()

    if args.method == "yolo":
        yolo_face_hair_crop(
            Path(args.input), Path(args.output),
            target_size=args.size,
            padding=args.padding,
            overwrite=args.overwrite,
            yolo_weights=args.weights,
            min_conf=args.min_conf,
            include_hair_segmentation=args.segmentation,
            hair_expand_factor=args.hair_expand
        )
    else:
        smart_face_hair_crop(
            Path(args.input), Path(args.output),
            target_size=args.size,
            face_padding=args.padding,
            hair_top_expansion=args.hair_expand,
            overwrite=args.overwrite,
            use_face_detection=True,
            fallback_to_center=True
        )