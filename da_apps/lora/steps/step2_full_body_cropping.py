from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm


@dataclass
class FullBodyCropConfig:
    target_size: int = 1024
    padding: float = 0.15
    yolo_weights: str = "yolov8n.pt"
    min_conf: float = 0.35


def _square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def yolo_fullbody_crop(input_folder: Path, output_folder: Path, target_size: int = 1024,
                       padding: float = 0.15, overwrite: bool = False,
                       yolo_weights: str = "yolov8n.pt", min_conf: float = 0.35) -> Dict[str, int]:
    """
    Detect largest person and crop with padding, then resize to SDXL 1024.
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

    model = YOLO(yolo_weights)

    stats = {"total": len(images), "saved": 0, "detected": 0, "fallback": 0, "skipped": 0, "errors": 0}

    for img_path in tqdm(images, desc="YOLO full-body crop", unit="img"):
        try:
            out_file = output_path / f"{img_path.stem}_yolo_cropped{img_path.suffix.lower()}"
            if out_file.exists() and not overwrite:
                stats["skipped"] += 1
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            results = model(img, verbose=False)

            person_boxes: List[Tuple[float, float, float, float, float]] = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == 0 and conf >= min_conf:  # class 0 = person
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        person_boxes.append((x1, y1, x2, y2, conf))

            if person_boxes:
                # largest area
                person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
                x1, y1, x2, y2, _ = person_boxes[0]

                pad_x = (x2 - x1) * padding
                pad_y = (y2 - y1) * padding

                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                stats["detected"] += 1
            else:
                # fallback center square
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

    print(f"Done. {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--padding", type=float, default=0.15)
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    yolo_fullbody_crop(
        Path(args.input), Path(args.output),
        target_size=args.size,
        padding=args.padding,
        overwrite=args.overwrite,
        yolo_weights=args.weights,
        min_conf=args.min_conf,
    )
