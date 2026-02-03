import cv2
from PIL import Image
from pathlib import Path
import numpy as np
import os


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

    fullbody_detected = 0
    upperbody_detected = 0
    manual_crop = 0

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")

        # Open image
        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size

        # Convert to OpenCV
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Resize for faster detection if image is very large
        scale = 1.0
        if max(orig_h, orig_w) > 2000:
            scale = 2000 / max(orig_h, orig_w)
            detect_img = cv2.resize(img_cv, None, fx=scale, fy=scale)
        else:
            detect_img = img_cv.copy()

        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)

        detection_made = False

        # Try full body detection first
        if fullbody_cascade:
            bodies = fullbody_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(50, 100)  # Minimum body size
            )

            if len(bodies) > 0:
                # Use the largest detection
                bodies = sorted(bodies, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = bodies[0]

                # Scale back to original size
                x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

                print(f"  ✓ Full body detected: {w}x{h}")
                fullbody_detected += 1
                detection_made = True

                # Calculate crop: expand around detected body
                # Add padding to ensure full body is included
                padding_w = int(w * 0.2)  # 20% padding on sides
                padding_h = int(h * 0.3)  # 30% padding top/bottom

                x1 = max(0, x - padding_w)
                y1 = max(0, y - padding_h)
                x2 = min(orig_w, x + w + padding_w)
                y2 = min(orig_h, y + h + padding_h)

        # Try upper body detection as fallback
        if not detection_made and upperbody_cascade:
            upper_bodies = upperbody_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(50, 50)
            )

            if len(upper_bodies) > 0:
                upper_bodies = sorted(upper_bodies, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = upper_bodies[0]

                # Scale back
                x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

                print(f"  ⚠ Upper body detected (assuming full body): {w}x{h}")
                upperbody_detected += 1
                detection_made = True

                # For upper body, assume it's about 40% of total height
                estimated_full_height = int(h / 0.4)

                # Position crop: upper body should be in top 40% of crop
                x1 = max(0, x - int(w * 0.3))
                y1 = max(0, y - int(h * 0.2))
                x2 = min(orig_w, x + w + int(w * 0.3))
                y2 = min(orig_h, y1 + estimated_full_height)

        # Manual/fallback crop
        if not detection_made:
            print(f"  ⚠ No body detected, using manual crop rules")
            manual_crop += 1

            # Rule-based cropping for common portrait/full-body compositions

            # Check image aspect ratio and size
            if orig_h > orig_w * 1.5:
                # Tall portrait - likely full body
                # Crop center with more height
                crop_height = min(orig_h, int(orig_w * 1.5))
                y1 = (orig_h - crop_height) // 2
                y2 = y1 + crop_height
                x1 = 0
                x2 = orig_w
            elif orig_w > orig_h * 1.5:
                # Wide landscape - might be full body horizontal
                crop_width = min(orig_w, int(orig_h * 1.5))
                x1 = (orig_w - crop_width) // 2
                x2 = x1 + crop_width
                y1 = 0
                y2 = orig_h
            else:
                # Square-ish - use center crop with some padding
                min_dim = min(orig_w, orig_h)
                crop_size = int(min_dim * 0.9)  # 90% of smaller dimension
                x1 = (orig_w - crop_size) // 2
                y1 = (orig_h - crop_size) // 2
                x2 = x1 + crop_size
                y2 = y1 + crop_size

        # Final crop coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        # Ensure we have a valid crop
        if x2 <= x1 or y2 <= y1:
            # Fallback to center square crop
            min_dim = min(orig_w, orig_h)
            x1 = (orig_w - min_dim) // 2
            y1 = (orig_h - min_dim) // 2
            x2 = x1 + min_dim
            y2 = y1 + min_dim

        # Apply crop
        cropped_pil = pil_img.crop((x1, y1, x2, y2))
        crop_w, crop_h = cropped_pil.size

        # Resize to target size (SDXL wants 1024x1024)
        # But first, pad if not square
        if crop_w != crop_h:
            # Create square canvas
            square_size = max(crop_w, crop_h)
            square_img = Image.new('RGB', (square_size, square_size), (0, 0, 0))

            # Paste cropped image centered
            paste_x = (square_size - crop_w) // 2
            paste_y = (square_size - crop_h) // 2
            square_img.paste(cropped_pil, (paste_x, paste_y))
            cropped_pil = square_img

        # Resize to final size
        final_img = cropped_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # Save
        suffix = ""
        if detection_made:
            if fullbody_detected > 0 and 'fullbody_detected' in locals():
                suffix = "_fullbody"
            else:
                suffix = "_upperbody"
        else:
            suffix = "_manual"

        output_file = output_path / suffix[1:] / f"{img_path.stem}{suffix}{img_path.suffix}"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_img.save(output_file, quality=95)
        print(f"  Saved: {output_file.name}")

    # Summary
    print(f"\n{'=' * 60}")
    print("FULL-BODY TRAINING CROP SUMMARY:")
    print(f"{'=' * 60}")
    print(f"Total images processed: {len(image_files)}")
    print(f"✓ Full body detected: {fullbody_detected}")
    print(f"⚠ Upper body detected: {upperbody_detected}")
    print(f"⏺ Manual/fallback crops: {manual_crop}")
    print(f"\nOutput folder: {output_path}")


# Install better detector first
# pip install ultralytics

def yolo_fullbody_crop(input_folder, output_folder, target_size=1024):
    """
    Use YOLO for better person detection.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install YOLO: pip install ultralytics")
        return

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load YOLO model (person detection is class 0)
    model = YOLO('yolov8n.pt')  # Nano model, small and fast

    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    for img_path in image_files:
        print(f"Processing: {img_path.name}")

        img = Image.open(img_path).convert('RGB')

        # Run YOLO detection
        results = model(img, verbose=False)

        person_boxes = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    person_boxes.append((x1, y1, x2, y2))

        if person_boxes:
            # Use the largest person detection
            person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            x1, y1, x2, y2 = person_boxes[0]

            # Add padding
            padding = 0.15  # 15% padding
            w, h = img.size
            pad_x = (x2 - x1) * padding
            pad_y = (y2 - y1) * padding

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            print(f"  ✓ Person detected: {int(x2 - x1)}x{int(y2 - y1)}")
        else:
            # No detection - use center crop
            print(f"  ⚠ No person detected, using center crop")
            w, h = img.size
            min_dim = min(w, h)
            x1 = (w - min_dim) // 2
            y1 = (h - min_dim) // 2
            x2 = x1 + min_dim
            y2 = y1 + min_dim

        # Crop and process
        cropped = img.crop((x1, y1, x2, y2))
        crop_w, crop_h = cropped.size

        # Make square
        if crop_w != crop_h:
            square_size = max(crop_w, crop_h)
            square_img = Image.new('RGB', (square_size, square_size), (0, 0, 0))
            paste_x = (square_size - crop_w) // 2
            paste_y = (square_size - crop_h) // 2
            square_img.paste(cropped, (paste_x, paste_y))
            cropped = square_img

        # Resize
        final = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

        output_file = output_path / f"{img_path.stem}_cropped{img_path.suffix}"
        final.save(output_file, quality=95)

    print(f"\nDone! Images saved to: {output_folder}")

if __name__ == "__main__":
    output_folder = Path(__file__).parent / "output/judi"
    output_folder.mkdir(parents=True, exist_ok=True)
    input_folder = Path("D:\Documents\general\downloads\pics\judi")
    full_body_crop_for_sdxl(input_folder, output_folder, target_size=1024)