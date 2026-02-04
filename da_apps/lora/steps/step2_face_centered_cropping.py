import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

# Try to import MediaPipe, with fallback to OpenCV
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("MediaPipe not found. Installing...")
    try:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])
        import mediapipe as mp

        MEDIAPIPE_AVAILABLE = True
        print("MediaPipe installed successfully.")
    except:
        MEDIAPIPE_AVAILABLE = False
        print("Failed to install MediaPipe. Using OpenCV fallback.")


class FaceCenteredCropper:
    def __init__(self, target_size=1024, expansion_factor=1.5, min_confidence=0.5):
        """
        Initialize the face-centered cropper for SDXL (1024x1024).

        Args:
            target_size: Output image size (1024 for SDXL)
            expansion_factor: How much to expand around detected face (1.5 = 150%)
            min_confidence: Minimum confidence for face detection
        """
        self.target_size = target_size
        self.expansion_factor = expansion_factor
        self.min_confidence = min_confidence

        # Initialize face detector (prefer MediaPipe, fallback to OpenCV)
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=min_confidence
                )
                self.detector_type = "mediapipe"
                print("✓ Using MediaPipe for face detection")
            except Exception as e:
                print(f"✗ MediaPipe initialization failed: {e}")
                self._init_opencv_detector()
        else:
            self._init_opencv_detector()

        print(f"Initialized FaceCenteredCropper for SDXL ({target_size}x{target_size})")

    def _init_opencv_detector(self):
        """Initialize OpenCV face detector as fallback."""
        # Load multiple cascade classifiers for better detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        self.detector_type = "opencv"
        print("✓ Using OpenCV Haar Cascades for face detection")

    def detect_face_mediapipe(self, image_cv):
        """Detect face using MediaPipe."""
        results = self.face_detection.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return None

        # Get the face with highest confidence
        best_detection = max(results.detections, key=lambda d: d.score[0])
        bbox = best_detection.location_data.relative_bounding_box

        h, w = image_cv.shape[:2]

        # Convert relative coordinates to absolute
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        width = min(w - x, width)
        height = min(h - y, height)

        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'confidence': best_detection.score[0]
        }

    def detect_face_opencv(self, image_cv):
        """Detect face using OpenCV."""
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Try frontal face detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(min(w, h) * 0.1), int(min(w, h) * 0.1))
        )

        # Try profile face detection if no frontal faces found
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(min(w, h) * 0.1), int(min(w, h) * 0.1))
            )

        if len(faces) == 0:
            return None

        # Use the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, width, height = faces[0]

        # Calculate confidence based on face size relative to image
        face_area = width * height
        image_area = w * h
        confidence = min(face_area / (image_area * 0.1), 1.0)  # Max 1.0

        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'confidence': confidence
        }

    def detect_face(self, image):
        """Detect the most prominent face in the image."""
        # Convert PIL to OpenCV format
        if isinstance(image, Image.Image):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image.copy()

        # Use appropriate detector
        if self.detector_type == "mediapipe":
            face_box = self.detect_face_mediapipe(image_cv)
        else:
            face_box = self.detect_face_opencv(image_cv)

        return face_box

    def calculate_crop_box(self, image_size, face_box):
        """Calculate the optimal crop box centered on the face."""
        img_h, img_w = image_size

        # Get face center
        face_center_x = face_box['x'] + face_box['width'] // 2
        face_center_y = face_box['y'] + face_box['height'] // 2

        # Calculate crop size based on face size
        # For SDXL and upper body, use larger expansion
        base_size = face_box['height'] * self.expansion_factor

        # Ensure minimum crop size
        min_crop = self.target_size * 0.6  # At least 60% of target size
        crop_size = int(max(base_size, min_crop))

        # Cap at image dimensions
        crop_size = min(crop_size, min(img_w, img_h))

        # Calculate crop boundaries
        half_crop = crop_size // 2

        # Start with face-centered crop
        x1 = face_center_x - half_crop
        y1 = face_center_y - half_crop
        x2 = face_center_x + half_crop
        y2 = face_center_y + half_crop

        # Adjust if crop goes out of image bounds
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

        # Final adjustments to ensure we're within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        # Ensure we have a valid crop
        if x2 <= x1 or y2 <= y1:
            # Fallback: center crop of the image
            print(f"  Warning: Invalid crop calculated, using center crop")
            min_dim = min(img_w, img_h)
            x1 = (img_w - min_dim) // 2
            y1 = (img_h - min_dim) // 2
            x2 = x1 + min_dim
            y2 = y1 + min_dim

        # Ensure square crop
        crop_width = x2 - x1
        crop_height = y2 - y1

        if crop_width != crop_height:
            # Make it square based on smaller dimension
            if crop_width > crop_height:
                diff = crop_width - crop_height
                y1 = max(0, y1 - diff // 2)
                y2 = min(img_h, y2 + diff // 2)
                if y2 - y1 < crop_width:
                    # Still not square, adjust width instead
                    x2 = x1 + (y2 - y1)
            else:
                diff = crop_height - crop_width
                x1 = max(0, x1 - diff // 2)
                x2 = min(img_w, x2 + diff // 2)
                if x2 - x1 < crop_height:
                    # Still not square, adjust height instead
                    y2 = y1 + (x2 - x1)

        return int(x1), int(y1), int(x2), int(y2)

    def smart_center_crop(self, image_cv):
        """Fallback: smart center crop when no face is detected."""
        h, w = image_cv.shape[:2]

        # Crop center 80% of the smaller dimension
        min_dim = min(h, w)
        crop_size = int(min_dim * 0.8)

        x1 = (w - crop_size) // 2
        y1 = (h - crop_size) // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Ensure square
        return x1, y1, x2, y2

    def process_image(self, image_path, output_path=None, verbose=True):
        """
        Process a single image: detect face, crop, and resize.

        Returns:
            (PIL Image, face_box or None)
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            try:
                pil_image = Image.open(image_path).convert('RGB')
                img_name = Path(image_path).name
            except Exception as e:
                if verbose:
                    print(f"✗ Error loading {image_path}: {e}")
                return None, None
        else:
            pil_image = image_path
            img_name = "image"

        # Convert to OpenCV for processing
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h, w = image_cv.shape[:2]

        if verbose:
            print(f"Processing: {img_name} ({w}x{h})")

        # Detect face
        face_box = self.detect_face(image_cv)

        if face_box and face_box['confidence'] > self.min_confidence:
            if verbose:
                print(f"  ✓ Face detected (confidence: {face_box['confidence']:.2f}, "
                      f"size: {face_box['width']}x{face_box['height']})")

            # Calculate crop box
            x1, y1, x2, y2 = self.calculate_crop_box((h, w), face_box)

            # Apply crop
            cropped_cv = image_cv[y1:y2, x1:x2]

        else:
            if verbose:
                if face_box:
                    print(f"  ⚠ Low confidence face ({face_box['confidence']:.2f}), using smart crop")
                else:
                    print(f"  ⚠ No face detected, using smart center crop")

            # Smart center crop
            x1, y1, x2, y2 = self.smart_center_crop(image_cv)
            cropped_cv = image_cv[y1:y2, x1:x2]
            face_box = None

        # Convert back to PIL and resize
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

        # Resize to target size (1024x1024 for SDXL)
        resized_pil = cropped_pil.resize(
            (self.target_size, self.target_size),
            Image.Resampling.LANCZOS
        )

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            resized_pil.save(output_path, quality=95, optimize=True)
            if verbose:
                print(f"  Saved to: {output_path}")

        return resized_pil, face_box

    def process_folder(self, input_folder, output_folder,
                       extensions=('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG')):
        """
        Process all images in a folder.

        Args:
            input_folder: Path to folder with input images
            output_folder: Path to save processed images
            extensions: Tuple of image extensions to process
        """
        # FIX: Use raw string for Windows paths
        input_path = Path(str(input_folder).replace('\\', '/'))
        output_path = Path(str(output_folder).replace('\\', '/'))
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Input folder: {input_path}")
        print(f"Output folder: {output_path}")

        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

        # Also check with pattern
        for pattern in ['*.[jJ][pP][gG]', '*.[jJ][pP][eE][gG]', '*.[pP][nN][gG]', '*.[wW][eE][bB][pP]']:
            image_files.extend(list(input_path.glob(pattern)))

        # Remove duplicates
        image_files = list(set(image_files))

        if not image_files:
            print(f"No images found in {input_folder}")
            # Try to list what's actually there
            print("Contents of folder:")
            for item in input_path.iterdir():
                print(f"  {item.name}")
            return

        print(f"Found {len(image_files)} images to process")

        # Process each image
        success_count = 0
        face_detected_count = 0
        fallback_count = 0
        error_count = 0

        for img_path in tqdm(image_files, desc="Processing images", unit="img"):
            try:
                # Create output filename
                output_file = output_path / f"{img_path.stem}_cropped{img_path.suffix.lower()}"

                # Skip if already processed and newer than source
                if output_file.exists():
                    source_time = os.path.getmtime(img_path)
                    dest_time = os.path.getmtime(output_file)
                    if dest_time >= source_time:
                        # Check if this had face detection
                        if "_face_" in output_file.stem:
                            face_detected_count += 1
                        success_count += 1
                        continue

                # Process image
                _, face_box = self.process_image(
                    img_path,
                    output_file,
                    verbose=False  # Less verbose in batch mode
                )

                if face_box:
                    face_detected_count += 1
                    # Rename to indicate face was detected
                    new_name = output_path / f"{img_path.stem}_face_cropped{img_path.suffix.lower()}"
                    output_file.rename(new_name)
                else:
                    fallback_count += 1

                success_count += 1

            except Exception as e:
                error_count += 1
                print(f"\n✗ Error processing {img_path.name}: {str(e)}")

        # Print summary
        print(f"\n{'=' * 60}")
        print("PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total images processed: {success_count}")
        print(f"  ✓ With face detection: {face_detected_count}")
        print(f"  ⚠ Fallback crops: {fallback_count}")
        print(f"  ✗ Errors: {error_count}")
        print(f"Output folder: {output_path}")

        if face_detected_count == 0 and fallback_count > 0:
            print(f"\n⚠ WARNING: No faces detected in any images!")
            print("  This may indicate:")
            print("  1. Poor quality images")
            print("  2. Faces at unusual angles")
            print("  3. Issues with face detection")
            print("  4. Try lowering --confidence threshold")

        return {
            'total': len(image_files),
            'success': success_count,
            'face_detected': face_detected_count,
            'fallback': fallback_count,
            'errors': error_count
        }


# def main():
#     parser = argparse.ArgumentParser(description='Crop and resize images for SDXL LoRA training')
#     parser.add_argument('--input', type=str, required=True,
#                         help='Input folder containing images (use forward slashes or double backslashes)')
#     parser.add_argument('--output', type=str, required=True,
#                         help='Output folder for processed images')
#     parser.add_argument('--size', type=int, default=1024,
#                         help='Target image size (default: 1024 for SDXL)')
#     parser.add_argument('--expansion', type=float, default=1.8,
#                         help='Expansion factor around face (default: 1.8 for upper body)')
#     parser.add_argument('--confidence', type=float, default=0.3,
#                         help='Minimum face detection confidence (default: 0.3)')
#     parser.add_argument('--preview', action='store_true',
#                         help='Show preview of first few processed images')
#
#     args = parser.parse_args()
#
#     # Fix Windows paths
#     input_path = args.input.replace('\\', '/')
#     output_path = args.output.replace('\\', '/')
#
#     print(f"Face-Centered Cropping Tool for SDXL")
#     print(f"Target size: {args.size}x{args.size}")
#     print(f"Face expansion: {args.expansion}x")
#     print(f"Min confidence: {args.confidence}")
#     print(f"Input: {input_path}")
#     print(f"Output: {output_path}")
#     print(f"{'=' * 60}")
#
#     # Initialize cropper
#     cropper = FaceCenteredCropper(
#         target_size=args.size,
#         expansion_factor=args.expansion,
#         min_confidence=args.confidence
#     )
#
#     # Process folder
#     results = cropper.process_folder(input_path, output_path)
#
#     # Preview if requested
#     if args.preview and results['success'] > 0:
#         preview_images(output_path, max_images=9)


def preview_images(folder_path, max_images=9):
    """Show preview of processed images."""
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path

        folder = Path(folder_path)
        images = list(folder.glob("*_cropped.*"))
        images = images[:max_images]

        if not images:
            print("No images to preview")
            return

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        for i, img_path in enumerate(images):
            if i >= len(axes):
                break
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(img_path.name[:20], fontsize=9)
            axes[i].axis('off')

        # Hide unused axes
        for i in range(len(images), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Preview of {len(images)} processed images", fontsize=14)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Could not show preview: {e}")


# Simple usage without command line arguments
# def simple_process():
#     """Simple function to run without command line arguments."""
#     # FIX: Use raw strings or forward slashes for Windows paths
#     input_folder = r"D:\Documents\general\downloads\pics\judi"  # Raw string
#     # OR: input_folder = "D:/Documents/general/downloads/pics/judi"  # Forward slashes
#
#     output_folder = r"D:\Documents\general\downloads\pics\judi_cropped"
#
#     cropper = FaceCenteredCropper(
#         target_size=1024,
#         expansion_factor=1.8,  # Good for upper body
#         min_confidence=0.3  # Lower threshold for better detection
#     )
#
#     results = cropper.process_folder(input_folder, output_folder)
#     print(f"Processing complete!")





if __name__ == "__main__":
    output_folder = Path(__file__).parent / "output/judi"
    output_folder.mkdir(parents=True, exist_ok=True)
    input_folder = Path("D:\Documents\general\downloads\pics\judi")

    # FaceCenteredCropper.execute(input, output)
    cropper = FaceCenteredCropper(
        target_size=1024,
        expansion_factor=1.8,  # Good for upper body
        min_confidence=0.3  # Lower threshold for better detection
    )

    results = cropper.process_folder(input_folder, output_folder)
    print(f"Processing complete!")
    # if args.preview and results['success'] > 0:
    #     preview_images(output_path, max_images=9)