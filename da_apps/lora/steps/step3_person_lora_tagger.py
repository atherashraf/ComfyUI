import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
import os
from datetime import datetime


class PersonLoRATagger:
    def __init__(self, token="sjx", use_blip=True, use_yolo=True):
        """
        Initialize tagger for person LoRA training.

        Args:
            token: Unique identifier token (e.g., "sjx")
            use_blip: Use BLIP for automatic captioning
            use_yolo: Use YOLO for pose/object detection
        """
        self.token = f"[{token}]"  # Wrap in brackets for SD
        self.token_raw = token

        # Initialize BLIP if requested
        self.blip_processor = None
        self.blip_model = None
        if use_blip:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("✓ BLIP model loaded for automatic captioning")
            except ImportError:
                print("⚠ Install transformers for BLIP: pip install transformers")
                use_blip = False

        # Initialize YOLO if requested
        self.yolo_model = None
        if use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO('yolov8n.pt')  # Nano model for person/object detection
                print("✓ YOLO model loaded for pose/object detection")
            except ImportError:
                print("⚠ Install ultralytics for YOLO: pip install ultralytics")
                use_yolo = False

        # Pose and clothing keywords database
        self.pose_keywords = {
            'standing': ['standing', 'upright', 'on feet'],
            'sitting': ['sitting', 'seated', 'on chair', 'on bench'],
            'walking': ['walking', 'strolling', 'in motion'],
            'running': ['running', 'jogging'],
            'lying': ['lying down', 'reclining', 'on bed'],
            'front': ['front view', 'facing forward', 'frontal'],
            'side': ['side view', 'profile', 'from side'],
            'back': ['back view', 'from behind', 'rear view'],
            'three_quarter': ['3/4 view', 'three quarter view', 'angled'],
            'arms_crossed': ['arms crossed', 'crossed arms'],
            'hands_on_hips': ['hands on hips', 'akimbo'],
            'leaning': ['leaning', 'against wall'],
            'crouching': ['crouching', 'squatting'],
            'jumping': ['jumping', 'in air']
        }

        self.clothing_keywords = {
            'top': ['shirt', 't-shirt', 'blouse', 'sweater', 'hoodie', 'jacket', 'coat', 'top'],
            'bottom': ['pants', 'jeans', 'trousers', 'shorts', 'skirt', 'leggings'],
            'dress': ['dress', 'gown', 'frock'],
            'footwear': ['shoes', 'sneakers', 'boots', 'sandals', 'heels'],
            'accessories': ['glasses', 'hat', 'cap', 'scarf', 'bag', 'jewelry']
        }

        print(f"Tagger initialized with token: {self.token}")

    def detect_pose_yolo(self, image_path):
        """Detect pose using YOLO pose estimation."""
        if not self.yolo_model:
            return None

        try:
            results = self.yolo_model(image_path, verbose=False)

            # Check for person detection
            for result in results:
                if len(result.boxes) > 0:
                    # Get keypoints if available (YOLO pose model)
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy[0].cpu().numpy()

                        # Simple pose estimation from keypoints
                        if len(keypoints) >= 17:  # COCO keypoints
                            # Check if person is sitting vs standing
                            # Compare hip and knee positions
                            left_hip = keypoints[11]  # Index 11: left hip
                            left_knee = keypoints[13]  # Index 13: left knee

                            if left_hip[1] > 0 and left_knee[1] > 0:
                                # If knees are significantly below hips, might be sitting
                                if abs(left_knee[1] - left_hip[1]) < 50:  # Small vertical difference
                                    return "sitting"
                                else:
                                    return "standing"

            return None
        except Exception as e:
            print(f"YOLO pose detection error: {e}")
            return None

    def detect_objects_yolo(self, image_path):
        """Detect objects in the image using YOLO."""
        if not self.yolo_model:
            return []

        try:
            results = self.yolo_model(image_path, verbose=False)
            objects = []

            # Common object classes for scene description
            scene_objects = ['chair', 'table', 'car', 'tree', 'person', 'dog', 'cat',
                             'book', 'phone', 'cup', 'bottle', 'bag']

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]

                    if class_name in scene_objects and class_name != 'person':
                        confidence = float(box.conf[0])
                        if confidence > 0.5:  # Only keep confident detections
                            objects.append(class_name)

            return list(set(objects))  # Remove duplicates
        except Exception as e:
            print(f"YOLO object detection error: {e}")
            return []

    def generate_blip_caption(self, image_path):
        """Generate automatic caption using BLIP."""
        if not self.blip_model:
            return None

        try:
            image = Image.open(image_path).convert('RGB')

            # Preprocess and generate caption
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

            return caption
        except Exception as e:
            print(f"BLIP caption error: {e}")
            return None

    def analyze_image_manual(self, image_path):
        """Manually analyze image for pose, clothing, and background."""
        image = Image.open(image_path).convert('RGB')
        w, h = image.size

        # Get basic image info
        aspect_ratio = w / h

        # Determine likely pose based on aspect ratio and manual inspection
        if aspect_ratio < 0.7:
            # Tall portrait - likely full body standing
            pose_guess = "standing, full body"
        elif aspect_ratio > 1.3:
            # Wide landscape - might be sitting or multiple people
            pose_guess = "wide shot"
        else:
            # Square-ish - hard to tell
            pose_guess = "portrait"

        return {
            'aspect_ratio': aspect_ratio,
            'pose_guess': pose_guess,
            'likely_full_body': aspect_ratio < 0.8  # Tall images more likely full body
        }

    def generate_caption(self, image_path, blip_caption=None, pose=None, objects=None,
                         clothing_desc="", background_desc="", style="photo"):
        """
        Generate a caption for the image.

        Args:
            image_path: Path to image
            blip_caption: Optional BLIP-generated caption
            pose: Detected pose (e.g., "standing", "sitting")
            objects: List of detected objects
            clothing_desc: Description of clothing
            background_desc: Description of background
            style: Image style (photo, portrait, etc.)
        """

        # Start with token
        caption_parts = [self.token]

        # Add pose if known
        if pose:
            if pose in ["standing", "sitting", "walking", "running"]:
                caption_parts.append(pose)
            elif "view" in pose:
                caption_parts.append(pose)

        # Add clothing description
        if clothing_desc:
            caption_parts.append(f"wearing {clothing_desc}")

        # Add style
        caption_parts.append(style)

        # Add background if provided
        if background_desc:
            caption_parts.append(f"in {background_desc}")

        # Add detected objects if any
        if objects:
            # Filter out "person" from objects
            objects = [obj for obj in objects if obj != 'person']
            if objects:
                caption_parts.append(f"with {', '.join(objects)}")

        # If we have a BLIP caption, use it as base but replace generic terms
        if blip_caption:
            # Replace generic person references with our token
            blip_caption = blip_caption.lower()

            # Common person references to replace
            person_terms = ['a man', 'a woman', 'a person', 'someone', 'the person',
                            'a girl', 'a boy', 'the man', 'the woman']

            for term in person_terms:
                if term in blip_caption:
                    blip_caption = blip_caption.replace(term, self.token)
                    break  # Only replace first occurrence

            # Use BLIP caption as base, add our prefix
            final_caption = f"{self.token}, {blip_caption}"
        else:
            # Build caption from parts
            final_caption = ", ".join(caption_parts)

        # Clean up: remove double spaces, trailing commas
        final_caption = final_caption.replace("  ", " ").strip()
        if final_caption.endswith(","):
            final_caption = final_caption[:-1]

        return final_caption

    def tag_single_image(self, image_path, use_auto_detection=True):
        """Tag a single image with comprehensive metadata."""
        print(f"\nTagging: {Path(image_path).name}")

        metadata = {
            'image_path': str(image_path),
            'token': self.token,
            'timestamp': datetime.now().isoformat()
        }

        # Get BLIP caption if available
        blip_caption = None
        if use_auto_detection and self.blip_model:
            blip_caption = self.generate_blip_caption(image_path)
            if blip_caption:
                print(f"  BLIP caption: {blip_caption}")
                metadata['blip_caption'] = blip_caption

        # Detect pose and objects with YOLO
        pose = None
        objects = []
        if use_auto_detection and self.yolo_model:
            pose = self.detect_pose_yolo(image_path)
            if pose:
                print(f"  Detected pose: {pose}")
                metadata['detected_pose'] = pose

            objects = self.detect_objects_yolo(image_path)
            if objects:
                print(f"  Detected objects: {', '.join(objects)}")
                metadata['detected_objects'] = objects

        # Manual analysis
        manual_info = self.analyze_image_manual(image_path)
        metadata.update(manual_info)

        # Generate final caption
        final_caption = self.generate_caption(
            image_path=image_path,
            blip_caption=blip_caption,
            pose=pose,
            objects=objects,
            clothing_desc="",  # Can be filled manually
            background_desc="",  # Can be filled manually
            style="photo"
        )

        metadata['caption'] = final_caption
        print(f"  Final caption: {final_caption}")

        return metadata

    def tag_folder(self, input_folder, output_folder=None,
                   caption_ext=".txt", json_output=True, preview=False):
        """
        Tag all images in a folder.

        Args:
            input_folder: Folder with images
            output_folder: Where to save captions (default: same as input)
            caption_ext: Extension for caption files (.txt or .caption)
            json_output: Save metadata as JSON
            preview: Show preview of tagged images
        """
        input_path = Path(input_folder)

        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path

        # Find all image files
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))

        print(f"\n{'=' * 60}")
        print(f"TAGGING {len(image_files)} IMAGES")
        print(f"Token: {self.token}")
        print(f"Output: {output_path}")
        print(f"{'=' * 60}")

        all_metadata = []
        tagged_count = 0

        for img_path in image_files:
            try:
                # Tag image
                metadata = self.tag_single_image(img_path, use_auto_detection=True)

                # Save caption as text file
                caption_file = output_path / f"{img_path.stem}{caption_ext}"
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(metadata['caption'])

                # Save metadata to JSON if requested
                if json_output:
                    metadata_file = output_path / f"{img_path.stem}_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)

                all_metadata.append(metadata)
                tagged_count += 1

                # Optional preview
                if preview and tagged_count <= 5:  # Show first 5
                    self.preview_tagged_image(img_path, metadata['caption'])

            except Exception as e:
                print(f"✗ Error tagging {img_path.name}: {e}")

        # Save summary
        summary = {
            'token': self.token,
            'total_images': len(image_files),
            'tagged_images': tagged_count,
            'timestamp': datetime.now().isoformat(),
            'images': all_metadata
        }

        summary_file = output_path / "tagging_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"TAGGING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Tagged: {tagged_count}/{len(image_files)} images")
        print(f"Caption files saved with extension: {caption_ext}")
        if json_output:
            print(f"Metadata saved as JSON files")
        print(f"Summary: {summary_file}")

        return summary

    def preview_tagged_image(self, image_path, caption):
        """Preview image with caption."""
        try:
            import matplotlib.pyplot as plt

            img = Image.open(image_path).convert('RGB')

            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"Caption: {caption}", fontsize=10, wrap=True)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("  (Install matplotlib for preview: pip install matplotlib)")

    def manual_caption_editor(self, input_folder):
        """Interactive editor to manually review and edit captions."""
        input_path = Path(input_folder)

        # Find caption files
        caption_files = list(input_path.glob('*.txt')) + list(input_path.glob('*.caption'))

        print(f"\nManual Caption Editor")
        print(f"Found {len(caption_files)} caption files")
        print("Press Enter to keep current caption, or type new caption")

        for caption_file in caption_files:
            # Find corresponding image
            img_file = input_path / f"{caption_file.stem}.jpg"
            if not img_file.exists():
                img_file = input_path / f"{caption_file.stem}.png"

            if img_file.exists():
                # Display image
                try:
                    import matplotlib.pyplot as plt

                    img = Image.open(img_file).convert('RGB')
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Current caption:", fontsize=9)
                    plt.show()
                except:
                    pass

                # Read current caption
                with open(caption_file, 'r', encoding='utf-8') as f:
                    current_caption = f.read().strip()

                print(f"\n{img_file.name}")
                print(f"Current: {current_caption}")

                # Get new caption
                new_caption = input(f"New caption (Enter to keep): ").strip()

                if new_caption:
                    # Save new caption
                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write(new_caption)
                    print(f"✓ Caption updated")
                else:
                    print(f"✓ Kept original caption")

        print("\nCaption editing complete!")


# Quick tagging function
def quick_tag_person_lora(INPUT_FOLDER):
    """One-click tagging for person LoRA."""

    # Configuration
    # INPUT_FOLDER = r"D:\Documents\general\downloads\pics\judi_cropped"  # Your cropped images
    TOKEN = "sjx"  # Your unique token
    USE_BLIP = True  # Auto-captioning
    USE_YOLO = True  # Pose/object detection

    print("=" * 60)
    print("PERSON LoRA TAGGING SYSTEM")
    print("=" * 60)

    # Initialize tagger
    tagger = PersonLoRATagger(
        token=TOKEN,
        use_blip=USE_BLIP,
        use_yolo=USE_YOLO
    )

    # Tag all images
    summary = tagger.tag_folder(
        input_folder=INPUT_FOLDER,
        output_folder=None,  # Save in same folder
        caption_ext=".txt",  # Standard extension
        json_output=True,  # Save metadata
        preview=False  # Don't show previews
    )

    # Show statistics
    print("\n" + "=" * 60)
    print("CAPTION EXAMPLES:")
    print("=" * 60)

    # Show first few captions
    for i, metadata in enumerate(summary['images'][:5]):
        print(f"{i + 1}. {metadata['caption']}")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Review caption files (.txt) in folder")
    print("2. Edit any captions that seem inaccurate")
    print("3. Remove any memorized details (specific clothing, backgrounds)")
    print("4. Ensure all captions start with your token: [sjx]")
    print("5. For training, each image should have a corresponding .txt file")

    return summary




if __name__ == "__main__":
    # Uncomment one of these:

    # Option 1: Quick one-click tagging
    lora_img_dir = Path('data/lora')
    quick_tag_person_lora(lora_img_dir)

