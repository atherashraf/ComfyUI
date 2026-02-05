from pathlib import Path

from da_apps.lora.steps.step2_full_body_cropping import yolo_fullbody_crop
from da_apps.lora.steps.step3_person_lora_tagger import quick_tag_person_lora
from da_apps.lora.steps.step4_training_setup import run_full_training_pipeline

if __name__ == "__main__":
    concept = "tyffani"

    # Base directory = folder where this script lives
    BASE_DIR = Path(__file__).parent

    # Output folder for cropped + tagged images
    lora_img_dir = BASE_DIR / "output" / concept
    lora_img_dir.mkdir(parents=True, exist_ok=True)

    # Raw input images
    data_dir = Path(rf"D:\Documents\general\downloads\pics\{concept}")

    is_images_cropped = False
    is_images_tagged = False
    is_training_setup_done = False

    if not is_images_cropped:
        yolo_fullbody_crop(data_dir, lora_img_dir, target_size=1024)

    if not is_images_tagged:
        quick_tag_person_lora(lora_img_dir, concept)

    if not is_training_setup_done:
        run_full_training_pipeline(lora_img_dir, concept_name=concept)
