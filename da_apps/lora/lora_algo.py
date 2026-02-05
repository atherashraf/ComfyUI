from pathlib import Path

from da_apps.lora.steps.step2_full_body_cropping import yolo_fullbody_crop
from da_apps.lora.steps.step3_person_lora_tagger import quick_tag_person_lora
from da_apps.lora.steps.step4_training_setup import run_full_training_pipeline


if __name__ == "__main__":
    concept = "tyffani"        # token will be [tyffani]
    token = concept

    base_dir = Path(__file__).parent
    lora_img_dir = base_dir / "output" / concept
    lora_img_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(rf"D:\Documents\general\downloads\pics\{concept}")

    # Step 2: crop
    yolo_fullbody_crop(data_dir, lora_img_dir, target_size=1024)

    # Step 3: tag (IMPORTANT: pass token)
    quick_tag_person_lora(lora_img_dir, token=token)

    # Step 4: train
    run_full_training_pipeline(lora_img_dir, concept_name=concept, repeats=1)
