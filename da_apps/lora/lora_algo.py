from pathlib import Path

from da_apps.lora.steps.step2_full_body_cropping import yolo_fullbody_crop
from da_apps.lora.steps.step3_person_lora_tagger import quick_tag_person_lora
from da_apps.lora.steps.step4_training_setup import run_full_training_pipeline


if __name__ == "__main__":
    concept = "suma"        # token will be [concept]
    repeats ="1"
    token = concept
    current_step = 1
    base_dir = Path(__file__).parent
    lora_img_dir = base_dir / "output" / concept
    lora_img_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(rf"D:\Documents\general\downloads\pics\{concept}")
    # Step 2: crop
    if current_step <= 2:
        yolo_fullbody_crop(data_dir, lora_img_dir, target_size=1024)
    if current_step <= 3:
        # Step 3: tag (IMPORTANT: pass token)
        quick_tag_person_lora(lora_img_dir, token=token)

    if current_step <= 4:
        # Step 4: train
        run_full_training_pipeline(lora_img_dir, concept_name=concept, repeats=repeats)
