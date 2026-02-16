from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from da_apps.lora.steps.kohya_setup.auto_training import train_sdxl_lora_auto
from da_apps.lora.steps.kohya_setup.download_sdxl import download_sdxl_models
from da_apps.lora.steps.kohya_setup.kohya_config import generate_kohya_config
from da_apps.lora.steps.kohya_setup.structure_setup import prepare_kohya_structure, DatasetStructureConfig


@dataclass
class PipelineConfig:
    working_dir: Path = Path(r"D:\lora_training")
    concept_name: str = "judi"
    repeats: int = 1


def run_full_training_pipeline(
    input_images_folder: Path,
    working_dir: str | Path = r"D:\lora_training",
    concept_name: str = "judi",
    repeats: int = 1,
) -> Dict[str, Path]:
    print(f"=== Starting LoRA Pipeline for '{concept_name}' repeated {repeats} ===")

    work_path = Path(working_dir)
    models_path = work_path / "models"
    base_model_file = models_path / "sd_xl_base_1.0.safetensors"
    # base_model_file = models_path / "pwsbeauty_v04.safetensors"

    output_dir = work_path / f"output_{concept_name}"
    logs_dir = work_path / "logs"
    config_file = work_path / f"{concept_name}_config.toml"

    # Step 1: dataset
    print("\n--- Step 1: Preparing Dataset ---")
    ds_cfg = DatasetStructureConfig(repeats=repeats, overwrite=True)
    dataset_parent = prepare_kohya_structure(Path(input_images_folder), work_path, concept_name, ds_cfg)

    # Step 2: models
    print("\n--- Step 2: Checking Model ---")
    if not base_model_file.exists():
        print(f"⚠ Model not found at {base_model_file}")
        print("Downloading now...")
        models_path.mkdir(parents=True, exist_ok=True)
        download_sdxl_models(str(models_path))
    else:
        print(f"✓ Found base model: {base_model_file}")

    # Step 3: config
    print("\n--- Step 3: Generating Config ---")
    config_path = generate_kohya_config(
        output_file=str(config_file),
        train_data_dir=dataset_parent,
        output_dir=output_dir,
        logging_dir=logs_dir,
        model_path=base_model_file,
        concept_name=concept_name,
    )

    # Step 4: train
    print("\n--- Step 4: Launching Training ---")
    train_sdxl_lora_auto(config_path)

    return {
        "work_dir": work_path,
        "dataset_dir": dataset_parent,
        "models_dir": models_path,
        "output_dir": output_dir,
        "logs_dir": logs_dir,
        "config_file": config_path,
    }
