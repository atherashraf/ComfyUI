from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DatasetStructureConfig:
    repeats: int = 1           # controls repeats (folder prefix)
    caption_ext: str = ".txt"
    default_caption: str = "photo of {concept}, full body"
    overwrite: bool = True


def prepare_kohya_structure(source_folder: Path, output_folder: Path, concept_name: str = "judi",
                           cfg: DatasetStructureConfig = DatasetStructureConfig()) -> Path:
    """
    Creates:
      <output>/<concept>_dataset/<repeats>_<concept>/concept_0001.jpg + .txt
    """
    source_path = Path(source_folder)
    work_path = Path(output_folder)
    dataset_parent = work_path / f"{concept_name}_dataset"
    concept_folder = dataset_parent / f"{cfg.repeats}_{concept_name}"

    if concept_folder.exists() and cfg.overwrite:
        print(f"Cleaning up old folder: {concept_folder}")
        shutil.rmtree(concept_folder)

    concept_folder.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images: List[Path] = []
    for p in source_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    images.sort()

    print(f"Preparing {len(images)} images for Kohya_SS...")

    for i, img_path in enumerate(images):
        img_ext = img_path.suffix.lower()
        new_img_name = f"{concept_name}_{i:04d}{img_ext}"
        new_img_path = concept_folder / new_img_name
        shutil.copy2(img_path, new_img_path)

        src_caption = source_path / f"{img_path.stem}{cfg.caption_ext}"
        new_caption_path = concept_folder / f"{concept_name}_{i:04d}{cfg.caption_ext}"

        if src_caption.exists():
            shutil.copy2(src_caption, new_caption_path)
        else:
            new_caption_path.write_text(cfg.default_caption.format(concept=concept_name), encoding="utf-8")

    print(f"\nKohya_SS dataset prepared at: {concept_folder}")
    print(f"Total images: {len(images)}")
    return dataset_parent
