from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class TaggerConfig:
    token: str
    use_blip: bool = True
    use_yolo: bool = True
    caption_ext: str = ".txt"
    save_json: bool = True
    preview: bool = False


class PersonLoRATagger:
    def __init__(self, cfg: TaggerConfig):
        self.cfg = cfg
        self.token_raw = cfg.token
        self.token = f"[{cfg.token}]"

        self.blip_processor = None
        self.blip_model = None
        if cfg.use_blip:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("✓ BLIP loaded")
            except Exception as e:
                print(f"⚠ BLIP disabled ({e})")
                self.cfg.use_blip = False

        self.yolo_model = None
        if cfg.use_yolo:
            try:
                from ultralytics import YOLO  # type: ignore
                self.yolo_model = YOLO("yolov8n.pt")
                print("✓ YOLO loaded")
            except Exception as e:
                print(f"⚠ YOLO disabled ({e})")
                self.cfg.use_yolo = False

        print(f"Tagger initialized with token: {self.token}")

    def _blip_caption(self, image_path: Path) -> Optional[str]:
        if not self.blip_model or not self.blip_processor:
            return None
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50)
            return self.blip_processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            return None

    def _yolo_objects(self, image_path: Path) -> List[str]:
        if not self.yolo_model:
            return []
        try:
            results = self.yolo_model(str(image_path), verbose=False)
            objs = []
            scene_objects = {"chair", "table", "car", "tree", "dog", "cat", "book", "phone", "cup", "bottle", "bag"}
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = self.yolo_model.names.get(cls_id, "")
                    conf = float(box.conf[0])
                    if name in scene_objects and conf >= 0.5:
                        objs.append(name)
            return sorted(set(objs))
        except Exception:
            return []

    def _make_caption(self, blip: Optional[str], objects: List[str]) -> str:
        # Keep captions training-friendly (don’t over-memorize clothes/backgrounds)
        if blip:
            low = blip.lower()
            for term in ["a man", "a woman", "a person", "someone", "the person", "a girl", "a boy"]:
                if term in low:
                    low = low.replace(term, self.token, 1)
                    break
            base = f"{self.token}, {low}"
        else:
            base = f"{self.token}, photo, full body"

        if objects:
            base += f", with {', '.join(objects)}"
        return base.strip().replace("  ", " ")

    def tag_folder(self, folder: Path) -> Dict[str, Any]:
        folder = Path(folder)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
        imgs.sort()

        all_meta: List[Dict[str, Any]] = []

        print("=" * 60)
        print(f"TAGGING {len(imgs)} IMAGES")
        print(f"Token: {self.token}")
        print("=" * 60)

        for img_path in imgs:
            meta: Dict[str, Any] = {
                "image_path": str(img_path),
                "token": self.token,
                "timestamp": datetime.now().isoformat(),
            }
            blip = self._blip_caption(img_path) if self.cfg.use_blip else None
            if blip:
                meta["blip_caption"] = blip

            objects = self._yolo_objects(img_path) if self.cfg.use_yolo else []
            if objects:
                meta["detected_objects"] = objects

            caption = self._make_caption(blip, objects)
            meta["caption"] = caption

            # save caption file next to image
            cap_file = folder / f"{img_path.stem}{self.cfg.caption_ext}"
            cap_file.write_text(caption, encoding="utf-8")

            if self.cfg.save_json:
                (folder / f"{img_path.stem}_metadata.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )

            all_meta.append(meta)

        summary = {
            "token": self.token,
            "total_images": len(imgs),
            "tagged_images": len(all_meta),
            "timestamp": datetime.now().isoformat(),
            "images": all_meta,
        }
        (folder / "tagging_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\n" + "=" * 60)
        print("CAPTION EXAMPLES:")
        print("=" * 60)
        for i, m in enumerate(all_meta[:5]):
            print(f"{i + 1}. {m['caption']}")

        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("1. Review .txt captions")
        print("2. Remove overly specific clothing/background details if memorization happens")
        print(f"3. Ensure captions start with your token: {self.token}")
        return summary


def quick_tag_person_lora(input_folder: Path, token: str) -> Dict[str, Any]:
    """
    Keeps your existing function shape (folder + token) :contentReference[oaicite:5]{index=5}
    """
    cfg = TaggerConfig(token=token, use_blip=True, use_yolo=True, caption_ext=".txt", save_json=True, preview=False)
    return PersonLoRATagger(cfg).tag_folder(Path(input_folder))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--token", default="sjx")
    ap.add_argument("--no-blip", action="store_true")
    ap.add_argument("--no-yolo", action="store_true")
    args = ap.parse_args()

    cfg = TaggerConfig(token=args.token, use_blip=not args.no_blip, use_yolo=not args.no_yolo)
    PersonLoRATagger(cfg).tag_folder(Path(args.input))
