
# SDXL Person LoRA — Automated Training Pipeline (Crop → Tag → Train)

This repo provides an end-to-end methodology + automation scripts to train a **single-person LoRA** (SDXL) using a small dataset (20+ images). It focuses on **repeatable structure**, **token consistency**, and **automation** (cropping, captioning, dataset layout, config generation, training + resume, monitoring).

---

## Why LoRA?

Instead of retraining a full diffusion model, LoRA trains small low-rank adapters injected into attention layers (few MB). It is:
- Fast(er) to train
- Low VRAM friendly (depending on config)
- Easy to share and reuse

---

## Core Concept: Token-Based Identity

To teach the model a specific person, we use a **unique trigger token**, wrapped in brackets:

- Token example: `[sjx]` or `[tyffani]`

During training, every image caption starts with that token, so the LoRA learns:
> “When this token appears, generate *that identity*.”

**Rule:** The token must be **rare** (avoid collisions with normal words).
- ✅ Good: `[sjx]`, `[tyffani]`, `[zxq_person]`
- ❌ Bad: `[woman]`, `[man]`, `[model]`

---

## Methodology (Step-by-Step)

### Phase 1 — Data Preparation (Most Important)

#### 1) Collect Images (20+ is a good start)
Recommended:
- 20–30 high quality images minimum (more is better)
- Variety of:
  - angles: front / side / 3/4
  - lighting: indoor / outdoor / studio
  - distance: close / mid / full body
  - expressions & poses

Avoid:
- heavy filters
- occluded faces (hands covering, sunglasses)
- multiple people in one image

---

#### 2) Cropping & Resizing (SDXL = 1024x1024)
This repo supports:
- **Full-body cropping (YOLO)** — recommended for person LoRAs
- **Face-centered cropping** — useful for portrait-heavy datasets

Goal:
- produce consistent 1024x1024 images
- keep the subject prominent
- reduce background noise (but keep some diversity)

---

#### 3) Captioning / Tagging (Token-based)
For each image `image.jpg`, create `image.txt`.

Caption guidelines:
- Start with the token
- Keep it simple and general (avoid over-memorization)
- Don’t describe permanent facial features (the model learns those from images)

Examples:

✅ Good:
- `[sjx], photo, full body`
- `a photo of [sjx] standing, studio lighting`
- `[sjx], portrait photo`

❌ Avoid overly specific memorization:
- `[sjx] wearing the red shirt on Tuesday at the beach with that palm tree`

Automation:
- BLIP can create a base caption (e.g., “a woman smiling…”)
- We replace generic person terms with your token (`[sjx]`) and optionally add YOLO scene objects.

---

## Repo Pipeline Overview (What the code does)

### Step 2 — Cropping
- `step2_full_body_cropping.py` (YOLO full-body)
- `step2_face_centered_cropping.py` (face-centered)

**Output:** cropped 1024 images saved into your working folder.

---

### Step 3 — Tagging (Captions + optional metadata)
- `step3_person_lora_tagger.py`

Creates:
- `image.txt` caption file beside each image
- optional per-image metadata `.json`
- `tagging_summary.json`

**Important:** Your token is passed into `quick_tag_person_lora(folder, token=...)`.
That token becomes your LoRA trigger token.

---

### Step 4 — Training Setup + Launch (Kohya)
- `step4_training_setup.py` orchestrates:
  1. dataset structure creation
  2. SDXL model download (if missing)
  3. TOML config generation
  4. training launch (accelerate)

The training launcher supports resume using `*-state` folders.

---

### Monitoring
- `monitor_training.py`

Checks:
- TensorBoard logs if present
- `.log` files tail
- `.safetensors` checkpoints in output folder

---

## Folder Structure (Actual Paths)

### Source Images
You keep raw photos here:
```

D:\Documents\general\downloads\pics<concept>\

```

### Cropped + Tagged Working Folder (example)
Usually produced by your entry script:
```

<repo_root>\output<concept>
image_001.jpg
image_001.txt
image_001_metadata.json (optional)
tagging_summary.json

```

### Training Workspace (default used by pipeline)
```

D:\lora_training
models
sd_xl_base_1.0.safetensors
sd_xl_refiner_1.0.safetensors (optional)
logs
... <concept>*dataset
1*<concept> <concept>_0000.jpg <concept>*0000.txt
...
output*<concept> <concept>_sdxl_lora-000005.safetensors
... <concept>-step00000500-state\   (resume state)

````

**Note on repeats:**  
The dataset folder prefix controls repeats per epoch.
Example: `1_<concept>` means repeat=1.

---

## Recommended Training Defaults (SDXL Person LoRA)

These are safe starting points:
- Resolution: **1024**
- Batch size: **1**
- Epochs: **20–30**
- Network dim (rank): **16–64** (your config uses 64)
- Alpha: typically **rank/2** or **rank**
- LR: ~ `4e-4` (your config uses `0.0004`)
- Optimizer: `AdamW8bit`
- Scheduler: `cosine_with_restarts`

Overfitting symptoms:
- identity appears even without token
- faces become “burnt”, distorted, too sharp
- style locks to a specific outfit/background

Underfitting symptoms:
- inconsistent face
- weak resemblance
- identity changes across generations

---

## How to Run (End-to-End)

### Example entry script
```py
from pathlib import Path

from da_apps.lora.steps.step2_full_body_cropping import yolo_fullbody_crop
from da_apps.lora.steps.step3_person_lora_tagger import quick_tag_person_lora
from da_apps.lora.steps.step4_training_setup import run_full_training_pipeline

if __name__ == "__main__":
    concept = "tyffani"
    token = concept  # token becomes [tyffani]

    data_dir = Path(rf"D:\Documents\general\downloads\pics\{concept}")
    out_dir = Path(__file__).parent / "output" / concept
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Crop to 1024
    yolo_fullbody_crop(data_dir, out_dir, target_size=1024)

    # 2) Create captions with token
    quick_tag_person_lora(out_dir, token=token)

    # 3) Setup + train
    run_full_training_pipeline(out_dir, concept_name=concept)
````

---

## Using the Trained LoRA

After training, you will get `.safetensors` checkpoints in:

```
D:\lora_training\output_<concept>\
```

In SD UIs (A1111 / ComfyUI), you typically load LoRA like:

```
<lora:tyffani_sdxl_lora:1>
```

Then prompt with your token:
**Positive prompt**

```
a photo of [tyffani], full body, standing in a studio, soft natural lighting, 85mm lens, ultra detailed
```

**Negative prompt**

```
low quality, blurry, bad anatomy, extra fingers, extra limbs, watermark, text, logo, cartoon, anime
```

Weight tips:

* `0.7–0.9` often looks more natural
* `1.0` standard
* `>1.1` stronger but can produce artifacts

---

## Troubleshooting

### Cropping misses person

* Some shots are too small/occluded
* Try face-centered crop for portrait sets
* Ensure `ultralytics` is installed

### Captions contain wrong token

* Ensure you pass `token=concept` into tagging
* Verify `.txt` files start with `[your_token]`

### Training won’t resume

* Resume requires `*-state` folders in output directory
* Make sure state saving is enabled in config

### Identity looks inconsistent

* Add more varied images
* Reduce epoch count if overfitting
* Keep captions simpler (avoid clothes/background specificity)

---

## Summary Workflow

1. **Gather** 20+ diverse images of one person
2. **Crop** to 1024 (YOLO full-body recommended)
3. **Caption** with a rare token `[token]`
4. **Prepare dataset** structure for Kohya
5. **Train** and watch samples/checkpoints
6. **Test** multiple checkpoints and choose the best
7. **Use** LoRA with `<lora:NAME:WEIGHT>` + token prompts

```

---

If you want, I can also:
- **merge your old README sections** (Rank/Alpha math + steps explanation) into this new one (keeping it concise),
- add a **“Recommended Settings by dataset size (20, 40, 80 images)”** section,
- add a **“Best practices for captions to avoid memorization”** section with examples from your BLIP/YOLO tagger.
```
