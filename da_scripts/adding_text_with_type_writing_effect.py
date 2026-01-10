from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------
# Paths / Inputs
# -------------------------
data_dir = Path(__file__).parent / "data"
# input_video = str(data_dir / "tm-mahi-01.mp4")          # ✅ set your input
input_video = "D:/DevelopmentProjects/PhotoAI/ComfyUI/output/video/ByteDance-Seedance_00032_.mp4"
output_video = str(data_dir / "output_text.mp4")        # ✅ set your output
font_path = str(data_dir / "Pak Nastaleeq Regular.otf") # ✅ or any .ttf/.otf

# -------------------------
# Video open
# -------------------------
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {input_video}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
video_duration = (total_frames / fps) if total_frames else 10.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError("VideoWriter did not open. Check codec/output path.")

# -------------------------
# Text segments (simple)
# -------------------------
lines = [
    "Happy New Year",
    # "This is line 2",
]

segment = video_duration / max(1, len(lines))
poem = [{"text": t, "start": segment * i, "end": segment * (i + 1)} for i, t in enumerate(lines)]

# -------------------------
# Style settings
# -------------------------
# font_size = 48
# pil_font = ImageFont.truetype(font_path, font_size)
font_size = 100
# pil_font = ImageFont.load_default()
pil_font = ImageFont.truetype("arial.ttf", 48)

base_y = height // 2
line_spacing = int(font_size * 1.3)

fill_rgb = (255, 0, 0)       # text color
stroke_rgb = (0, 0, 0)       # outline color
stroke_w = 3                 # outline thickness

shadow_rgb = (0, 0, 0)       # shadow color
shadow_offset = (3, 3)       # shadow offset (x,y)

def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def draw_text_styled(frame_bgr, text, center_xy):
    """Draw centered styled text on a BGR frame using PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    cx, cy = center_xy

    # Shadow
    if shadow_offset:
        sx, sy = shadow_offset
        draw.text(
            (cx + sx, cy + sy),
            text,
            font=pil_font,
            fill=shadow_rgb,
            anchor="mm",
        )

    # Main text with outline
    draw.text(
        (cx, cy),
        text,
        font=pil_font,
        fill=fill_rgb,
        stroke_width=stroke_w,
        stroke_fill=stroke_rgb,
        anchor="mm",
    )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# -------------------------
# Render loop
# -------------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_idx / fps

    # Find active line for this time
    active = [ln for ln in poem if ln["start"] <= current_time < ln["end"]]

    for i, ln in enumerate(active):
        dur = max(0.001, ln["end"] - ln["start"])
        prog = ease((current_time - ln["start"]) / dur)

        # Type-on effect
        show_chars = max(1, int(len(ln["text"]) * prog))
        partial = ln["text"][:show_chars]

        # Vertical stacking (if multiple active)
        y = base_y + (i - (len(active) - 1) / 2) * line_spacing
        frame = draw_text_styled(frame, partial, (width // 2, int(y)))

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("DONE:", output_video)
import subprocess

# -------------------------
# WhatsApp-compatible re-encode via FFmpeg
# -------------------------
# This fixes: pixel format, H.264 profile, and "faststart" (moov atom position).
output_whatsapp = str(data_dir / "output_text_whatsapp.mp4")
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-i", output_video,
    "-c:v", "libx264",
    "-profile:v", "baseline",
    "-level", "3.0",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    output_whatsapp,
]

try:
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    print("DONE (WhatsApp):", output_whatsapp)
except FileNotFoundError:
    print("❌ FFmpeg not found. Install it and ensure 'ffmpeg' is in PATH.")
except subprocess.CalledProcessError as e:
    print("❌ FFmpeg failed.")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
