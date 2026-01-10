import cv2
from pathlib import Path

video_path = "movie.mp4"
output_dir = Path("frames")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out_path = output_dir / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(out_path), frame)
    frame_idx += 1

cap.release()

print(f"Extracted {frame_idx} frames to '{output_dir}'")
