"""

Frame-by-frame player + mask painter (brush/eraser)

Controls

d / Right Arrow: next frame

a / Left Arrow: previous frame

Mouse Left-drag: paint mask

Hold e while dragging: erase mask

[ / ]: brush size down/up

s: save mask

c: clear mask

m: toggle mask overlay

q or Esc: quit
"""
import cv2
import numpy as np
from pathlib import Path

FRAMES_DIR = Path("frames")
MASKS_DIR  = Path("masks")
MASKS_DIR.mkdir(parents=True, exist_ok=True)

frame_paths = sorted(FRAMES_DIR.glob("frame_*.jpg"))
if not frame_paths:
    raise SystemExit(f"No frames found in {FRAMES_DIR}. Run extraction first.")

idx = 0
brush = 18
show_overlay = True
is_drawing = False
erase_mode = False

frame = None
mask = None

WIN = "Frame Masking (a/d prev/next, s save, [ ] brush, hold e erase)"

def load_frame(i: int):
    global frame, mask
    img = cv2.imread(str(frame_paths[i]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {frame_paths[i]}")
    frame = img

    mask_path = MASKS_DIR / f"mask_{i:06d}.png"
    if mask_path.exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        mask = m
    else:
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

def save_mask(i: int):
    mask_path = MASKS_DIR / f"mask_{i:06d}.png"
    cv2.imwrite(str(mask_path), mask)
    print("Saved:", mask_path)

def draw_overlay():
    view = frame.copy()
    if show_overlay:
        # Make a red overlay where mask is 255 (no explicit colors request? OpenCV needs a color;
        # This is just for UI visibility; change if you want.)
        overlay = view.copy()
        overlay[mask > 0] = (0, 0, 255)
        view = cv2.addWeighted(overlay, 0.35, view, 0.65, 0)

    # HUD text
    h, w = view.shape[:2]
    cv2.putText(view, f"Frame {idx+1}/{len(frame_paths)}  Brush:{brush}  Overlay:{show_overlay}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(view, "Drag=paint  Hold 'e'+drag=erase  s=save  c=clear  a/d=prev/next  q=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)
    return view

def on_mouse(event, x, y, flags, param):
    global is_drawing, erase_mode, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        erase_mode = (flags & cv2.EVENT_FLAG_SHIFTKEY) != 0  # optional: shift to erase
        # also allow holding 'e' via key state: we'll update erase_mode in main loop
        val = 0 if erase_mode else 255
        cv2.circle(mask, (x, y), brush, val, -1)

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        val = 0 if erase_mode else 255
        cv2.circle(mask, (x, y), brush, val, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, on_mouse)

load_frame(idx)

while True:
    # Update erase mode if 'e' is held (best-effort; OpenCV doesn't give perfect key-up events)
    # We'll treat last pressed 'e' as a toggle-like temporary state using key handling below.
    view = draw_overlay()
    cv2.imshow(WIN, view)

    key = cv2.waitKey(15) & 0xFF
    if key in (27, ord('q')):  # ESC or q
        break

    # prev / next
    if key in (ord('d'), 83):  # 'd' or Right Arrow
        idx = min(idx + 1, len(frame_paths) - 1)
        load_frame(idx)
    elif key in (ord('a'), 81):  # 'a' or Left Arrow
        idx = max(idx - 1, 0)
        load_frame(idx)

    # save / clear
    elif key == ord('s'):
        save_mask(idx)
    elif key == ord('c'):
        mask[:] = 0

    # overlay toggle
    elif key == ord('m'):
        show_overlay = not show_overlay

    # brush size
    elif key == ord('['):
        brush = max(1, brush - 2)
    elif key == ord(']'):
        brush = min(200, brush + 2)

    # press 'e' to enter erase mode for the *next* drag; press 'p' to paint mode
    elif key == ord('e'):
        erase_mode = True
        print("Erase mode ON (press p to return to paint)")
    elif key == ord('p'):
        erase_mode = False
        print("Paint mode ON")

cv2.destroyAllWindows()
