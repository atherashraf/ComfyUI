from pathlib import Path
import cv2
import numpy as np
from pytesseract import pytesseract

# ----------------------------------------
#   IMPORTANT: Set path for Tesseract
#   (you aren't using OCR now, but this keeps it ready)
# ----------------------------------------
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

data_dir = Path(__file__).parent / "data"
input_video = str(data_dir / "Wan2.5_00021_.mp4")
output_video = str(data_dir / "output_clean.mp4")

# Open input video
cap = cv2.VideoCapture(input_video)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))


def inpaint_text(frame_bgr, mask):
    """
    Takes original frame + binary mask (255 on text), dilates a bit,
    then inpaints. No OCR here, just pure mask-based.
    """
    # Thicken mask slightly so we cover full glyphs
    kernel = np.ones((3, 3), np.uint8)   # smaller kernel than before
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    cleaned = cv2.inpaint(
        frame_bgr,
        mask_dilated,
        inpaintRadius=10,                 # slightly smaller radius
        flags=cv2.INPAINT_TELEA
    )

    return cleaned, mask_dilated


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # Threshold to detect bright text (you can adjust 200 → 150 or 220)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Optional: expand mask slightly (helps fill full letters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply inpainting (Telea algorithm is best for text)
    # cleaned_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    cleaned_frame, mask = inpaint_text(frame, mask)
    out.write(cleaned_frame)

cap.release()
out.release()

print("🎉 Done! Clean video saved as:", output_video)
