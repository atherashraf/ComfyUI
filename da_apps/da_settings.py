from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
input_dir = BASE_DIR / "input"
output_dir = BASE_DIR / "output"
static_dir = BASE_DIR / "static"
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
static_dir.mkdir(parents=True, exist_ok=True)