#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

echo "======================================"
echo "  Activating Virtual Environment"
echo "======================================"
echo

if [ ! -f ".venv/bin/activate" ]; then
  echo "❌ .venv not found!"
  echo "Create it with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -U pip setuptools wheel"
  exit 1
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"
echo "✅ Virtual environment activated!"
echo

echo "======================================"
echo "    Setting VRAM-Safe Environment"
echo "======================================"

export CUDA_VISIBLE_DEVICES=0

# Helps reduce CUDA memory fragmentation + avoids sudden spikes
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,garbage_collection_threshold:0.6"

# Sometimes helps stability on some systems (safe to keep)
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo
echo "======================================"
echo "      Cleaning TEMP (optional)"
echo "======================================"
echo

rm -rf temp 2>/dev/null || true
mkdir -p temp

echo "======================================"
echo "      Starting ComfyUI"
echo "======================================"
echo

# Log to file too (very useful if the terminal just shows "Killed")
#exec python -u main.py --lowvram --cpu-vae 2>&1 | tee -a comfyui.log
#exec python main.py --lowvram --cpu-vae --force-fp16 --disable-smart-memory
#exec python main.py --lowvram --cpu-vae --use-split-cross-attention
exec python -u main.py --lowvram --cpu-vae --use-split-cross-attention --disable-smart-memory 2>&1 | tee -a comfyui.log
