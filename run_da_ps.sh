#!/usr/bin/env bash
# ------------------------------------------------
#
## Normal run
 #./run_server.sh
 #
 ## With specific GPU
 #CUDA_VISIBLE_DEVICES=1 ./run_server.sh
 #
 ## With different memory settings
 #PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256" ./run_server.sh
 #
 ## Different ComfyUI location
 #COMFYUI_PATH="/path/to/ComfyUI" ./run_server.sh
 # ------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "Error: .venv not found. Please create virtual environment first."
    echo "Run: python -m venv .venv"
    echo "Then: source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate

# Optional: Clear CUDA cache before starting
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Force ONLY NVIDIA GPU 0 to be visible to CUDA apps
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# PyTorch allocator tuning to reduce fragmentation / OOM
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
