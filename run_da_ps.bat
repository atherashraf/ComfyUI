@echo off
REM Stop on errors
setlocal EnableExtensions EnableDelayedExpansion

REM Go to the directory where this script lives
cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Force ONLY NVIDIA GPU 0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0

REM PyTorch allocator tuning (reduces fragmentation / OOM)
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8

REM Optional CPU thread limits
if not defined OMP_NUM_THREADS set OMP_NUM_THREADS=8
if not defined MKL_NUM_THREADS set MKL_NUM_THREADS=8

REM Run your app
python -m da_apps.da_main %*

endlocal
