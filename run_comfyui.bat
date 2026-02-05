@echo off
title ComfyUI - RTX 4070 Laptop (8GB VRAM)

echo ======================================
echo   Activating Virtual Environment
echo ======================================
echo.

REM ---- Activate virtual environment
call ".venv\Scripts\activate.bat"

if %errorlevel% neq 0 (
    echo ❌ Failed to activate .venv!
    echo Make sure the .venv folder exists.
    pause
    exit /b
)

echo ✅ Virtual environment activated!
echo.

echo ======================================
echo     Setting CUDA / VRAM Options
echo ======================================
echo.

REM ---- Select GPU (if multiple GPUs exist)
set CUDA_VISIBLE_DEVICES=0

REM ---- Reduce CUDA memory fragmentation (safe)
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6

REM ---- Allow gradual memory growth (safe)
set TORCH_FORCE_GPU_ALLOW_GROWTH=true

REM ---- LOW VRAM hint (ComfyUI reads this)
set COMFYUI_LOWVRAM=1

echo ======================================
echo       Cleaning TEMP (optional)
echo ======================================
echo.

if exist temp (
    rmdir /s /q temp
)
mkdir temp

echo ======================================
echo        Starting ComfyUI
echo ======================================
echo.

REM ---- Recommended launch for 8GB VRAM
REM ----- python main.py --lowvram --cpu-vae --use-split-cross-attention
python main.py --lowvram --use-split-cross-attention

echo.
pause
