@echo off
title ComfyUI - RTX 4070 Laptop (8GB VRAM) - Optimized

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

REM ---- CPU offloading for VAE and attention
set COMFYUI_CPU_OFFLOAD=1

echo ======================================
echo       Cleaning TEMP (optional)
echo ======================================
echo.

if exist temp (
    rmdir /s /q temp
)
mkdir temp

echo ======================================
echo     System Memory Optimization
echo ======================================
echo.

REM ---- Set high priority for ComfyUI process
wmic process where name="python.exe" call setpriority "high priority" 2>nul

REM ---- Display current page file info (requires admin)
echo Current Page File Settings:
wmic pagefile list /format:list 2>nul
echo.

echo ======================================
echo        Starting ComfyUI
echo ======================================
echo.

REM ---- AGGRESSIVE MEMORY SAVING MODE (best for 8GB VRAM)
REM ---- Options explained:
REM ---- --lowvram: Keep models in VRAM only when needed
REM ---- --cpu-vae: Run VAE on CPU (saves 1-2GB VRAM)
REM ---- --use-split-cross-attention: Optimized attention (for SDXL)
REM ---- --disable-smart-memory: More aggressive offloading
REM ---- --reserve-vram 0.5: Reserve 0.5GB for system (adjust as needed)

python main.py --lowvram --cpu-vae --use-split-cross-attention --disable-smart-memory

@REM python main.py --novram --cpu-vae --use-split-cross-attention --highvram

REM ---- Alternative if above fails (slower but more stable):
REM ---- python main.py --novram --cpu-vae --use-split-cross-attention

echo.
echo ======================================
echo        ComfyUI Has Exited
echo ======================================
pause