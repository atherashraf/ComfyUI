@echo off
title ComfyUI - 8GB VRAM (RTX 4070 Laptop)
echo ======================================
echo   Activating Virtual Environment
echo ======================================
echo.

REM ---- Activate .venv (Windows)
call ".venv\Scripts\activate.bat"

if %errorlevel% neq 0 (
    echo Failed to activate .venv!
    echo Make sure .venv folder exists in the same directory as this batch file.
    pause
    exit /b
)

echo Virtual environment activated!
echo.

echo ======================================
echo     Setting VRAM-Safe Environment
echo ======================================

REM --- Optional: Select GPU 0
set CUDA_VISIBLE_DEVICES=0

REM --- Prevent PyTorch memory fragmentation
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

REM --- Disable CUDA malloc async (fixes OOM on many GPUs)
set DISABLE_CUDA_MALLOC=1

REM --- Low VRAM mode for ComfyUI
set COMFYUI_LOWVRAM=1

REM --- Allow gradual memory growth
set TORCH_FORCE_GPU_ALLOW_GROWTH=true

echo.
echo ======================================
echo       Cleaning TEMP (optional)
echo ======================================
echo.

rmdir /s /q temp 2>nul
mkdir temp

echo ======================================
echo       Starting ComfyUI
echo ======================================
echo.

REM --- Launch with arguments
python main.py --lowvram --cpu-vae

echo.
pause
