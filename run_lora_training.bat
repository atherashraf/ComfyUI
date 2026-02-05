@echo off
:: 1. Set the paths
set "PROJECT_ROOT=D:\DevProjects\PhotoAI\ComfyUI"
set "PYTHON=D:\DevProjects\PhotoAI\ComfyUI\.venv\Scripts\python.exe"

:: 2. Switch to the Project Root directory
:: This is critical so Python can find the 'da_apps' folder
cd /d "%PROJECT_ROOT%"

echo Starting LoRA Training as a Module...

:: 3. Run using the -m flag (module mode)
:: Note the dots (.) instead of slashes (\) and no .py extension
"%PYTHON%" -m da_apps.lora.lora_algo

echo.
echo ========================================================
echo Training finished or interrupted.
echo See above for errors.
echo ========================================================
pause