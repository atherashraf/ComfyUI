@echo off
setlocal EnableExtensions

echo.
echo [BAT] Script location: %~dp0
echo.

set /p DO_BUILD=Build PhotoAI Studio frontend now? (y/n):
echo.
set DO_BUILD=%DO_BUILD:~0,1%

if /I "%DO_BUILD%"=="y" goto :build
goto :skipbuild

:build
echo ==========================================
echo Building PhotoAI Studio (frontend)...
echo ==========================================

cd /d "%~dp0\da_apps\photoai-studio"
if errorlevel 1 goto :badpath

if not exist package.json goto :nopkg

if not exist node_modules (
  echo Installing frontend dependencies...
  npm install
  if errorlevel 1 goto :npmfail
)

npm run build
if errorlevel 1 goto :buildfail

goto :afterbuild

:skipbuild
echo Skipping frontend build.
goto :afterbuild

:afterbuild
echo.
echo ==========================================
echo Activating Python venv...
echo ==========================================

cd /d "%~dp0"
if errorlevel 1 goto :badpathroot

if not exist ".venv\Scripts\activate.bat" goto :novenvroot
call ".venv\Scripts\activate.bat"

echo [BAT] Using python:
where python
python --version

REM GPU & PyTorch settings
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8

if not defined OMP_NUM_THREADS set OMP_NUM_THREADS=8
if not defined MKL_NUM_THREADS set MKL_NUM_THREADS=8

echo.
echo ==========================================
echo Starting FastAPI (PhotoAI backend)...
echo ==========================================
echo Open PhotoAI Studio:
echo   http://localhost:8000/api/ps/
echo.

REM IMPORTANT: run from ComfyUI root so python -m da_apps.da_main resolves package
python -m da_apps.da_main %*

echo.
echo [BAT] FastAPI exited (errorlevel=%ERRORLEVEL%)
pause
exit /b %ERRORLEVEL%

:badpath
echo ERROR: Could not cd to %~dp0\da_apps\photoai-studio
pause
exit /b 1

:nopkg
echo ERROR: package.json not found in %cd%
pause
exit /b 1

:npmfail
echo ERROR: npm install failed
pause
exit /b 1

:buildfail
echo ERROR: npm run build failed
pause
exit /b 1

:badpathroot
echo ERROR: Could not cd to %~dp0
pause
exit /b 1

:novenvroot
echo ERROR: venv not found at %~dp0\.venv\Scripts\activate.bat
pause
exit /b 1
