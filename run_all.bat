@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo ==========================================
echo PhotoAI Launcher
echo ==========================================
echo 1) Run ComfyUI only
echo 2) Run PhotoAI FastAPI + Studio only
echo 3) Run BOTH (ComfyUI + FastAPI + Studio)
echo.

set /p CHOICE=Choose (1/2/3):

if "%CHOICE%"=="1" goto :COMFY
if "%CHOICE%"=="2" goto :DA
if "%CHOICE%"=="3" goto :BOTH

echo Invalid choice.
pause
exit /b 1

:COMFY
echo Starting ComfyUI...
start "ComfyUI" cmd /k "%~dp0run_comfyui.bat"
goto :END

:DA
echo Starting PhotoAI Studio + FastAPI...
start "PhotoAI FastAPI" cmd /k "%~dp0run_da_ps.bat"
echo Open:
echo   http://localhost:8000/api/ps/
goto :END

:BOTH
echo Starting ComfyUI...
start "ComfyUI" cmd /k "%~dp0run_comfyui.bat"

echo Starting PhotoAI Studio + FastAPI...
start "PhotoAI FastAPI" cmd /k "%~dp0run_da_ps.bat"

echo.
echo Open:
echo   http://localhost:8000/api/ps/
goto :END

:END
endlocal
