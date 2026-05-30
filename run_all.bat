@echo off
setlocal EnableExtensions

REM ---- Auto-elevate to administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator privileges...
    powershell start -verb runas '%0' %*
    exit /b
)

cd /d "%~dp0"

echo ==========================================
echo PhotoAI Launcher (Administrator Mode)
echo ==========================================
echo.

REM ---- Check page file size
call :CHECK_PAGEFILE

echo ==========================================
echo Main Menu
echo ==========================================
echo 1) Run ComfyUI only
echo 2) Run PhotoAI FastAPI + Studio only
echo 3) Run BOTH (ComfyUI + FastAPI + Studio)
echo 4) Setup Page File (increase virtual memory)
echo.

set /p CHOICE=Choose (1/2/3/4):

if "%CHOICE%"=="1" goto :COMFY
if "%CHOICE%"=="2" goto :DA
if "%CHOICE%"=="3" goto :BOTH
if "%CHOICE%"=="4" goto :PAGEFILE

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

:PAGEFILE
echo Running Page File Setup...
if exist "%~dp0setup_pagefile.bat" (
    call "%~dp0setup_pagefile.bat"
) else (
    echo ❌ setup_pagefile.bat not found!
    echo Creating a basic page file setup...
    call :CREATE_PAGEFILE_SCRIPT
    call "%~dp0setup_pagefile.bat"
)
goto :END

:CHECK_PAGEFILE
for /f "skip=1 tokens=2 delims==" %%a in ('wmic pagefileset get MaximumSize /value 2^>nul') do set PAGEFILE_MAX=%%a

if "%PAGEFILE_MAX%"=="" (
    echo ⚠️  Using automatically managed page file
    echo    Recommended minimum: 32768 MB (32GB)
    echo.
) else if %PAGEFILE_MAX% LSS 32768 (
    echo ⚠️  WARNING: Page file is only %PAGEFILE_MAX% MB
    echo    Recommended: 32768 MB (32GB) minimum for AI models
    echo    Select option 4 to fix this
    echo.
    timeout /t 3 /nobreak >nul
) else (
    echo ✅ Page file is adequate: %PAGEFILE_MAX% MB
    echo.
)
goto :EOF

:CREATE_PAGEFILE_SCRIPT
echo Creating setup_pagefile.bat...
(
echo @echo off
echo title Page File Setup for PhotoAI
echo.
echo echo ==========================================
echo echo  Setting Up Page File for AI Workloads
echo echo ==========================================
echo echo.
echo.
echo REM ---- Disable automatic management
echo wmic computersystem where name="%%computername%%" set AutomaticManagedPagefile=False
echo.
echo REM ---- Remove existing custom page files
echo for /f "skip=1" %%%%a in ('wmic pagefileset get name ^| findstr /i ":"') do ^(
echo     wmic pagefileset where "name='%%%%a'" delete
echo ^)
echo.
echo REM ---- Create new page file on C: drive
echo wmic pagefileset create name="C:\\pagefile.sys",InitialSize=32768,MaximumSize=65536
echo.
echo if %%errorlevel%% equ 0 ^(
echo     echo.
echo     echo ✅ Page file successfully set to 32GB - 64GB
echo     echo.
echo     echo ==========================================
echo     echo  RESTART YOUR COMPUTER NOW
echo     echo ==========================================
echo     echo.
echo     choice /C RN /M "Restart now (R) or restart later (N)"
echo     if errorlevel 2 goto :END
echo     if errorlevel 1 shutdown /r /t 10 /c "Restarting to apply page file changes..."
echo ^) else ^(
echo     echo.
echo     echo ❌ Failed to configure page file
echo ^)
echo.
echo :END
echo pause
) > "%~dp0setup_pagefile.bat"
goto :EOF

:END
endlocal