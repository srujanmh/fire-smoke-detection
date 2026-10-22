@echo off
title AegisVision AI — Fire & Smoke Detection System
cd /d "%~dp0"

:: Detect Python executable
set "PY_CMD="
py -3.11 --version >nul 2>&1 && set "PY_CMD=py -3.11"
if not defined PY_CMD (
    py --version >nul 2>&1 && set "PY_CMD=py"
)
if not defined PY_CMD (
    python --version >nul 2>&1 && set "PY_CMD=python"
)

if not defined PY_CMD (
    echo [ERROR] Python was not found on your system!
    echo Please install Python 3.10 or 3.11 and ensure it is added to your PATH.
    pause
    exit /b 1
)

:MENU
cls
echo ======================================================================
echo    AegisVision AI - Real-Time Fire & Smoke Detection System
echo ======================================================================
echo.
echo   Select an application interface to launch:
echo.
echo     [1] Modern Web Dashboard  (FastAPI - http://localhost:8000)
echo     [2] Native Desktop GUI    (CustomTkinter standalone window)
echo     [3] Direct OpenCV Window  (High-speed camera HUD)
echo     [4] Run Test Suite        (Pytest verification)
echo     [5] Exit
echo.
echo ======================================================================
set /p "CHOICE=Enter choice [1-5]: "

if "%CHOICE%"=="1" goto LAUNCH_WEB
if "%CHOICE%"=="2" goto LAUNCH_GUI
if "%CHOICE%"=="3" goto LAUNCH_OPENCV
if "%CHOICE%"=="4" goto LAUNCH_TESTS
if "%CHOICE%"=="5" goto EXIT_SCRIPT

echo.
echo [!] Invalid selection. Please choose 1, 2, 3, 4, or 5.
timeout /t 2 >nul
goto MENU

:LAUNCH_WEB
cls
echo ======================================================================
echo Starting AegisVision AI Web Dashboard on http://localhost:8000 ...
echo ======================================================================
echo Press CTRL+C in this terminal window to stop the server.
echo.
start "" http://localhost:8000
%PY_CMD% web_app.py
goto MENU

:LAUNCH_GUI
cls
echo ======================================================================
echo Launching Native Desktop GUI (CustomTkinter)...
echo ======================================================================
%PY_CMD% gui.py
goto MENU

:LAUNCH_OPENCV
cls
echo ======================================================================
echo Launching Direct OpenCV Window...
echo   Press [q] or [ESC] to quit
echo   Press [s] to save a manual incident snapshot
echo   Press [m] to toggle simulation mode
echo   Press [r] to reset the alert buffer
echo ======================================================================
%PY_CMD% main.py
goto MENU

:LAUNCH_TESTS
cls
echo ======================================================================
echo Running Automated Test Suite (Pytest)...
echo ======================================================================
%PY_CMD% -m pytest tests/ -v
echo.
pause
goto MENU

:EXIT_SCRIPT
echo Exiting. Goodbye!
exit /b 0
