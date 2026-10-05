@echo off
setlocal enabledelayedexpansion
title OphthalmoAI Launcher
cd /d "%~dp0..\.."

echo ======================================================================
echo           OPHTHALMOAI - POINT-OF-CARE RETINAL SCREENING
echo ======================================================================

:: 1. Locate best Python virtual environment
set "PYTHON_EXE="
if exist "%CD%\.venv_gpu\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv_gpu\Scripts\python.exe"
    echo [*] Selected GPU Runtime: %CD%\.venv_gpu
) else if exist "%CD%\venv_gpu\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\venv_gpu\Scripts\python.exe"
    echo [*] Selected GPU Runtime: %CD%\venv_gpu
) else if exist "%CD%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    echo [*] Selected CPU Runtime: %CD%\.venv
) else if exist "%CD%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
    echo [*] Selected CPU Runtime: %CD%\venv
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
        echo [*] Selected System Python
    ) else (
        echo [!] Error: Python not found. Please create a virtual environment:
        echo     python -m venv venv
        echo     pip install -r backend\requirements.txt
        echo.
        pause
        exit /b 1
    )
)

:: 2. Execute universal application launcher with any passed arguments
echo [*] Launching application services...
"%PYTHON_EXE%" -u scripts\start_app.py %*
set "EXIT_CODE=%errorlevel%"

if %EXIT_CODE% neq 0 (
    echo.
    echo [!] Process exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
