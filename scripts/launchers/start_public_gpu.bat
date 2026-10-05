@echo off
setlocal enabledelayedexpansion
title OphthalmoAI - Public GPU Launcher
cd /d "%~dp0..\.."

echo ======================================================================
echo           OPHTHALMOAI - PUBLIC GPU LAUNCHER & CLOUDFLARE TUNNEL
echo ======================================================================

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
    set "PYTHON_EXE=python"
    echo [*] Selected System Python
)

"%PYTHON_EXE%" -u scripts\start_public_gpu.py %*
set "EXIT_CODE=%errorlevel%"

if %EXIT_CODE% neq 0 (
    echo.
    echo [!] Process exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
