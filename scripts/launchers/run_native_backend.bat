@echo off
setlocal enabledelayedexpansion
title OphthalmoAI - FastAPI Backend
cd /d "%~dp0..\.."

echo ======================================================================
echo           OPHTHALMOAI - FASTAPI BACKEND SERVER
echo ======================================================================

if not exist "logs" mkdir "logs"

set "VENV_ACTIVATE="
if exist "%CD%\.venv_gpu\Scripts\activate.bat" (
    set "VENV_ACTIVATE=%CD%\.venv_gpu\Scripts\activate.bat"
    echo [*] Selected GPU Runtime: %CD%\.venv_gpu
) else if exist "%CD%\venv_gpu\Scripts\activate.bat" (
    set "VENV_ACTIVATE=%CD%\venv_gpu\Scripts\activate.bat"
    echo [*] Selected GPU Runtime: %CD%\venv_gpu
) else if exist "%CD%\.venv\Scripts\activate.bat" (
    set "VENV_ACTIVATE=%CD%\.venv\Scripts\activate.bat"
    echo [*] Selected CPU Runtime: %CD%\.venv
) else if exist "%CD%\venv\Scripts\activate.bat" (
    set "VENV_ACTIVATE=%CD%\venv\Scripts\activate.bat"
    echo [*] Selected CPU Runtime: %CD%\venv
)

if defined VENV_ACTIVATE (
    call "!VENV_ACTIVATE!"
)

echo [*] Starting Uvicorn on http://127.0.0.1:8000 ...
echo [*] Interactive API Docs: http://127.0.0.1:8000/docs
echo [*] Press CTRL+C to stop the server.
echo ======================================================================
echo.

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 %*

