@echo off
title OphthalmoAI - Public GPU Launcher
cd /d "%~dp0"
echo ===================================================
echo Starting OphthalmoAI Public GPU Server...
echo ===================================================
venv_gpu\Scripts\python.exe -u scripts\start_public_gpu.py
pause
