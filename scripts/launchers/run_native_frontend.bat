@echo off
setlocal enabledelayedexpansion
title OphthalmoAI - Vite React Frontend
cd /d "%~dp0..\..\frontend"

echo ======================================================================
echo           OPHTHALMOAI - VITE REACT FRONTEND
echo ======================================================================

if not exist "..\logs" mkdir "..\logs"

echo [*] Starting Vite Development Server on http://127.0.0.1:5176 ...
echo [*] Press CTRL+C to stop the server.
echo ======================================================================
echo.

set CI=false
call npm.cmd run dev -- --port 5176 --host 127.0.0.1 %*

