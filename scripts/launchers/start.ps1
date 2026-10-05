# start.ps1 - PowerShell entry point for OphthalmoAI
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = (Resolve-Path "$ScriptDir\..\..").Path
Set-Location $RootDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "         OPHTHALMOAI - POINT-OF-CARE RETINAL SCREENING" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Locate best Python interpreter
$PythonExe = if (Test-Path "$RootDir\.venv_gpu\Scripts\python.exe") {
    Write-Host "[*] Selected GPU Runtime: $RootDir\.venv_gpu" -ForegroundColor Green
    "$RootDir\.venv_gpu\Scripts\python.exe"
} elseif (Test-Path "$RootDir\venv_gpu\Scripts\python.exe") {
    Write-Host "[*] Selected GPU Runtime: $RootDir\venv_gpu" -ForegroundColor Green
    "$RootDir\venv_gpu\Scripts\python.exe"
} elseif (Test-Path "$RootDir\.venv\Scripts\python.exe") {
    Write-Host "[*] Selected CPU Runtime: $RootDir\.venv" -ForegroundColor Yellow
    "$RootDir\.venv\Scripts\python.exe"
} elseif (Test-Path "$RootDir\venv\Scripts\python.exe") {
    Write-Host "[*] Selected CPU Runtime: $RootDir\venv" -ForegroundColor Yellow
    "$RootDir\venv\Scripts\python.exe"
} else {
    $sysPy = Get-Command python -ErrorAction SilentlyContinue
    if ($sysPy) {
        Write-Host "[*] Selected System Python" -ForegroundColor Yellow
        "python"
    } else {
        Write-Host "[!] Error: No Python executable found. Run 'python -m venv venv' first." -ForegroundColor Red
        exit 1
    }
}

& $PythonExe -u "$RootDir\scripts\start_app.py" @args
exit $LASTEXITCODE
