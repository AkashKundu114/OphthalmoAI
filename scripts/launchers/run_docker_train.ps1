$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = (Resolve-Path "$ScriptDir\..\..").Path
Set-Location $RootDir

Write-Host "Building Docker Image for GPU Training..." -ForegroundColor Cyan
docker build -t retinal-gpu-trainer -f Dockerfile.gpu .

Write-Host "Starting Docker Container (NVIDIA GPU Passthrough enabled)..." -ForegroundColor Cyan
docker run --rm `
    --gpus all `
    --ipc=host `
    -e EPOCHS=20 `
    -v "${RootDir}\dataset:/workspace/app/dataset" `
    retinal-gpu-trainer
