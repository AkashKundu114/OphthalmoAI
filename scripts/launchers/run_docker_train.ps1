# OphthalmoAI: Docker GPU Training Launcher for NVIDIA RTX 5060 (Blackwell)
param(
    [string]$Model = "convnext_small",
    [int]$Epochs = 5,
    [int]$BatchSize = 32,
    [string]$Precision = "fp16"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = (Resolve-Path "$ScriptDir\..\..").Path
Set-Location $RootDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " OPHTHALMOAI: DOCKER GPU TRAINING PIPELINE (NVIDIA CUDA PASSTHROUGH)" -ForegroundColor Cyan
Write-Host " Target Architecture: $Model | Batch Size: $BatchSize | Precision: $Precision" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "`n[STEP 1/2] Building Docker GPU Training Image..." -ForegroundColor Yellow
docker build -t ophthalmoai-gpu-trainer -f Dockerfile.gpu .

Write-Host "`n[STEP 2/2] Launching Container with NVIDIA GPU Passthrough..." -ForegroundColor Green
docker run --rm `
    --gpus all `
    --ipc=host `
    -v "${RootDir}\dataset:/workspace/app/dataset" `
    -v "${RootDir}\models:/workspace/app/models" `
    ophthalmoai-gpu-trainer `
    python scripts/train_model.py `
        --model $Model `
        --precision $Precision `
        --batch-size $BatchSize `
        --epochs $Epochs `
        --device cuda
