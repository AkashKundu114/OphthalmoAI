# OphthalmoAI: Full Multi-Model 20-Epoch GPU Training Suite via Docker
# ======================================================================
# Sequentially trains each constituent backbone and the meta-ensemble on the
# NVIDIA RTX 5060 GPU with CUDA acceleration and FP16 mixed precision:
#   1. ConvNeXt-Small      (20 Epochs, GPU/FP16) -> models/convnext_small.pth
#   2. DenseNet-201        (20 Epochs, GPU/FP16) -> models/densenet201.pth
#   3. EfficientNet-V2-M   (20 Epochs, GPU/FP16) -> models/efficientnet_v2_m.pth
#   4. RetinalMetaEnsemble (20 Epochs, GPU/FP16) -> models/meta_classifier.pth
#   5. Conformal AW-CRC Recalibration & Extended Clinical Battery
#   6. 18 Publication Figures Generation (300+ DPI, RGB)
# ======================================================================

param(
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [string]$Precision = "fp16",
    [int]$WaitSeconds = 300
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = (Resolve-Path "$ScriptDir\..\..").Path
Set-Location $RootDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " OPHTHALMOAI: 20-EPOCH DOCKER GPU TRAINING ORCHESTRATION" -ForegroundColor Cyan
Write-Host " Target Hardware: NVIDIA GeForce RTX 5060 (8GB GDDR7)" -ForegroundColor Cyan
Write-Host " Epochs per Model: $Epochs | Batch Size: $BatchSize | Precision: $Precision | Cooldown: ${WaitSeconds}s" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Step 1: Build Docker GPU Image
Write-Host "`n[PHASE 0] Building/Verifying Docker GPU Container Image..." -ForegroundColor Yellow
docker build -t ophthalmoai-gpu-trainer -f Dockerfile.gpu .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker build failed. Exiting." -ForegroundColor Red
    exit 1
}

$Models = @("resnet50", "efficientnet_b4", "convnext_small", "densenet201", "efficientnet_v2_m")

# Step 2: Train Individual Backbones sequentially
$StepIndex = 1
$TotalSteps = $Models.Length + 2
foreach ($m in $Models) {
    Write-Host "`n----------------------------------------------------------------------" -ForegroundColor Green
    Write-Host "[PHASE $StepIndex/$TotalSteps] Training Backbone on GPU: $m ($Epochs Epochs, BS=$BatchSize, $Precision)" -ForegroundColor Green
    Write-Host "----------------------------------------------------------------------" -ForegroundColor Green

    docker run --rm `
        --gpus all `
        --ipc=host `
        -v "${RootDir}\dataset:/workspace/app/dataset" `
        -v "${RootDir}\models:/workspace/app/models" `
        -v "${RootDir}\scripts:/workspace/app/scripts" `
        -v "$env:USERPROFILE\.cache\torch:/root/.cache/torch" `
        ophthalmoai-gpu-trainer `
        python scripts/train_model.py `
            --model $m `
            --precision $Precision `
            --batch-size $BatchSize `
            --epochs $Epochs `
            --device cuda

    $StepIndex++

    if ($StepIndex -le $Models.Length) {
        Write-Host "`n[COOLDOWN] Pausing for $($WaitSeconds / 60) minutes (${WaitSeconds}s) for GPU thermal stabilization..." -ForegroundColor Cyan
        Start-Sleep -Seconds $WaitSeconds
    }
}

# Cooldown before Meta-Ensemble
Write-Host "`n[COOLDOWN] Pausing for $($WaitSeconds / 60) minutes (${WaitSeconds}s) before Meta-Ensemble..." -ForegroundColor Cyan
Start-Sleep -Seconds $WaitSeconds

# Step 3: Train Retinal Meta-Ensemble
Write-Host "`n----------------------------------------------------------------------" -ForegroundColor Green
Write-Host "[PHASE 4/5] Training Retinal Meta-Ensemble on GPU ($Epochs Epochs, BS=$BatchSize, $Precision)" -ForegroundColor Green
Write-Host "----------------------------------------------------------------------" -ForegroundColor Green

docker run --rm `
    --gpus all `
    --ipc=host `
    -v "${RootDir}\dataset:/workspace/app/dataset" `
    -v "${RootDir}\models:/workspace/app/models" `
    -v "${RootDir}\scripts:/workspace/app/scripts" `
    -v "$env:USERPROFILE\.cache\torch:/root/.cache/torch" `
    ophthalmoai-gpu-trainer `
    python scripts/train_ensemble.py `
        --precision $Precision `
        --batch-size $BatchSize `
        --epochs $Epochs `
        --device cuda

# Cooldown before calibration and evaluation
Write-Host "`n[COOLDOWN] Pausing for $($WaitSeconds / 60) minutes (${WaitSeconds}s) before Clinical Battery..." -ForegroundColor Cyan
Start-Sleep -Seconds $WaitSeconds

# Step 4: Run AW-CRC Conformal Calibration & Clinical Battery
Write-Host "`n----------------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "[PHASE 5/5] Executing Calibration, Extended Clinical Battery & Figure Suite..." -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor Yellow

docker run --rm `
    --gpus all `
    --ipc=host `
    -v "${RootDir}\dataset:/workspace/app/dataset" `
    -v "${RootDir}\models:/workspace/app/models" `
    -v "${RootDir}\scripts:/workspace/app/scripts" `
    -v "${RootDir}\research:/workspace/app/research" `
    ophthalmoai-gpu-trainer `
    python scripts/run_aw_crc_calibration.py

docker run --rm `
    --gpus all `
    --ipc=host `
    -v "${RootDir}\dataset:/workspace/app/dataset" `
    -v "${RootDir}\models:/workspace/app/models" `
    -v "${RootDir}\scripts:/workspace/app/scripts" `
    -v "${RootDir}\research:/workspace/app/research" `
    ophthalmoai-gpu-trainer `
    python scripts/evaluate_extended_clinical_battery.py

# Regenerate all 18 figures locally
if (Test-Path ".\venv\Scripts\python.exe") {
    .\venv\Scripts\python.exe scripts/generate_all_manuscript_figures.py
}

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " [SUCCESS] All 20-Epoch GPU Trainings, Benchmarks & Figures Completed!" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
