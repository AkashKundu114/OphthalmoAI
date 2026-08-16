Write-Host "Building Docker Image for GPU Training..." -ForegroundColor Cyan
docker build -t retinal-gpu-trainer -f Dockerfile.gpu .

Write-Host "Starting Docker Container (NVIDIA GPU Passthrough enabled)..." -ForegroundColor Cyan
# We mount the dataset folder so the container can access the images
# We also use --gpus all to ensure the container can see the RTX 5060
docker run --rm `
    --gpus all `
    --ipc=host `
    -e EPOCHS=20 `
    -v "${PWD}\dataset:/workspace/app/dataset" `
    retinal-gpu-trainer
