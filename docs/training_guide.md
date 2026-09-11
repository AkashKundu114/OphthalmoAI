# OphthalmoAI Retinal Fundus Deep Learning Training & Hardware Telemetry Guide

## 1. System Architecture & Hardware Target Profile
This training system has been calibrated for the user's workstation:
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR6 Dedicated VRAM, Tensor Cores, cuDNN acceleration, 100.85W TGP).
- **CPU**: AMD Ryzen 9 HX Multi-Core Processor (16 Physical Cores / 32 Threads, multi-threaded PyTorch BLAS).
- **Memory**: System RAM with dynamic virtual pagefile tracking.
- **Dataset**: Standardized Posterior Pole Retinal Fundus Photography (6 Classes, 6,248 images, Ben Graham circular illumination subtraction at 384×384).

---

## 2. Exhaustive Telemetry Specification: What We Track
The telemetry engine (`scripts/metric_logger.py`) records metrics per epoch to both console logs and persistent JSON files (`dataset/logs/telemetry_<model>_<timestamp>.json`):

### A. CPU & Thread Telemetry (AMD Ryzen 9 HX)
- **Overall CPU Utilization (`%`)**: Total system-wide processor load.
- **Per-Core Utilization (`%`)**: Array of utilization percentages across all **32 logical cores**.
- **CPU Core Frequency (`MHz`)**: Real-time operating clock speed.
- **CPU Temperature (`°C`)**: Polled via hardware thermal sensors (`coretemp`, `k10temp`, `zenpower`, ACPI).
- **Core Allocation Topology**: Physical core count (16) vs Logical core count (32).

### B. System RAM & Virtual Memory Telemetry
- **Process RSS RAM (`GB`)**: Actual physical resident RAM consumed by the training process.
- **Process VMS (`GB`)**: Virtual address space allocated.
- **Total System RAM (`GB`)**: Total installed physical memory (15.21 GB).
- **System RAM Used & Free (`GB`)**: Real-time memory pressure.
- **System RAM Utilization (`%`)**: Percent of system RAM in active use.
- **Swap / Pagefile Usage (`GB` & `%`)**: Windows paging file activity to prevent out-of-memory memory spikes.

### C. GPU Core & Thermal Telemetry (NVIDIA GeForce RTX 5060 Laptop GPU)
- **GPU Compute Utilization (`%`)**: Streaming Multiprocessor (SM) active execution time.
- **GPU Memory Controller Utilization (`%`)**: Memory bus bandwidth saturation.
- **GPU Core Temperature (`°C`)**: Real-time sensor reading from the RTX 5060 silicon.
- **Thermal Slowdown Threshold (`102°C`)**: Hardware thermal throttling guardrail.
- **Thermal Shutdown Threshold (`105°C`)**: Emergency silicon safety shutdown limit.
- **GPU Power Consumption (`Watts`)**: Live power draw.
- **GPU Enforced Power Limit (`Watts`)**: Target TGP limit (e.g. 92.35W - 100.85W).
- **Graphics Core Clock (`MHz`)**: Live clock speed of graphics cores.
- **Memory Clock (`MHz`)**: Live GDDR6 clock speed.
- **SM (Streaming Multiprocessor) Clock (`MHz`)**: Execution engine clock speed.
- **PCIe Bus Throughput (`MB/s`)**: Live TX (transmit) and RX (receive) bandwidth over the PCIe lane.

### D. Dedicated 8GB VRAM Telemetry
- **PyTorch Allocated VRAM (`MB` & `GB`)**: Exact memory held by active PyTorch tensors.
- **PyTorch Reserved VRAM (`MB` & `GB`)**: Total memory cached by PyTorch allocator.
- **Total Dedicated VRAM (`8,151 MB`)**: Total 8GB hardware limit.
- **NVML Total System VRAM Used (`MB`)**: Total VRAM used by PyTorch + desktop compositor.
- **VRAM Utilization (`%`)**: Ratio of VRAM consumption against the 8,151 MB hardware ceiling.
- **VRAM Headroom (`MB`)**: Exact remaining memory before an Out-Of-Memory (`CUDA OOM`) error.

### E. Model Training Dynamics & Performance
- **Epoch Duration (`seconds`)**: Total wall-clock time per epoch.
- **Training Throughput (`samples/sec`)**: Number of fundus images processed per second.
- **Training Loss & Accuracy (`%`)**: Cross-entropy loss with label smoothing (0.05).
- **Validation Loss & Accuracy (`%`)**: Metric evaluated on held-out validation split.
- **Validation Macro F1 & Weighted F1**: Balanced multi-class diagnostic performance.
- **Learning Rate (`lr`)**: Current learning rate from Cosine Annealing scheduler.
- **GradScaler Scale Factor**: Dynamic FP16 mixed-precision scale factor (tracks gradient stability).

---

## 3. Retinal Disease Classification Classes (6 Classes)
1. `Normal` (ICD-10: `Z01.00`, SNOMED: `17621005`)
2. `Diabetic Retinopathy` (ICD-10: `E11.319`, SNOMED: `4855003`)
3. `Glaucoma` (ICD-10: `H40.9`, SNOMED: `23986001`)
4. `Cataract` (ICD-10: `H25.9`, SNOMED: `193570009`)
5. `Age-related Macular Degeneration (AMD)` (ICD-10: `H35.30`, SNOMED: `267718000`)
6. `Hypertensive Retinopathy / Pathological Myopia` (ICD-10: `H35.03 / H44.2`, SNOMED: `38822007`)

---

## 4. Script Catalog & Dedicated File Reference

| Script Filename | Architecture / Function | Target Device | Precision Modes | Recommended Batch Size (8GB VRAM) |
| :--- | :--- | :---: | :---: | :---: |
| [`scripts/train_model.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_model.py) | **Universal Trainer**: Any model CLI | `cuda` / `cpu` | `fp16`, `bf16`, `fp32` | 32 (GPU) / 16 (CPU) |
| [`scripts/train_ensemble.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_ensemble.py) | **Meta-Ensemble**: ConvNeXt + DenseNet + EfficientNet | `cuda` / `cpu` | `fp16`, `bf16`, `fp32` | 32 |
| [`scripts/train_evidential_meta.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_evidential_meta.py) | **Evidential AC-HDL**: Dirichlet Uncertainty | `cuda` / `cpu` | `fp32` | 32 |
| [`scripts/train_efficientnet_b4.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_efficientnet_b4.py) | **Production Target**: EfficientNet-B4 (Grad-CAM) | `cuda` / `cpu` | `fp16`, `bf16`, `fp32` | 32 |
| [`scripts/train_convnext_small.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_convnext_small.py) | **ConvNeXt-Small**: 7x7 Depthwise ConvNet | `cuda` / `cpu` | `fp16`, `bf16`, `fp32` | 16 (Safe) / 32 |
| [`scripts/train_densenet201.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_densenet201.py) | **DenseNet-201**: Feature Reuse Network | `cuda` / `cpu` | `fp16`, `bf16`, `fp32` | 16 (Prevents paging) |
| [`scripts/train_resnet50.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_resnet50.py) | **ResNet-50**: Universal Baseline | `cuda` / `cpu` | `fp16`, `bf16`, `fp32` | 16 / 32 |
| [`scripts/train_cpu_resnet50.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_cpu_resnet50.py) | **CPU ResNet-50**: Ryzen 9 HX (32 threads) | `cpu` only | `fp32` | 16 |
| [`scripts/train_ensemble_bf16.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_ensemble_bf16.py) | **BF16 Meta-Ensemble**: Tensor Core bfloat16 | `cuda` / `cpu` | `bf16` | 16 / 32 |
| [`scripts/train_ensemble_bs32.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_ensemble_bs32.py) | **BS32 Meta-Ensemble**: 8GB VRAM Optimized | `cuda` / `cpu` | `fp16` | 16 / 32 |
| [`scripts/run_all_trainings.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/run_all_trainings.py) | **Master Orchestrator**: Automated sweeps & smoke tests | Any | Any | Configurable (Default: 16) |
| [`scripts/run_conformal_calib.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/run_conformal_calib.py) | **Conformal Calibration**: 99% emergency coverage | `cuda` / `cpu` | `fp32` | 16 / 32 |
| [`scripts/calibrate_models.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/calibrate_models.py) | **Temperature Scaling**: Platt ECE Minimizer | `cuda` / `cpu` | `fp32` | 16 / 32 |
| [`scripts/evaluate_models.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/evaluate_models.py) | **Evaluation Suite**: AUROC, ECE, Confusion Matrix | `cuda` / `cpu` | `fp32` | 16 / 32 |
| [`scripts/evaluate_ensemble.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/evaluate_ensemble.py) | **Ensemble Evaluator**: Tri-Backbone Test Report | `cuda` / `cpu` | `fp32` | 16 / 32 |

---

## 5. Copy-Paste Execution Commands (PowerShell / Windows)

### A. Universal Multi-Model Trainer (`train_model.py`)
```powershell
# 1. Train Production Target (EfficientNet-B4) with FP16 on GPU (Batch Size 16, 10 Epochs)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_model.py --model efficientnet_b4 --precision fp16 --batch-size 16 --epochs 10 --device cuda

# 2. Train ConvNeXt-Small with BF16 on GPU (Batch Size 16, 10 Epochs)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_model.py --model convnext_small --precision bf16 --batch-size 16 --epochs 10 --device cuda

# 3. Train DenseNet-201 with FP16 on GPU (Batch Size 16, 10 Epochs)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_model.py --model densenet201 --precision fp16 --batch-size 16 --epochs 10 --device cuda

# 4. Train ResNet-50 with FP32 on CPU (32 Ryzen Threads, Batch Size 16, 5 Epochs)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_model.py --model resnet50 --precision fp32 --batch-size 16 --epochs 5 --device cpu
```

---

### B. Meta-Ensemble Training
```powershell
# 1. Meta-Ensemble with FP16 (ConvNeXt + DenseNet + EfficientNet-V2)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_ensemble.py --precision fp16 --batch-size 16 --epochs 12 --device cuda

# 2. Meta-Ensemble with Native BF16 (No GradScaler needed)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_ensemble_bf16.py --batch-size 16 --epochs 12 --device cuda

# 3. Evidential Meta-Classifier (AC-HDL Cost-Sensitive Loss & Dirichlet Vacuity)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_evidential_meta.py --epochs 15 --batch-size 16 --device cuda
```

---

### C. Dedicated Architecture Scripts
```powershell
# EfficientNet-B4 Dedicated Script (Recommended Safe Batch Size: 16)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_efficientnet_b4.py --precision fp16 --batch-size 16 --epochs 10 --device cuda

# ConvNeXt-Small Dedicated Script
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_convnext_small.py --precision fp16 --batch-size 16 --epochs 10 --device cuda

# DenseNet-201 Dedicated Script
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_densenet201.py --precision fp16 --batch-size 16 --epochs 10 --device cuda

# ResNet-50 Dedicated Script (GPU)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_resnet50.py --precision fp16 --batch-size 16 --epochs 10 --device cuda

# CPU-Optimized ResNet-50 (Ryzen 9 HX 32 Threads)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\train_cpu_resnet50.py --batch-size 16 --epochs 5 --threads 32
```

---

### D. Automated Sweeps & Benchmarking Suite
```powershell
# Quick Smoke Test (1 epoch test on CPU & GPU)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\run_all_trainings.py --mode smoke

# Precision Comparison Sweep (FP32 vs FP16 vs BF16)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\run_all_trainings.py --mode precision_sweep --model efficientnet_b4 --epochs 3 --batch-size 16

# Batch Size Sweep (16 vs 32 vs 64)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\run_all_trainings.py --mode batch_sweep --model efficientnet_b4 --precision fp16 --epochs 3

# Full Architecture Comparison Sweep
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\run_all_trainings.py --mode model_sweep --precision fp16 --batch-size 16 --epochs 3
```

---

### E. Calibration & Final Evaluation
```powershell
# Conformal Risk Control Calibration (99% Emergency / 95% Routine)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\run_conformal_calib.py

# Temperature Scaling Calibration (Minimizing ECE)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\calibrate_models.py --model efficientnet_b4

# Full Test Set Evaluation (AUROC, Confusion Matrix, Sensitivity/Specificity)
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\evaluate_models.py --model efficientnet_b4 --device cuda

# Ensemble Test Set Evaluation
& "d:\AI-based Retinal Disease Predictor\venv_gpu\Scripts\python.exe" scripts\evaluate_ensemble.py --device cuda
```
