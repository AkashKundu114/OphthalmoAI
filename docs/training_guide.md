# OphthalmoAI Retinal Fundus Deep Learning Training & Benchmarking Guide

## Executive Overview
This guide documents the modular training infrastructure for **OphthalmoAI**. The training pipeline supports training across:
- **Neural Architectures**: ConvNeXt-Small, DenseNet-201, EfficientNet-V2-M, EfficientNet-B4 (Production Target), ResNet-50, and Tri-Backbone Meta-Ensembles.
- **Precision Formats**: Single Precision (`FP32`), Automatic Mixed Precision (`FP16` with dynamic loss scaling), and Brain Floating Point (`BF16` for high dynamic range without scaling).
- **Batch Sizes**: Mini-batches of `16`, `32`, `64` tuned for 8GB VRAM targets (NVIDIA RTX 5060 Laptop GPU).
- **Compute Backends**: Multi-threaded Intel/AMD CPUs and CUDA 12.8 GPU acceleration with cuDNN auto-tuning.

---

## 1. Retinal Disease Classification Taxonomy (6 Classes)
All scripts are synchronized to classify fundus photography into the following 6 posterior pole conditions:
1. **Normal** (`ICD-10: Z01.00`, `SNOMED: 17621005`)
2. **Diabetic Retinopathy** (`ICD-10: E11.319`, `SNOMED: 4855003`)
3. **Glaucoma** (`ICD-10: H40.9`, `SNOMED: 23986001`)
4. **Cataract** (`ICD-10: H25.9`, `SNOMED: 193570009`)
5. **Age-related Macular Degeneration (AMD)** (`ICD-10: H35.30`, `SNOMED: 267718000`)
6. **Hypertensive Retinopathy / Pathological Myopia** (`ICD-10: H35.03 / H44.2`, `SNOMED: 38822007`)

Dataset manifests are located at:
- Training: `dataset/processed/train.csv` (4,373 fundus images)
- Validation: `dataset/processed/val.csv` (937 fundus images)
- Held-Out Test: `dataset/processed/test.csv` (938 fundus images)

---

## 2. Script Matrix & CLI Reference

| Script | Target Architecture / Purpose | Default Precision | Default Batch Size | Compute |
| :--- | :--- | :---: | :---: | :---: |
| [`scripts/train_model.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_model.py) | **Universal Trainer**: Any backbone | Configurable (`fp16`/`bf16`/`fp32`) | Configurable (16, 32, 64) | `cuda` / `cpu` |
| [`scripts/train_ensemble.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_ensemble.py) | **Meta-Ensemble**: ConvNeXt + DenseNet + EfficientNet | Configurable (`fp16`/`bf16`/`fp32`) | 32 | `cuda` / `cpu` |
| [`scripts/train_evidential_meta.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_evidential_meta.py) | **Evidential AC-HDL**: Dirichlet Uncertainty & Clinical Cost | FP32 | 32 | `cuda` / `cpu` |
| [`scripts/train_efficientnet_b4.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_efficientnet_b4.py) | **Production Target**: EfficientNet-B4 + Grad-CAM | FP16 | 32 | `cuda` / `cpu` |
| [`scripts/train_resnet50.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_resnet50.py) | ResNet50 Classifier | FP16 | 32 | `cuda` / `cpu` |
| [`scripts/train_cpu_resnet.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_cpu_resnet.py) | CPU-Optimized ResNet50 (Multi-threaded) | FP32 | 16 | CPU only |
| [`scripts/train_efficientnet_v2_s.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_efficientnet_v2_s.py) | EfficientNet-V2-S / EfficientNet-V2-M | FP16 | 32 | `cuda` / `cpu` |
| [`scripts/train_ensemble_bf16.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_ensemble_bf16.py) | Bfloat16 Meta-Ensemble (Tensor Cores) | BF16 | 32 | `cuda` / `cpu` |
| [`scripts/train_ensemble_bs32.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/train_ensemble_bs32.py) | 8GB VRAM Tuned Meta-Ensemble (Batch Size 32) | FP16 | 32 | `cuda` / `cpu` |
| [`scripts/run_conformal_calib.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/run_conformal_calib.py) | Distribution-free Conformal Risk Control Cutoffs | FP32 | 32 | `cuda` / `cpu` |
| [`scripts/evaluate_models.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/evaluate_models.py) | Test Set Evaluation (AUROC, ECE, Confusion Matrix) | FP32 | 32 | `cuda` / `cpu` |
| [`scripts/evaluate_ensemble.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/evaluate_ensemble.py) | Meta-Ensemble Test Set Evaluation | FP32 | 32 | `cuda` / `cpu` |
| [`scripts/run_all_trainings.py`](file:///d:/AI-based%20Retinal%20Disease%20Predictor/scripts/run_all_trainings.py) | Master Orchestration & Benchmark Sweep Suite | Any | Any | Any |

---

## 3. Quick-Start Commands (PowerShell / Windows)

### A. Universal Trainer (`train_model.py`)
Run any model with custom precision, batch size, and device:

```powershell
# EfficientNet-B4 with FP16 on GPU (Production Baseline)
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_model.py --model efficientnet_b4 --precision fp16 --batch-size 32 --epochs 10 --device cuda

# ConvNeXt-Small with BF16 on GPU
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_model.py --model convnext_small --precision bf16 --batch-size 32 --epochs 10 --device cuda

# DenseNet-201 with FP16 on GPU
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_model.py --model densenet201 --precision fp16 --batch-size 32 --epochs 10 --device cuda

# ResNet-50 with FP32 on CPU (Baseline Benchmarking)
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_model.py --model resnet50 --precision fp32 --batch-size 16 --epochs 5 --device cpu
```

---

### B. Meta-Ensemble Training
Train the triple-backbone meta-classifier combining ConvNeXt + DenseNet + EfficientNet:

```powershell
# Meta-Ensemble with FP16 (Batch Size 32)
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_ensemble.py --precision fp16 --batch-size 32 --epochs 15 --device cuda

# Meta-Ensemble with Native BF16
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_ensemble_bf16.py --batch-size 32 --epochs 15 --device cuda

# Evidential Meta-Classifier (AC-HDL Cost-Sensitive Loss)
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_evidential_meta.py --epochs 15 --batch-size 32 --device cuda
```

---

### C. Dedicated Architecture Scripts
```powershell
# Train EfficientNet-B4 directly
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_efficientnet_b4.py --precision fp16 --batch-size 32 --epochs 10 --device cuda

# Train ResNet-50 on GPU
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_resnet50.py --precision fp16 --batch-size 32 --epochs 10 --device cuda

# Train CPU-optimized ResNet-50
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_cpu_resnet.py --batch-size 16 --epochs 5 --threads 8

# Train EfficientNet-V2-S
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\train_efficientnet_v2_s.py --variant s --precision fp16 --batch-size 32 --epochs 10 --device cuda
```

---

### D. Automated Benchmarks & Sweeps (`run_all_trainings.py`)
```powershell
# 1. Quick Smoke Test across CPU and GPU
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\run_all_trainings.py --mode smoke

# 2. Precision Comparison Sweep (FP32 vs FP16 vs BF16) on EfficientNet-B4
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\run_all_trainings.py --mode precision_sweep --model efficientnet_b4 --epochs 3

# 3. Mini-Batch Sweep (16 vs 32 vs 64)
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\run_all_trainings.py --mode batch_sweep --model efficientnet_b4 --precision fp16 --epochs 3

# 4. Multi-Architecture Sweep
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\run_all_trainings.py --mode model_sweep --precision fp16 --batch-size 32 --epochs 3
```

---

### E. Conformal Calibration & Model Evaluation
```powershell
# Calculate Conformal Risk Control cutoffs (99% emergency coverage)
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\run_conformal_calib.py

# Platt / Temperature Scaling Calibration
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\calibrate_models.py --model efficientnet_b4

# Full Test Set Evaluation (AUROC, ECE, Specificity, Sensitivity)
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\evaluate_models.py --model efficientnet_b4 --device cuda

# Meta-Ensemble Test Set Evaluation
& "d:\AI-based Retinal Disease Predictor\venv\Scripts\python.exe" scripts\evaluate_ensemble.py --device cuda
```

---

## 4. Hardware Telemetry & Conformal Guarantees
During every epoch, telemetry metrics are logged to `dataset/logs/telemetry_<model>_<timestamp>.json`:
- **GPU Metrics**: Peak VRAM allocated, VRAM reserved, temperature, utilization %
- **CPU Metrics**: Process RSS memory, system RAM %, multi-core utilization
- **Throughput**: Samples/second and seconds per epoch
- **Calibration**: Expected Calibration Error (ECE) and Conformal quantile bounds $\hat{q}$
