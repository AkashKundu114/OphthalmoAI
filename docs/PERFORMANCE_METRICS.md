# Performance & Architecture Metrics

This document details the engineering performance benchmarks of the **AI-Based Retinal Disease Predictor**, contrasting the training speeds, efficiency, and resource utilization across CPU, Bare-Metal GPU, and Dockerized GPU execution environments.

## 1. Executive Summary

Transitioning from CPU-only training to a hardware-accelerated pipeline on an RTX 5060 yielded a massive **~9x speedup** for ResNet50 (from 451.7s per epoch to 55.0s per epoch). 

However, the most significant breakthrough came from containerizing the **EfficientNet-B4** model using the highly optimized NVIDIA PyTorch Docker image (`nvcr.io/nvidia/pytorch:26.07-py3`). Instead of a container overhead, Dockerized execution actually *outperformed* bare-metal Windows execution, accelerating EfficientNet-B4 from 38.4s/epoch to just **23.5s/epoch** while hitting a staggering **98.61% Accuracy**.

## 2. Training Time Benchmarks

| Environment | Model | Time per Epoch | Peak VRAM | Final Accuracy |
|-------------|-------|----------------|-----------|----------------|
| **Ryzen 9 8940HX (Bare-Metal)** | ResNet50 | `451.7s` | N/A | 81.61% |
| **RTX 5060 (Bare-Metal)** | ResNet50 | `55.0s` | 1.77 GB | 92.96% |
| **RTX 5060 (Bare-Metal)** | EfficientNet-V2-S | `51.9s` | 2.74 GB | 91.30% |
| **RTX 5060 (Bare-Metal)** | EfficientNet-B4 | `38.4s` | 2.05 GB | 96.27% |
| **RTX 5060 (Docker/NGC 26.07)** | EfficientNet-B4 | `23.5s` | 2.05 GB | **98.61%** |

![Training Time Comparison](images/training_time_comparison.png)

## 3. VRAM and System Memory Efficiency

By enabling PyTorch Automatic Mixed Precision (`torch.amp`), we successfully trained the massive EfficientNet-B4 model using only **2.05 GB of VRAM**. This is a major optimization, ensuring that the model easily runs within the strict 8GB limits of mobile Blackwell/Ada GPUs.

![GPU VRAM and System Memory Usage](images/memory_usage_comparison.png)

## 4. Loss Convergence and Accuracy

Due to the vastly accelerated throughput, the Dockerized EfficientNet-B4 model reaches 98.61% convergence exponentially faster in wall-clock time compared to CPU training.

![Model Convergence Comparison](images/convergence_comparison.png)

## 5. Thermal Performance

Despite the aggressive CUDA utilization during the Dockerized EfficientNet-B4 run (sustained 65% GPU load), the RTX 5060 Laptop GPU maintained a highly stable thermal profile, peaking at just **63°C**.

![Thermal Performance Comparison](images/thermal_comparison.png)

## 6. Hardware & Software Stack

- **CPU:** AMD Ryzen 9 8940HX
- **GPU:** NVIDIA RTX 5060 8GB GDDR7 (Laptop GPU)
- **Host OS:** Windows
- **Docker Image:** `nvcr.io/nvidia/pytorch:26.07-py3`
- **Container Features:** `--gpus all`, `--ipc=host` (Shared Memory Optimization)

## 7. Setup & Reproducibility

To reproduce these benchmarks and capture your own telemetry:

```bash
# 1. Run Dockerized EfficientNet-B4
powershell -ExecutionPolicy Bypass -File .\run_docker_train.ps1

# 2. Generate the Telemetry Plots
python scripts\benchmark_plotter.py
```
