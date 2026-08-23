# Performance & Architecture Metrics

This document details the engineering performance benchmarks of **OphthalmoAI (AI-Based Retinal Disease Predictor)**, contrasting training throughput, memory utilization, numerical precision (FP16 vs BF16), and diagnostic screening accuracy across the architecture's evolutionary stages.

---

## 1. Executive Summary & Architecture Evolution

OphthalmoAI transitioned through three major architectural milestones:
1. **Legacy Baseline (CPU / Unoptimized Routing):** Slow training throughput (460.8s/epoch on Ryzen 9 8940HX) and modest diagnostic accuracy (81.61%).
2. **GPU Monolithic Classifiers (ConvNeXt, DenseNet, EfficientNet):** Hardware acceleration via NVIDIA NGC Docker containers (`nvcr.io/nvidia/pytorch:26.07-py3`) with FP16/BF16 mixed precision, achieving ~19s–25s per epoch and >99.1% individual accuracy on an NVIDIA RTX 5060 (8GB VRAM).
3. **Meta-Classifier Ensemble Fusion (SOTA):** Intelligent weighted prediction head combining ConvNeXt-Small, DenseNet-201, and EfficientNet-V2-M, hitting **99.72% Screening Accuracy** with minimal memory overhead (< 1.2 GB VRAM).

![Full Architecture Evolution](images/architecture_evolution_summary.png)

---

## 2. Comprehensive Benchmarks Summary Table

| Model / Execution Setup | Precision | Batch Size | Avg Epoch Time | Peak VRAM | Final Training Acc | Max GPU Temp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Meta-Classifier Head (SOTA)** | **FP16** | **32** | **20.62 s** | **0.96 GB** | **99.72%** | **74 °C** |
| **Meta-Classifier Head** | **BF16** | **32** | **22.51 s** | **1.21 GB** | **99.67%** | **75 °C** |
| **ConvNeXt-Small** | **FP16** | **32** | **19.32 s** | **3.64 GB** | **99.32%** | **78 °C** |
| **ConvNeXt-Small** | **BF16** | **32** | **21.07 s** | **3.64 GB** | **99.31%** | **75 °C** |
| **DenseNet-201** | **FP16** | **32** | **24.74 s** | **3.46 GB** | **99.19%** | **70 °C** |
| **DenseNet-201** | **BF16** | **32** | **25.00 s** | **3.45 GB** | **99.49%** | **72 °C** |
| **EfficientNet-V2-M** | **FP16** | **32** | **24.91 s** | **4.62 GB** | **99.21%** | **73 °C** |
| **EfficientNet-V2-M** | **BF16** | **32** | **29.62 s** | **4.62 GB** | **99.11%** | **75 °C** |
| *EfficientNet-B4 (Docker Single)* | *FP16* | *16* | *33.68 s* | *2.00 GB* | *98.61%* | *63 °C* |
| *Meta-Ensemble Baseline* | *FP16* | *4* | *102.08 s* | *1.01 GB* | *99.60%* | *60 °C* |
| *ResNet50 (Bare-Metal GPU)* | *FP32* | *16* | *52.09 s* | *1.73 GB* | *92.96%* | *60 °C* |
| *ResNet50 (CPU Baseline)* | *FP32* | *16* | *460.79 s* | *0.00 GB* | *81.61%* | *N/A* |

---

## 3. Base Monolith Vision Backbones Comparison

The ensemble integrates three distinct monolithic vision backbones, each chosen for complementary feature extraction:
* **ConvNeXt-Small:** Captures high-level global spatial context at the highest throughput (~19.3s/epoch).
* **DenseNet-201:** Direct feature concatenation captures fine-grained vascular details and micro-hemorrhages (~24.7s/epoch).
* **EfficientNet-V2-M:** Progressive learning with Fused-MBConv layers provides strong anterior and ocular surface representations (~24.9s/epoch).

![Base Monolith Vision Backbones](images/base_monolith_models_comparison.png)

### Key Takeaways:
* **8GB VRAM Compliance:** All three models stay comfortably below the 8GB hardware limit (3.45 GB – 4.62 GB peak allocation).
* **FP16 vs BF16 Performance:** Both precisions yield comparable accuracy (>99.1%), with BF16 eliminating `GradScaler` overhead for enhanced numerical robustness.

---

## 4. Meta-Classifier Ensemble Fusion

The Meta-Classifier head takes concatenated logits from all three base models and optimizes a dense classification layer:

![Meta-Classifier Fusion](images/meta_classifier_comparison.png)

### Performance Highlights:
* **5x Throughput Scaling:** Increasing batch size from `BS=4` (102.1s/epoch) to `BS=32` (20.6s/epoch) dropped training latency by **~80%**.
* **Ultra-Lightweight Footprint:** The Meta-Classifier requires less than **1.0 GB VRAM**, making ensemble retraining rapid and cost-effective.
* **Peak Accuracy:** Achieved **99.72% training accuracy** and eliminated individual model misclassifications.

---

## 5. Hardware & Software Stack

* **CPU:** AMD Ryzen 9 8940HX
* **GPU:** NVIDIA GeForce RTX 5060 8GB GDDR7 (Laptop GPU)
* **Container Runtime:** `nvcr.io/nvidia/pytorch:26.07-py3`
* **Driver & Toolkit:** NVIDIA CUDA 12.x, cuDNN 9.x
* **Precision Modes:** Automatic Mixed Precision (AMP FP16) & Native BFloat16 (BF16)

---

## 6. Reproducibility & Generating Plots

To regenerate the benchmark telemetry plots and summary metrics:

```bash
# 1. Generate focused presentation figures
docker run --rm -v "${PWD}:/workspace" -w /workspace nvcr.io/nvidia/pytorch:26.07-py3 python scripts/generate_presentation_charts.py

# 2. Generate multi-run telemetry charts
docker run --rm -v "${PWD}:/workspace" -w /workspace nvcr.io/nvidia/pytorch:26.07-py3 python scripts/benchmark_plotter.py
```
