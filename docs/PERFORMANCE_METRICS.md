# Performance & Evaluation Benchmarks: OphthalmoAI

This document details the empirical clinical benchmarks, model calibration evaluation, and hardware telemetry for **OphthalmoAI (Calibrated Retinal Disease Predictor)**.

Hardware Testbed: **NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR6, 100.85W TGP)** paired with an **AMD Ryzen 9 HX (16 Cores / 32 Threads)** running PyTorch with Automatic Mixed Precision (AMP FP16).

---

## 1. Empirical Test Set Evaluation ($n = 938$)

The models were evaluated on a strictly segregated held-out test split of **938 verified color fundus photographs** across the 6 target classes.

<p align="center">
  <img src="images/benchmark_accuracy_comparison.png" alt="Benchmark Accuracy Comparison" width="90%" />
</p>

| Architecture / Model | Test Accuracy | Macro AUROC | Macro F1 | Calibration $T$ | Calibrated ECE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Tri-Backbone Soft-Voting Ensemble (SOTA)** | **85.18%** | **0.9805** | **0.8292** | **Ensemble** | **0.0644** |
| DenseNet-201 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0519 |
| ConvNeXt-Small | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0614 |
| EfficientNet-V2-M | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0268 |
| EfficientNet-B4 (Grad-CAM Engine) | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0582 |
| ResNet-50 (Baseline) | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0412 |

---

## 2. Multi-Class ROC & Calibration Temperatures

<p align="center">
  <img src="images/multiclass_roc_curves.png" alt="Multiclass ROC Curves" width="48%" />
  <img src="images/calibration_temperatures_chart.png" alt="Calibration Temperatures" width="48%" />
</p>

### Key Insights:
- **Macro AUROC (0.9805)**: Demonstrates exceptional discriminatory power across all 6 retinal categories.
- **Platt Temperature Scaling**: Dividing logits by learned temperatures $T \in [1.06, 1.34]$ corrects neural overconfidence, cutting Expected Calibration Error (ECE) to under $0.065$.

---

## 3. Class Sensitivity, Specificity & Confusion Matrix

<p align="center">
  <img src="images/ensemble_sensitivity_specificity.png" alt="Ensemble Sensitivity and Specificity" width="48%" />
  <img src="images/confusion_matrix_ensemble.png" alt="Confusion Matrix" width="48%" />
</p>

### Diagnostic Metrics per Class:
- **Normal**: Sensitivity 89.2% | Specificity 94.5%
- **Diabetic Retinopathy**: Sensitivity 88.5% | Specificity 95.8%
- **Glaucoma**: Sensitivity 82.1% | Specificity 96.2%
- **Cataract**: Sensitivity 86.4% | Specificity 97.1%
- **Age-related Macular Degeneration (AMD)**: Sensitivity 83.7% | Specificity 96.5%
- **Hypertensive Retinopathy / Pathological Myopia**: Sensitivity 81.1% | Specificity 95.9%

---

## 4. Hardware Telemetry & Resource Profiling

<p align="center">
  <img src="images/training_time_comparison.png" alt="Training Time Comparison" width="48%" />
  <img src="images/memory_usage_comparison.png" alt="Memory Usage Comparison" width="48%" />
</p>

### Telemetry Summary (NVIDIA RTX 5060 Laptop GPU):
- **Epoch Duration**: Scaled from 460.8s on multi-threaded CPU down to 10.9s–25.0s on GPU with PyTorch AMP FP16.
- **Dedicated VRAM**: Peak allocation remained under 4.62 GB during batch training (batch size 32), leaving ample headroom on the 8GB GPU.
- **Thermal Safety**: Operating temperatures stabilized at 70°C–78°C under full Tensor Core load, well beneath the 102°C thermal throttling ceiling.
