# Performance & Evaluation Benchmarks: OphthalmoAI

This document details the complete empirical clinical benchmarks, model calibration evaluation, hardware telemetry, and precision comparison (FP16 vs. BF16) for **OphthalmoAI (Calibrated Retinal Disease Predictor)**.

Hardware Testbed: **NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR6, 100.85W TGP)** paired with an **AMD Ryzen 9 HX (16 Cores / 32 Threads)** running PyTorch with Automatic Mixed Precision (AMP FP16 & Native BF16).

---

## 1. Empirical Test Set Evaluation ($n = 938$)

Evaluated on a strictly segregated held-out test split of **938 verified color fundus photographs** across the 6 target classes.

<p align="center">
  <img src="images/benchmark_accuracy_comparison.png" alt="Benchmark Accuracy Comparison" width="90%" />
</p>

| Architecture / Model | Precision | Test Accuracy | Macro AUROC | Macro F1 | Calibration $T$ | Calibrated ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Tri-Backbone Soft Ensemble (SOTA)** | **FP16** | **85.18%** | **0.9805** | **0.8292** | **Ensemble** | **0.0644** |
| DenseNet-201 | FP16 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0519 |
| ConvNeXt-Small | FP16 | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0614 |
| EfficientNet-V2-M | FP16 | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0268 |
| EfficientNet-B4 (Grad-CAM Engine) | FP16 | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0582 |
| ResNet-50 Baseline | FP32 | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0412 |
| **Calibrated Tri-Backbone Soft Ensemble** | **BF16** | **81.02%** | **0.9752** | **0.7814** | **Ensemble** | **0.0626** |
| EfficientNet-B4 | BF16 | 80.92% | 0.9697 | 0.7786 | 1.1188 | 0.0502 |
| EfficientNet-V2-M | BF16 | 80.49% | 0.9709 | 0.7773 | 1.1042 | 0.0410 |
| DenseNet-201 | BF16 | 80.28% | 0.9719 | 0.7743 | 1.1215 | 0.0433 |
| ResNet-50 | BF16 | 80.28% | 0.9687 | 0.7746 | 1.0824 | 0.0485 |
| ConvNeXt-Small | BF16 | 79.74% | 0.9688 | 0.7742 | 1.1450 | 0.0314 |

---

## 2. Multi-Class ROC & Calibration Temperatures

<p align="center">
  <img src="images/multiclass_roc_curves.png" alt="Multiclass ROC Curves" width="48%" />
  <img src="images/calibration_temperatures_chart.png" alt="Calibration Temperatures" width="48%" />
</p>

### Key Insights:
- **Macro AUROC (0.9805)**: Exceptional discriminatory power across all 6 retinal categories.
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

## 4. Hardware Telemetry: Dual-Memory (RAM + VRAM) & Training Speed

<p align="center">
  <img src="images/memory_usage_comparison.png" alt="Memory Usage Comparison" width="48%" />
  <img src="images/training_time_comparison.png" alt="Training Time Comparison" width="48%" />
</p>

### Dual Memory Allocation Profile:
| Model Architecture | Precision | Dedicated GPU VRAM (GB) | Host System RAM (GB) | VRAM Headroom |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-V2-M** | FP16 | 6.30 GB | 2.41 GB | 1.85 GB Free |
| **EfficientNet-B4** | FP16 | 5.31 GB | 2.57 GB | 2.84 GB Free |
| **ConvNeXt-Small** | FP16 | 4.97 GB | 2.48 GB | 3.18 GB Free |
| **DenseNet-201** | FP16 | 4.82 GB | 3.24 GB | 3.33 GB Free |
| **ResNet-50 (GPU)** | FP16 | 2.38 GB | 2.28 GB | 5.77 GB Free |
| **Meta-Ensemble Fusion** | FP16 | 0.86 GB | 2.21 GB | 7.29 GB Free |
| **ResNet-50 (CPU Baseline)**| FP32 | 0.00 GB | 5.33 GB | N/A |

### Training Speedup:
- **~25x Speedup vs CPU Baseline**: Moving from Ryzen 9 multi-threaded execution (1949.2s per epoch) to RTX 5060 GPU execution (78.5s on ConvNeXt-Small).
- All backbones operate well within the 8.0 GB GDDR6 physical limit, maintaining GPU temperatures below 78 °C.

---

## 5. Comparative Study: FP16 Mixed Precision vs. BF16 Precision

To investigate precision-calibration tradeoffs on modern Blackwell-generation Tensor Cores (RTX 5060), all six architectures were trained and calibrated independently under both **FP16 (Half Precision, 5-bit exponent, 10-bit mantissa)** and **BF16 (Bfloat16, 8-bit exponent, 7-bit mantissa)**.

<p align="center">
  <img src="images/bf16_vs_fp16_accuracy_comparison.png" alt="BF16 vs FP16 Accuracy Comparison" width="48%" />
  <img src="images/bf16_vs_fp16_calibration_comparison.png" alt="BF16 vs FP16 Calibration Comparison" width="48%" />
</p>
<p align="center">
  <img src="images/bf16_vs_fp16_training_time.png" alt="BF16 vs FP16 Training Time Comparison" width="70%" />
</p>

### Empirical Head-to-Head Comparison ($n = 938$ held-out test split):

| Architecture / Model | Precision | Test Accuracy | Macro AUROC | Macro F1 | Optimal $T$ | Calibrated ECE | Epoch Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ensemble (Production)** | **FP16** | **85.18%** | **0.9805** | **0.8292** | **Soft-Vote** | **0.0644** | — |
| Ensemble (Research) | BF16 | 81.02% | 0.9752 | 0.7814 | Soft-Vote | 0.0626 | — |
| DenseNet-201 | FP16 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0519 | 100.8s |
| DenseNet-201 | BF16 | 80.28% | 0.9719 | 0.7743 | 1.1215 | 0.0433 | 102.1s |
| ConvNeXt-Small | FP16 | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0614 | 78.5s |
| ConvNeXt-Small | BF16 | 79.74% | 0.9688 | 0.7742 | 1.1450 | 0.0314 | 79.2s |
| EfficientNet-V2-M | FP16 | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0268 | 87.2s |
| EfficientNet-V2-M | BF16 | 80.49% | 0.9709 | 0.7773 | 1.1042 | 0.0410 | 88.0s |
| EfficientNet-B4 | FP16 | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0582 | 92.4s |
| EfficientNet-B4 | BF16 | 80.92% | 0.9697 | 0.7786 | 1.1188 | 0.0502 | 93.6s |
| ResNet-50 | FP16 | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0412 | 49.8s |
| ResNet-50 | BF16 | 80.28% | 0.9687 | 0.7746 | 1.0824 | 0.0485 | 50.4s |

### Architectural Findings & Deployment Decision:
1. **Discriminative Precision**: FP16 outperforms BF16 on the primary vision ensemble by **+4.16% in test accuracy** (85.18% vs 81.02%) and **+0.0478 in Macro F1** (0.8292 vs 0.7814). The higher fraction of mantissa bits in FP16 (10 bits vs. 7 bits in BF16) preserves subtle gradient variations essential for detecting micro-lesions (punctate blot hemorrhages and fine microaneurysms < 15µm).
2. **Post-Hoc Calibration Stability**: BF16 architectures demonstrated slightly lower variance in Platt temperatures ($T_{\text{BF16}} \in [1.08, 1.15]$ vs $T_{\text{FP16}} \in [1.06, 1.34]$) and achieved marginal calibration gains on ConvNeXt-Small (ECE: 0.0314 vs 0.0614).
3. **Deployment Strategy**: In clinical triage, misdiagnosis carries severe irreversible patient risk. Because FP16 provides significantly higher sensitivity and specificity across critical vision backbones, **FP16 is designated as the active production engine**, while BF16 weights (`models/*_bf16_bs16.pth`) and calibration metadata (`models/calibration_bf16.json`) are preserved for comparative benchmarking.

---

## 6. Retinal Domain Verification Guardrails & Adversarial Robustness

To ensure clinical safety and prevent spurious inference on invalid inputs (everyday photographs, pet photos, documents, and adversarial noise), OphthalmoAI enforces multi-stage input validation prior to running model inferences or generating AI chat summaries.

### Multi-Stage Domain Filter Pipeline:
1. **Aperture & Geometry Verification**: Detects characteristic circular boundary and masked optical perimeter of color fundus cameras ($\ge 35\%$ corner darkness).
2. **Chorioretinal Chromophore Ratio**: Color fundus images exhibit strong red-to-blue backscatter from retinal pigment epithelium and choroidal blood flow ($\bar{R}/\bar{B} \ge 1.05$). Non-fundus imagery (e.g. blue skies, foliage, general indoor scenes) violates this physiological constraint.
3. **Spatial Autocorrelation Filtering**: Evaluates lag-1 horizontal and vertical pixel correlation ($r_{\text{spatial}} \ge 0.35$). Synthetic Gaussian noise and adversarial high-frequency perturbation patterns are rejected immediately.
4. **Vascular Contrast Ratio**: Quantifies green-channel contrast in the vascular bed relative to background retina.

### Adversarial Red-Team Audit Results:
- **Non-Fundus Rejection Rate**: **100% (7/7 adversarial test images blocked with HTTP 422)** including landscape photographs, domestic animals, random uniform noise, and indoor scenes.
- **Clinical Chatbot Prompt Defense**: **100% (13/13 attacks blocked)** including jailbreaks ("DAN", system prompt extraction, developer override modes), off-topic generation requests (code synthesis, creative writing, political essays), and unverified scan diagnosis attempts.

