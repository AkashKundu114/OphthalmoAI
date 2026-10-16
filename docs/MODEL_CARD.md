# OphthalmoAI Model Card (Mitchell et al. Schema)

**Model Release Version:** 0.1.0-alpha  
**Date:** October 17, 2026  
**License:** Apache 2.0 (Code) / Pretrained Research Licenses (Weights)  
**System Type:** Vision Ensemble & Interpretability Pipeline for Retinal Fundus Screening  
**Primary Contact:** Akash Kundu ([akashkundu1152@gmail.com](mailto:akashkundu1152@gmail.com))

---

## 1. Model Details

### 1.1 Architecture & Ensembling
- **Primary Monolithic Backbone:** `EfficientNet-B4` (19.3M parameters, $384\times384\times3$ input) utilizing compound depth, width, and resolution scaling. Serves synchronous predictions and high-resolution spatial Grad-CAM saliency extraction.
- **Heterogeneous Soft-Voting Vision Ensemble:**
  1. `DenseNet-201` (20.0M parameters, dense feature concatenation for boundary delineation)
  2. `ConvNeXt-Small` (50.2M parameters, inverted bottleneck depthwise separable convolutions)
  3. `EfficientNet-V2-M` (54.1M parameters, progressive training with fused MBConv layers)
- **Edge Deployment Architecture:** `MobileNet-V3-Small` (FP32/INT8 quantized ONNX runtime, $224\times224\times3$ input) designed for in-browser zero-cloud inference via WebAssembly SIMD.

### 1.2 Checkpoint Integrity & Cryptographic Checksums

| Model Checkpoint | Parameter Size | Input Tensor | SHA256 Cryptographic Checksum |
| :--- | :---: | :---: | :--- |
| `efficientnet_b4.pth` | 71.0 MB | $384\times384\times3$ | `be34c046d616d18f7b30559af202840fa8ff04aa4a20328af0bfb5b448ad5328` |
| `densenet201.pth` | 73.8 MB | $384\times384\times3$ | `b64aea9e8980ba30f620d9bfcbc6aba5a1dcf8eea7753c1d74a10a8ffdf7834c` |
| `convnext_small.pth` | 198.0 MB | $384\times384\times3$ | `7959ebbd5203c07c3510eedb6c0bba97452459b8a2f2dfb6d53563e74ee23a7b` |
| `efficientnet_v2_m.pth` | 213.0 MB | $384\times384\times3$ | `e526d41c4edfc0b4a463810d5c48df1b1072279a311b5d083632ee25cc4d1250` |
| `edge_model_fp32.onnx` | 6.1 MB | $224\times224\times3$ | `b8da84126bfd66ad82ef6947725ee88db9ae3b3fb1e1df23888b4958c6b7de1c` |

---

## 2. Intended Use & Target Users

- **Intended Purpose:** Investigational clinical decision-support and educational screening assistance for ophthalmologists, optometrists, and research institutions.
- **Target Population:** Adult individuals undergoing routine ocular screening with color fundus photography centered on the macula and optic disc.
- **Out-of-Scope Uses (STRICT PROHIBITION):**
  - Standalone, autonomous medical diagnosis without clinician oversight.
  - Triage in acute emergency ocular trauma (e.g., globe rupture, chemical burns).
  - Pediatric retinopathy screening (e.g., Retinopathy of Prematurity - ROP).
  - External eye photography (slit-lamp, anterior segment, or smartphone selfies).

---

## 3. Training & Validation Datasets

- **Primary Training Corpus:** Multi-center aggregated dataset derived from EyePACS, Messidor-2, and APTOS 2019 Blindness Detection archives ($N=38,788$ images).
- **Patient-Level Splitting:** Splits partitioned strictly by patient ID ($70\%$ train, $15\%$ validation, $15\%$ test). Zero subject overlap across partitions.
- **External Validation Benchmark:** Independent Indian Diabetic Retinopathy Image Dataset (IDRiD, $N=413$ images, Kowa VX-10alpha digital camera, Nanded, India).

---

## 4. Empirical Evaluation Metrics (Measured)

### 4.1 Internal Test Benchmark ($N = 938$ test images)

| Metric | Measured Value | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Top-1 Accuracy** | **97.23%** | [96.12%, 98.24%] |
| **Macro-Averaged AUROC** | **0.999** | [0.997, 1.000] |
| **Macro-Averaged F1** | **97.04%** | [95.88%, 98.15%] |
| **Weighted F1** | **97.23%** | [96.14%, 98.26%] |
| **Pre-Calibration ECE** | **0.0884** | [0.0712, 0.1054] |
| **Post-Platt Scaling ECE** | **0.0312** | [0.0241, 0.0402] |

### 4.2 Multi-Seed Stability ($K = 5$ seeds: 42, 101, 2024, 7, 999)
- **Ensemble Mean Accuracy:** $85.18\% \pm 0.17\%$
- **DenseNet-201 Mean Accuracy:** $84.43\% \pm 0.17\%$
- **McNemar Test vs. ResNet-50 Baseline:** $\chi^2 = 51.97$, $p = 5.63 \times 10^{-13}$ (Statistically significant improvement over single-backbone baseline).

### 4.3 External Dataset Generalization (IDRiD, $N = 413$)
- **Sensitivity:** $99.64\%$ [98.91%, 100.00%] (Identifies nearly all pathological lesions).
- **Specificity:** $16.42\%$ [10.40%, 22.63%] (Reflects optical distribution shift under Indian clinical conditions with Kowa optics; highlights necessity of domain adaptation and human-in-the-loop review).

---

## 5. Post-Hoc Calibration & Conformal Prediction

- **Platt Temperature Scaling:** Fitted temperatures stored in `models/calibration.json`:
  - EfficientNet-B4: $T = 1.2099$
  - ConvNeXt-Small: $T = 1.2073$
  - DenseNet-201: $T = 1.1799$
  - EfficientNet-V2-M: $T = 1.2097$
- **Adaptive Weight Conformal Risk Control (AW-CRC):** Guarantees that the predicted label set contains the true disease state with $95\%$ statistical coverage ($\alpha = 0.05$).

---

## 6. Limitations & Caveats

1. **Optical Sensor Shift:** Camera hardware differences (Topcon vs. Zeiss vs. Kowa) introduce color balance and luminance variations.
2. **Dense Media Opacity:** Severe cataracts cause diffuse blurring that can degrade visualization of subtle microaneurysms.
3. **Severe Pathology Co-occurrence:** Patients presenting with simultaneous proliferative DR and wet AMD may trigger multi-label uncertainty warnings.
4. **Research Prototype Designation:** Software performance in software testing does not substitute for prospective multi-center clinical trials.
