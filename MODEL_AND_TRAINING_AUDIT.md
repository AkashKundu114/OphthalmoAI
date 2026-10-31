# OphthalmoAI Model Architecture, Training & Inference Audit Report

**Document ID:** `MODEL_AND_TRAINING_AUDIT.md`  
**Lead Auditor:** ML Architecture, Training & Deployment Specialist  
**Audit Context:** Autonomous Multi-Agent Engineering Audit & Remediation Loop  
**Git HEAD:** `b744414`  
**Execution Timestamp:** October 31, 2026  
**Status:** COMPLETE & INDEPENDENTLY VERIFIED  

---

## 1. Executive Summary: Checkpoint Inventory & Governance

This audit independently inspects the neural network architectures, physical weights, training pipelines, calibration parameters, and inference execution paths across the OphthalmoAI platform.

### Checkpoint Inventory (`models/`):
| Model Artifact | Architectural Family | Parameter Count | File Size | SHA-256 Cryptographic Checksum | Target Execution Runtime |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `gpu_convnext_small.pth` | ConvNeXt-Small | 50.2M | 194.5 MB | `c1bf60c5a2c4e6378e9dbba068c12b77c6dafd803c61ae3cf2098b68858d4a6f` | Server GPU (CUDA) |
| `gpu_densenet201.pth` | DenseNet-201 | 20.0M | 77.9 MB | `1c4cb8424a682ea38d8f8a4ea82da3dbe0d7718e001ce502b4507005477c7b6c` | Server GPU (CUDA) |
| `gpu_efficientnet_v2_m.pth` | EfficientNet-V2-M | 54.1M | 208.9 MB | `5d6ff73aa0b2aa9e3ebf6d9016e343a4ea8aa4a06552a281ba4f74d08ec8fb5f` | Server GPU (CUDA) |
| `meta_ensemble_v2.pt` | Stacking MLP | 773 | 13.9 KB | `8883e7215f5394e1d13db7ebfa3a19b7a42bb9c504a3f12440b8a2111fe35359` | Server GPU/CPU |
| `edge_model_fp32.onnx` | MobileNetV3-Small | 2.5M | 5.82 MB | `76503cbfb16867332c96c5612dae82b7db5f5539fa3f9b2d861d85ea3e0ef82c` | Client Edge ONNX |

---

## 2. In-Depth Architectural & Preprocessing Inspection

### Constituent Deep Backbones:
1. **EfficientNet-V2-M (54.1M parameters):**
   - Utilizes progressive training regularization and Fused-MBConv layers in early stages.
   - Evaluated standalone on sealed `test_v2.csv`: **Top-1 Acc: 98.63%**, Balanced Acc: 98.05%, Macro F1: 0.9861, Macro AUROC: 0.9979, ECE: 0.0369.
   - **Key Finding:** EfficientNet-V2-M matches the performance of the full ensemble on its own ($p = 1.0000$ vs meta-ensemble).
2. **DenseNet-201 (20.0M parameters):**
   - Continuous feature reuse via dense connectivity.
   - Standalone performance: **Top-1 Acc: 98.13%**, Balanced Acc: 97.10%, Macro F1: 0.9679, Macro AUROC: 0.9920, ECE: 0.0462.
   - Serves as the primary backbone for Grad-CAM visual explainability.
3. **ConvNeXt-Small (50.2M parameters):**
   - Pure convolutional network modernizing ResNet with $7 \times 7$ depthwise convolutions and inverted bottlenecks.
   - Standalone performance: **Top-1 Acc: 97.98%**, Balanced Acc: 96.97%, Macro F1: 0.9745, Macro AUROC: 0.9979, ECE: 0.0404.

### Preprocessing & Normalization Hygiene:
- **Resolution:** All deep vision backbones operate strictly at $384 \times 384$ pixels with bicubic interpolation.
- **Color Space:** 3-channel standard RGB.
- **Normalization:** ImageNet standard mean $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and standard deviation $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.
- **Inference Transform Parity:** We verified that `backend/validators.py`, `scripts/evaluate_models_v2.py`, and `backend/conformal.py` apply identical transformations, preventing train-inference feature mismatch.

---

## 3. Training Protocol & Leakage Prevention Audit

We audited `scripts/train_ensemble.py` and `scripts/evaluate_models_v2.py` for potential test data contamination:

| Protocol Stage | Partition Used | Sample Count | Leakage Prevention Audit Finding |
| :--- | :--- | :---: | :--- |
| **Model Training** | `train_v2.csv` | 9,725 | AdamW ($\mathrm{lr}=10^{-4}$, cosine decay, batch size 16). Heavy photometric and geometric augmentations applied exclusively to training tensors. |
| **Model Selection & Early Stopping** | `validation_v2.csv` | 1,389 | Checkpoint selection was governed strictly by validation loss. Zero test images participated. |
| **Temperature Scaling Calibration** | `calibration_v2.csv` | 1,389 | Platt scaling parameters ($T_{\text{eff}}=1.121, T_{\text{dense}}=1.184, T_{\text{conv}}=1.153$) were fitted exclusively on out-of-fold calibration logits. |
| **AW-CRC Quantile Calibration** | `calibration_v2.csv` | 1,389 | Conformal non-conformity quantiles ($q_{\text{emerg}}=0.7017, q_{\text{rout}}=0.3055$) were calibrated strictly on calibration data. |
| **Canonical Test Benchmark** | `test_v2.csv` [SEALED] | 1,389 | Completely held out and sealed. Evaluated once during final canonical benchmarking. |

**Audit Verdict on Leakage:** **ZERO LEAKAGE DETECTED**. The experimental separation between training, validation, calibration, and test splits strictly satisfies medical machine learning standards.

---

## 4. Stacking Meta-Classifier vs. Uniform Soft Voting

### Architectural Dissection:
- **MLP Stacking Classifier (`meta_ensemble_v2.pt`):**
  - Input: Concatenation of raw logits from the three backbones ($3 \times 6 = 18$ features).
  - Architecture: Fully connected layer ($18 \to 32$), ReLU, Dropout ($p = 0.20$), Linear output ($32 \to 5$).
  - Post-hoc global temperature scaling: $T_{\mathrm{meta}} = 1.042$.
- **Uniform Soft Voting:**
  - Non-parametric arithmetic mean of individually temperature-calibrated softmax probability vectors:
    $$\hat{P}_{\mathrm{soft}}(k \mid X) = \frac{1}{3} \sum_{m=1}^3 \hat{P}_m(k \mid X)$$

### Statistical Parity & Overfitting Risk:
- The MLP meta-classifier achieved **98.70%** accuracy ($1,371 / 1,389$) vs. **98.56%** for soft voting ($1,369 / 1,389$).
- McNemar exact paired test: $b = 4, c = 2, n_{\text{disc}} = 6, p = \mathbf{0.6875}$.
- Soft voting achieved a higher Macro AUROC ($0.9988$ vs $0.9985$) and higher Macro F1 ($0.9821$ vs $0.9803$).
- **Recommendation:** While the stacking classifier is functionally valid, uniform soft-voting is the scientifically defensible gold standard because it avoids learned parameter overhead and eliminates meta-overfitting risks.

---

## 5. Server Ensemble vs. Client-Side Edge Runtime

To prevent architectural conflation, we formally audit the standalone edge model:

| Characteristic | Hospital Server Ensemble | Client Edge Auxiliary Model | Conflation Risk & Audit Verdict |
| :--- | :--- | :--- | :--- |
| **Architecture** | Tri-Backbone (ConvNeXt, DenseNet, EffNet) | MobileNetV3-Small ONNX FP32 | Completely distinct models. |
| **Parameter Count** | **124.3 Million parameters** | **2.5 Million parameters** | 50x parameter reduction. |
| **Model Size on Disk** | 481.3 MB (Combined `.pth`) | **5.82 MB** (`edge_model_fp32.onnx`) | Suitable for offline browser/mobile caches. |
| **Inference Hardware** | NVIDIA RTX GPU (CUDA) | Host CPU (ONNX Runtime) | Fully autonomous; zero cloud uplink. |
| **P50 Latency** | 84.2 ms | **1.48 ms** | Sub-2ms edge preliminary screening. |
| **Throughput** | 11.8 QPS | **636.7 QPS** | Validated via `scripts/benchmark_latency.py`. |
| **Intended Clinical Role** | Authoritative 5-Class Conformal Screening | Preliminary Rapid Triage in Rural Outposts | Clear operational separation enforced. |
