# Machine Learning Pipeline, Evaluation & Reproducibility Report: Project B — OphthalmoAI

**Target Repository:** `d:\OphthalmoAI`  
**Date:** October 16, 2026  
**Auditor:** Senior Medical ML Engineer & Adversarial Reviewer  
**Frameworks:** PyTorch 2.12.0.dev+cu128, ONNX Runtime Web 1.30, Scikit-Learn 1.9.1, NumPy 2.5.3

---

## 1. Executive Summary: Evidence vs. Projections

An exhaustive audit of the OphthalmoAI machine learning pipeline was conducted to evaluate model artifacts, preprocessing parity, serving latency, and statistical evaluation scripts.

While the core PyTorch ensemble (`densenet201.pth`, `convnext_small.pth`, `efficientnet_v2_m.pth`) executes genuine live inference on fundus imagery with valid Platt calibration and conformal risk guarantees, **multiple secondary benchmark numbers and reproducibility scripts were found to be synthetic, pre-baked, or hardcoded.**

---

## 2. Model Inventory & Serving Reality

### 2.1 Verified PyTorch Model Checkpoints
The following neural network weights exist on disk in `models/` and were verified for inference capability:
- `densenet201.pth` (FP32 PyTorch state dict)
- `convnext_small.pth` (FP32 PyTorch state dict)
- `efficientnet_v2_m.pth` (FP32 PyTorch state dict)
- `efficientnet_b4.pth` (FP32 PyTorch state dict, used for Grad-CAM explainability)
- `retfound_finetuned.pth` (Comparative foundation model checkpoint)
- `hybrid_meta_classifier.pth` (Meta-classifier checkpoint)

### 2.2 Serving Engine Reality: PyTorch FP32 vs. ONNX Serving
- **Production Screening (`POST /predict`):** Executes **100% PyTorch eager inference** on GPU/CPU.
- **Precision:** Executed in full **FP32 precision**. While FP16 checkpoints exist in `models/`, live inference in `main.py` does not cast tensors to `torch.float16`.
- **Server-Side ONNX Status:**
  - Exported and verified [`models/edge_model_fp32.onnx`](file:///d:/OphthalmoAI/models/edge_model_fp32.onnx) for FP32 serving with strict numerical parity ($\Delta_{max} = 3.81 \times 10^{-6}$ vs. PyTorch eager weights).
  - With dynamic input dimension adaptation implemented in `backend/onnx_inference.py`, real micro-benchmarks were executed and logged in [`docs/benchmarks/onnx_benchmark_results.json`](file:///d:/OphthalmoAI/docs/benchmarks/onnx_benchmark_results.json):
    - **FP32 ONNX Runtime (CPU):** **1.97 ms p50** (504.2 QPS, 91.88x speedup) with `is_synthetic: false` and `mode: "REAL BENCHMARK"`.
    - **INT8 Quantized Edge ONNX:** **9.99 ms p50** (87.8 QPS).
    - **PyTorch FP32 Eager Baseline:** 181.00 ms.
  - The historical 84.2 ms synthetic projection has been reconciled with measured real values.
- **Edge ONNX Status:** Client-side in-browser edge screening executes a genuine quantized MobileNetV3-Small INT8 model (`frontend/public/models/edge_model.onnx`, 1.69 MB) via `onnxruntime-web` with Ben Graham preprocessing.
- **Live E2E Verification:** Automated live end-to-end integration tests in [`tests/backend/test_e2e_live_pipeline.py`](file:///d:/OphthalmoAI/tests/backend/test_e2e_live_pipeline.py) execute authentic color fundus scans (`fundus_normal.jpg`, `fundus_diabetic_retinopathy.jpg`), verifying schema compliance, Platt calibration, and non-fundus rejection.

---

## 3. Retinal Preprocessing & Transformation Parity

The preprocessing pipeline was verified for numerical consistency between training and inference:

$$\mathbf{x}_{\text{norm}} = \frac{\text{Resize}_{384\times 384}(\text{ApertureCrop}(\mathbf{x})) - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

- **Input Size:** 384 × 384 pixels for server-side ensemble backbones; 224 × 224 pixels for edge MobileNetV3.
- **Aperture Detection:** Automated circular retinal bounding box crop removes black borders.
- **Color Space:** RGB channels with ImageNet normalization ($\boldsymbol{\mu} = [0.485, 0.456, 0.406]$, $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$).
- **Post-Hoc Platt Scaling:** Temperature scaling parameters from `models/calibration.json` are applied directly to raw logits prior to softmax:
  - `densenet201`: $T = 1.1799$
  - `convnext_small`: $T = 1.2073$
  - `efficientnet_v2_m`: $T = 1.2097$
  - `efficientnet_b4`: $T = 1.2099$

---

## 4. Evaluation Suite & Reproducibility Script Audit

### 4.1 "0.3s Reproducibility" Investigation
- `REPRODUCIBILITY.md` states: *"Replicate all paper metrics in <0.30s on single CPU."*
- **Audit Finding:** `scripts/reproduce_evaluation.py` executes in $<0.30\text{ s}$ because Suites 2, 3, 5, 6, and 7 are static `print()` statements iterating over pre-defined tuples rather than executing neural network evaluation on the 938 clinical images.
- **Recommendation:** Document that `reproduce_evaluation.py` is a fast summary reporter of saved benchmark artifacts, and direct users to `evaluate_ensemble.py` for full forward-pass evaluation on test cohorts.

### 4.2 Synthetic Statistical Vectors in Auxiliary Scripts
- **McNemar Tests (`evaluate_multi_seed.py:171`):** The contingency arrays for McNemar's paired test are synthesized via vector concatenation (`np.array([1]*775 + [0]*24 + ...)`) rather than derived from side-by-side model predictions.
- **Calibration Curves & DCA (`evaluate_extended_clinical_battery.py:258`):** Calibration metrics (ECE across 10/15/20 bins, Brier score, NLL) and Decision Curve Analysis (DCA net benefit) are computed on random numbers drawn from a Beta distribution (`rng.beta(9.5, 1.8)`), not real test data.

---

## 5. Explainability (Grad-CAM) Verification

- **Surrogate Model:** Saliency maps are generated by backpropagating through `MONOLITHIC_MODEL.features[-1]` on EfficientNet-B4.
- **Pixel-Aligned Grounding:**
  - `backend/main.py` has been updated to return `preprocessed_image` (the clean 384x384 cropped retina) alongside the Grad-CAM heatmap overlay.
  - Saliency maps align with verified retinal anatomy (optic disc and macular vascular arcades) without spatial distortion.
