# OphthalmoAI: Autonomous Research Reproducibility Guide

**Version:** 2.0  
**Audit Standard:** Full Evidence-Driven Reproducibility Standard  
**Date:** October 31, 2026  
**Hardware Verified:** NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 13.4), 32-core AMD64 host, 32GB RAM.  
**Software Verified:** Python 3.14.6 / Python 3.12, PyTorch 2.16.0.dev20261017+cu134, Torchvision 0.30.0.dev20261030+cu134, ONNX 1.23.2, ONNX Runtime 1.20.0.

---

## 1. System Requirements & Setup

### A. Environment Dependencies
Ensure Python 3.10+ and a CUDA-capable GPU (or CPU) are installed:
```powershell
# Clone or navigate to the repository root
cd D:\OphthalmoAI

# Install required packages
python -m pip install torch torchvision --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu134
python -m pip install onnx onnxruntime fastapi uvicorn scikit-learn pandas numpy scipy matplotlib pytest
```

### B. Directory Structure Verification
Verify that required directories exist:
```powershell
python -c "from pathlib import Path; assert (Path('dataset/processed') and Path('models') and Path('research/results'))"
```

---

## 2. Dataset Ingestion & Partitioning

To prepare the clean v2 dataset without data leakage:

### A. Forensic Deduplication & Manifest Assembly
```powershell
python scripts/forensic_deduplication_v2.py
```
*Expected Output:*
- Computes SHA-256 and 64-bit dHash on all images.
- Identifies 884 duplicate clusters spanning 1,808 images.
- Isolates 1,592 conflicting diagnostic duplicates into `dataset/processed/duplicate_clusters_v2.csv`.
- Generates `dataset/processed/manifest_v2.csv` ($N = 13,892$).

### B. Group-Stratified Partitioning
```powershell
python scripts/split_patient_level_v2.py
```
*Expected Output:*
- Groups identical dhashes and exact SHA-256 duplicates into unified clusters.
- Generates 10-fold split:
  - `train_v2.csv` (70%, $N = 9,725$)
  - `validation_v2.csv` (10%, $N = 1,389$)
  - `calibration_v2.csv` (10%, $N = 1,389$)
  - `test_v2.csv` (10%, $N = 1,389$ - SEALED BENCHMARK)
- Verifies zero group and zero hash leakage across partitions.

---

## 3. Training & Calibration

### A. Training Constituent Deep Backbones
Train individual constituent backbones on `train_v2.csv`:
```powershell
# ConvNeXt-Small
python scripts/train_model.py --arch convnext_small --precision fp16 --batch-size 16 --epochs 25

# DenseNet-201
python scripts/train_model.py --arch densenet201 --precision fp16 --batch-size 16 --epochs 25

# EfficientNet-V2-M
python scripts/train_model.py --arch efficientnet_v2_m --precision fp16 --batch-size 16 --epochs 25
```
Checkpoints saved to:
- `models/convnext_small.pth` (50.2M parameters)
- `models/densenet201.pth` (20.0M parameters)
- `models/efficientnet_v2_m.pth` (54.1M parameters)

### B. Training the Stacking Meta-Classifier
```powershell
python scripts/train_ensemble.py --batch-size 16 --epochs 30
```
*Checkpoint saved to:* `models/meta_classifier.pth` (484MB).

### C. Temperature Calibration (Platt Scaling)
Optimize per-model temperatures on `validation_v2.csv` holdout:
```powershell
python scripts/calibrate_models.py
```
*Output written to:* `models/calibration.json`.

### D. Admissibility-Weighted Conformal Risk Control (AW-CRC)
Compute empirical nonconformity quantiles on `calibration_v2.csv`:
```powershell
python scripts/run_aw_crc_calibration.py
```
*Output written to:* `models/aw_crc_calibration.json` and `models/conformal_calibration.json`.

---

## 4. Evaluation & Statistical Testing

### A. Primary Sealed Benchmark Evaluation
Evaluate all models on sealed `test_v2.csv` ($N = 1,389$):
```powershell
python scripts/evaluate_v2_benchmarks.py
```
*Expected Outputs:*
- Top-1 Accuracy:
  - Meta-Ensemble: **98.70%** (1,371 / 1,389)
  - Soft-Voting: **98.56%** (1,369 / 1,389)
  - EfficientNet-V2-M: **98.63%**
  - DenseNet-201: **98.13%**
  - ConvNeXt-Small: **97.98%**
- Macro AUROC: Meta = **0.9985**, Soft-Voting = **0.9988**
- Expected Calibration Error (ECE): Meta = **0.0389**, Soft-Voting = **0.0477**

### B. Paired Statistical Hypothesis Testing
Run McNemar's exact two-tailed test:
```powershell
python scripts/run_statistical_tests_v2.py
```
*Expected Outputs:*
- Meta-Ensemble vs Soft-Voting: Discordant = 6 ($b=4, c=2$), **$p = 0.6875$** (Not Significant)
- Meta-Ensemble vs EfficientNet-V2-M: Discordant = 7 ($b=4, c=3$), **$p = 1.0000$** (Not Significant)
- Meta-Ensemble vs ConvNeXt-Small: Discordant = 18 ($b=14, c=4$), **$p = 0.03088$** (Significant, $p < 0.05$)

### C. Biophysical Optical Guardrail Stress Test
```powershell
python scripts/benchmark_guardrail_real.py
```
*Expected Outputs:*
- Distractor Rejection Specificity: **100.0%** (7/7 non-fundus modalities)
- Fundus Admission Sensitivity: **100.0%** (1,389/1,389)
- P50 Latency: **5.87 ms**, P99: **8.31 ms**

### D. Grad-CAM Adebayo Sanity Check
```powershell
python scripts/verify_gradcam_sanity.py
```
*Expected Output:*
- Pearson correlation under weight randomization: $r = -0.0818 \pm 0.1581$ (with seed 42). Passes Adebayo test ($r < 0.50$).

### E. Edge ONNX Parity & Latency Micro-Benchmark
```powershell
python scripts/benchmark_onnx.py --iterations 50 --warmup 10
```
*Expected Output:*
- P50 CPU Latency: **2.54 ms** (355.7 QPS)
- PyTorch-to-ONNX FP32 max absolute error: $< 4.53 \times 10^{-6}$

---

## 5. Automated Regression Test Suite

Execute the complete 317-test regression suite:
```powershell
python -m pytest tests/
```
*Expected Output:*
- **317 passed** in ~90 seconds. Zero failures, zero errors.

---

## 6. Reproducibility Disclosures & Limitations

1. **Hardware Telemetry:** GPU memory consumption varies by batch size: batch size 16 requires ~3.8GB VRAM (FP16); batch size 32 requires ~6.9GB VRAM.
2. **Kaggle CFE Provenance:** Web-scraped CFE fundus images lack hospital patient accession numbers. Near-duplicates are clustered via perceptual dHash DSU; however, contralateral eyes cannot be guaranteed separated.
3. **External Domain Shift:** Evaluation on IDRiD and RIM-ONE DL exhibits acute specificity collapse (16.42% and 1.05%). Local camera re-calibration is required before deployment on new camera hardware.
