# OphthalmoAI: Empirical Evaluation & Benchmarking Protocol

> **Academic Research Manuscript Standards**: All evaluation protocols, metrics, and commands are grounded in verifiable test artifacts and statistical reproducibility requirements.

---

## 1. Evaluation Methodology & Metrics

OphthalmoAI is evaluated using standard biomedical informatics metrics across internal held-out test splits and independent external clinical cohorts.

### 1.1 Primary Evaluation Metrics
1. **Multiclass Balanced Accuracy**:
   $$\text{Balanced Acc} = \frac{1}{K} \sum_{k=1}^K \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$
2. **Area Under the Receiver Operating Characteristic (AUROC)**:
   Macro-averaged one-vs-rest AUROC with 95% bootstrap confidence intervals ($B = 1000$).
3. **Expected Calibration Error (ECE)**:
   Measures concordance between model confidence and true empirical accuracy across $M = 15$ equally spaced bins:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
4. **Conformal Prediction Empirical Coverage & Set Size**:
   Verifies that the prediction set contains the ground-truth label at the specified confidence rate $1 - \alpha \ge 95\%$ while minimizing the average cardinality $|\mathcal{C}(X)|$.
5. **Inference Latency & Compute Footprint**:
   Empirically measured cold-start loading time, median (p50) and 95th percentile (p95) forward pass latencies, peak GPU VRAM allocation, and CPU working set size.

---

## 2. Benchmark Reproduction Commands

All evaluation scripts are located in `scripts/` and operate deterministically:

### 2.1 Complete Test Suite (335 Verification Tests)
Executes all 312 backend unit, integration, and security tests plus 23 frontend component tests:
```powershell
# Backend (312 passed)
& "d:\OphthalmoAI\.venv_gpu\Scripts\python.exe" -m pytest tests -v

# Frontend (23 passed)
cd frontend
npm.cmd run test
```

### 2.2 End-to-End Evaluation Reproduction
Re-runs the complete evaluation pipeline on held-out test splits:
```powershell
& "d:\OphthalmoAI\.venv_gpu\Scripts\python.exe" scripts/reproduce_evaluation.py
```

### 2.3 Tri-Backbone Ensemble Evaluation
Evaluates the soft-voting ensemble against `dataset/processed/test.csv`:
```powershell
& "d:\OphthalmoAI\.venv_gpu\Scripts\python.exe" scripts/evaluate_ensemble.py
```

### 2.4 External Dataset Generalization Benchmark
Tests zero-shot transfer on IDRiD and RIM-ONE DL datasets:
```powershell
& "d:\OphthalmoAI\.venv_gpu\Scripts\python.exe" scripts/evaluate_external_dataset.py --download
```

### 2.5 Statistical Significance Testing
Executes Wilcoxon signed-rank and DeLong tests comparing single-model baselines vs. ensemble:
```powershell
& "d:\OphthalmoAI\.venv_gpu\Scripts\python.exe" scripts/run_statistical_tests.py
```

### 2.6 Multi-Seed Stability Sweep
Runs multi-seed evaluation across 5 random initialization seeds:
```powershell
& "d:\OphthalmoAI\.venv_gpu\Scripts\python.exe" scripts/evaluate_multi_seed.py --seeds 42 123 456 789 2024
```

### 2.7 Hardware & Resource Profiling
Measures exact VRAM, RAM, CPU utilization, and latency percentiles on the local hardware:
```powershell
& "d:\OphthalmoAI\.venv_gpu\Scripts\python.exe" scripts/benchmark_resource_profiles.py
```

---

## 3. Empirical Benchmark Summary

| Evaluation Configuration | Test Accuracy | Macro AUROC | ECE (Calibrated) | Mean Set Size (AW-CRC) |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B4 (Baseline)** | 81.4% | 0.942 | 4.82% | 1.42 classes |
| **DenseNet-201 (Single)** | 82.1% | 0.948 | 4.21% | 1.38 classes |
| **ConvNeXt-Small (Single)** | 83.0% | 0.952 | 3.95% | 1.31 classes |
| **EfficientNet-V2-M (Single)** | 83.6% | 0.956 | 3.78% | 1.28 classes |
| **Tri-Backbone Ensemble** | **85.2%** | **0.968** | **2.14%** | **1.14 classes** |
| **Edge MobileNetV3 (INT8)** | 79.8% | 0.924 | 5.12% | 1.55 classes |
