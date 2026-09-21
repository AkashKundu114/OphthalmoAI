# OphthalmoAI: Empirical Reproducibility & Benchmark Verification Guide

This repository provides full independent reproducibility for all empirical benchmarks, biophysical domain guardrails, conformal coverage guarantees, and demographic fairness audits presented in the publication:

> **"Uncertainty-Aware Multi-Class Fundus Screening with Conformal Sets"**  
> *Research Manuscript & Empirical Benchmark Evaluation.*  
> Author: **Akash Kundu** (Techno India University & Independent Researcher)  
> Repository: [https://github.com/AkashKundu114/OphthalmoAI](https://github.com/AkashKundu114/OphthalmoAI)

---

## 1. Quickstart: One-Command Reproducibility Check

Reviewers and researchers can reproduce and verify all metrics in the paper without setting up any cloud compute capsules or proprietary services.

### Clone the Repository
```bash
git clone https://github.com/AkashKundu114/OphthalmoAI.git
cd OphthalmoAI
```

### Run the Standalone Reproducibility Suite
```bash
python scripts/reproduce_evaluation.py
```

The script executes 5 automated verification suites in `< 1.0 second`:

1. **Suite 1: Biophysical Domain Guardrails (OAC-DG Rejection Audit)**
   - Audits the pre-inference optical admissibility operator $\Phi(X)$ against $n = 7$ non-ocular adversarial stress inputs (blank frames, document scans, Gaussian white noise, sub-resolution artifacts, and non-ocular natural scenes).
   - Verifies **100.0% rejection** prior to GPU allocation with deterministic HTTP 422 triggers.
   
2. **Suite 2: Multi-Backbone Model Benchmarks (Table IV & Table V)**
   - Replicates diagnostic accuracy, Macro AUROC, Macro F1, and Calibrated ECE across all backbones (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M, ResNet-50) and the fused Tri-Backbone Ensemble across FP16 and BF16 precision modes on the held-out test cohort ($n = 938$).
   - Replicates per-class clinical sensitivity and specificity for Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, and Pathological Myopia.

3. **Suite 3: Urgency-Stratified Conformal Risk Control (US-CRC, Table VIII)**
   - Verifies distribution-free finite-sample statistical coverage:
     - Emergency sight-threatening conditions: $\ge 99.0\%$ coverage ($\alpha_{\text{emerg}} = 0.01$, observed: **99.36%**).
     - Routine ambulatory conditions: $\ge 95.0\%$ coverage ($\alpha_{\text{routine}} = 0.05$, observed: **95.72%**).
     - Verifies average conformal set size remains compact at **1.18 classes/patient** with 83.4% singletons.

4. **Suite 4: Demographic Fairness & EEOC Four-Fifths Rule Audit (Table VII)**
   - Audits diagnostic sensitivity, specificity, and Disparate Impact Ratio across:
     - Age Brackets ($<50$, $50-65$, $>65$ yrs)
     - Optical Image Quality Grades (Grade A, Grade B, Grade C)
     - Chorioretinal Pigmentation Tiers (Blonde, Moderate/Tessellated, Deeply Pigmented)
     - Hardware Sensor Tiers (Tabletop Desktop vs. Handheld Smartphone Adapters)
   - Confirms all subgroups achieve $\text{DIR} \ge 0.962 \ge 0.80$, fully compliant with the EEOC Four-Fifths Rule and FDA SaMD demographic equity requirements.

5. **Suite 5: Hardware Telemetry & Dual-Memory Profiling (Table VI)**
   - Profiles host system CPU/GPU RAM, VRAM allocations, and runtime execution latency.

---

## 2. Granular Verification Commands

To run specific verification modules individually:

```bash
# Verify only the biophysical domain guardrails:
python scripts/reproduce_evaluation.py --guardrail

# Verify only the classification metrics (Tables IV and V):
python scripts/reproduce_evaluation.py --benchmarks

# Verify only the conformal risk control coverage guarantees (Table VIII):
python scripts/reproduce_evaluation.py --conformal

# Verify only the demographic fairness audit (Table VII):
python scripts/reproduce_evaluation.py --fairness

# Profile local hardware and memory utilization (Table VI):
python scripts/reproduce_evaluation.py --telemetry
```

---

## 3. Full End-to-End System Execution (Optional)

To spin up the complete full-stack edge application locally:

### Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Launch the Backend API & Optical Guardrails
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
API Documentation will be live at `http://localhost:8000/docs`.

### Run Test Suite
```bash
pytest tests/
```
All 28 backend test modules cover the full spectrum from biophysical guardrails, multi-tenant Row-Level Security, asynchronous job telemetry, to FHIR interoperability.