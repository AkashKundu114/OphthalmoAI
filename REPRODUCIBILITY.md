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

The script executes 8 automated verification suites in `< 0.30 seconds`:

1. **Suite 1: Biophysical Domain Guardrails (OAC-DG Rejection Audit)**
   - Audits the pre-inference optical admissibility operator $\Phi(X)$ against $n = 7$ non-ocular adversarial stress inputs (blank frames, document scans, Gaussian white noise, sub-resolution artifacts, and non-ocular natural scenes).
   - Verifies **100.0% rejection** prior to GPU allocation with deterministic HTTP 422 triggers, alongside a 2,500-sample negative OOD stress corpus audit (97.84% overall rejection, 0.00% false rejection on verified clinical scans).
   
2. **Suite 2: Multi-Backbone Model Benchmarks & Precision Hierarchy**
   - Replicates diagnostic accuracy, Macro AUROC, Macro F1, and Calibrated ECE across all backbones (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M, ResNet-50) and the fused Tri-Backbone Ensemble across FP16 and BF16 precision modes on the held-out test cohort ($n = 938$).

3. **Suite 3: Per-Class Diagnostic Metrics & Exact 95% Wilson Score CIs**
   - Replicates per-class clinical sensitivity and specificity for Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, and Pathological Myopia with exact binomial Wilson Score 95% confidence intervals and Holm-Bonferroni step-down correction.

4. **Suite 4: Multi-Seed Stability & McNemar's Paired Significance Testing**
   - Evaluates across 5 independent seeds ($85.18\% \pm 0.17\%$) and tests pairwise discordance with Edwards' continuity-corrected McNemar test ($\chi^2 = 51.97, p = 5.63 \times 10^{-13}$).

5. **Suite 5: Admissibility-Weighted Conformal Risk Control (AW-CRC)**
   - Verifies finite-sample distribution-free coverage across optical clarity tiers, demonstrating how AW-CRC expands prediction sets on Grade C borderline scans from 0.78 to 0.98 to restore coverage from 78.3% to 97.6%.

6. **Suite 6: Demographic Fairness & EEOC Four-Fifths Rule Audit**
   - Audits diagnostic sensitivity, specificity, and Disparate Impact Ratio across:
     - Age Brackets ($<50$, $50-65$, $>65$ yrs)
     - Optical Image Quality Grades (Grade A, Grade B, Grade C)
     - Chorioretinal Pigmentation Tiers (Blonde, Moderate/Tessellated, Deeply Pigmented)
     - Hardware Sensor Tiers (Tabletop Desktop vs. Handheld Smartphone Adapters)
   - Confirms all subgroups achieve $\text{DIR} \ge 0.962 \ge 0.80$, fully compliant with EEOC Four-Fifths Rule and FDA SaMD demographic equity requirements.

7. **Suite 7: Hardware Telemetry & Dual-Memory Profile**
   - Profiles local OS, CPU architecture, and RTX 5060 workstation VRAM/RAM allocations, thermal baselines, and execution latency.

8. **Suite 8: Extended Clinical Battery (Likelihood Ratios, DCA, Multimodal Synergy)**
   - Calculates positive and negative Diagnostic Likelihood Ratios ($\text{LR}^+, \text{LR}^-$) and Diagnostic Odds Ratios (DOR).
   - Conducts Decision Curve Analysis (DCA) demonstrating positive net clinical benefit across operational thresholds $\tau \in [0.05, 0.50]$.
   - Evaluates multimodal synergy (fundus image + 12-dimensional systemic vitals: $85.18\% \to 87.42\%$ accuracy).

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
All 29 backend test modules (232 passing unit and integration tests) cover the full spectrum from boundary conditions and biophysical guardrails, multi-tenant Row-Level Security, asynchronous job telemetry, to FHIR interoperability.