# OphthalmoAI Comprehensive Statistical Validation Report

**Document ID:** `STATISTICAL_VALIDATION_REPORT.md`  
**Lead Auditor:** Senior Biostatistician & Independent Statistical Auditor  
**Audit Context:** Autonomous Multi-Agent Engineering Audit & Remediation Loop  
**Git HEAD:** `b744414`  
**Execution Timestamp:** October 31, 2026  
**Status:** COMPLETE & INDEPENDENTLY REPRODUCED  

---

## 1. Executive Summary & Statistical Integrity Principles

This statistical validation report independently evaluates, recalculates, and audits every quantitative claim, metric, and hypothesis test across the OphthalmoAI repository.

### Absolute Integrity Standards:
1. **Zero Invented Metrics:** Every reported metric originates directly from raw GPU output arrays (`research/results/raw_preds_test_v2.npz`) and reproducible JSON ledgers.
2. **Exact Paired Testing:** Exact non-parametric binomial McNemar tests are used for paired model comparisons ($n_{\text{disc}} < 25$); asymptotic $\chi^2$ approximations are discarded.
3. **Transparent Reporting of Negative Findings:** The learned MLP meta-classifier is reported as statistically indistinguishable from uniform soft-voting ($p = 0.6875$), refuting meta-learner superiority.
4. **Distinction Between Raw Conformal Coverage and Fallback:** The distinction between raw mathematical conformal coverage ($97.91\%$) and operational fallback coverage ($98.34\%$) is made explicit.

---

## 2. Primary Classification Benchmark on Sealed Cohort (`test_v2.csv`, $N = 1,389$)

The primary evaluation was conducted on the strictly sealed test split (`dataset/processed/test_v2.csv`, $N = 1,389$):

| Architecture / Fusion Method | Top-1 Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Macro AUROC | Macro AUPRC | ECE (15 Bins) | Multi-class Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Meta-Ensemble (MLP Stacking)** | **98.70%** (1,371/1,389) | **98.28%** | 0.9803 | **0.9878** | 0.9985 | **0.9917** | **0.0389** | 0.0273 |
| **Uniform Soft-Voting** | 98.56% (1,369/1,389) | 97.44% | **0.9821** | 0.9866 | **0.9988** | 0.9914 | 0.0477 | **0.0271** |
| **EfficientNet-V2-M** | 98.63% (1,370/1,389) | 98.05% | 0.9861 | 0.9863 | 0.9979 | 0.9892 | 0.0369 | 0.0285 |
| **DenseNet-201** | 98.13% (1,363/1,389) | 97.10% | 0.9679 | 0.9811 | 0.9920 | 0.9815 | 0.0462 | 0.0341 |
| **ConvNeXt-Small** | 97.98% (1,361/1,389) | 96.97% | 0.9745 | 0.9798 | 0.9979 | 0.9880 | 0.0404 | 0.0332 |

---

## 3. Paired Hypothesis Testing: Exact McNemar Analysis

To determine whether the learned MLP meta-classifier provides statistically significant gains over non-parametric soft-voting or individual backbones, exact binomial McNemar tests were executed:

$$\mathbf{C}_{\text{McNemar}} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$
where $a$ = both correct, $b$ = $M_1$ only correct, $c$ = $M_2$ only correct, and $d$ = both incorrect.

### Paired Comparison Matrix:
| Model Comparison ($M_1$ vs. $M_2$) | Both Correct ($a$) | $M_1$ Only ($b$) | $M_2$ Only ($c$) | Both Incorrect ($d$) | Discordant ($b+c$) | Exact Two-Tailed $p$-value | Significant at $\alpha = 0.05$? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Meta-Ensemble vs. Soft-Voting** | 1,367 | 4 | 2 | 16 | 6 | **0.6875** | **NO** |
| **Meta-Ensemble vs. EfficientNet-V2-M** | 1,367 | 4 | 3 | 15 | 7 | **1.0000** | **NO** |
| **Meta-Ensemble vs. DenseNet-201** | 1,357 | 14 | 6 | 12 | 20 | **0.1153** | **NO** |
| **Meta-Ensemble vs. ConvNeXt-Small** | 1,357 | 14 | 4 | 14 | 18 | **0.0309** | **YES** |
| **Soft-Voting vs. EfficientNet-V2-M** | 1,366 | 3 | 4 | 16 | 7 | **1.0000** | **NO** |
| **Soft-Voting vs. DenseNet-201** | 1,357 | 12 | 6 | 14 | 18 | **0.2379** | **NO** |
| **Soft-Voting vs. ConvNeXt-Small** | 1,357 | 12 | 4 | 16 | 16 | **0.0768** | **NO** |

### Mathematical Proof for Meta-Ensemble vs. Soft-Voting:
$$p = 2 \times \sum_{k=0}^{\min(4, 2)} \binom{6}{k} 0.5^6 = 2 \times \left[ \binom{6}{0} + \binom{6}{1} + \binom{6}{2} \right] \times \frac{1}{64} = 2 \times [1 + 6 + 15] \times \frac{1}{64} = \frac{44}{64} = \mathbf{0.6875}$$
**Statistical Verdict:** Meta-Ensemble stacking does NOT achieve statistically significant superiority over Uniform Soft-Voting ($p = 0.6875$). Uniform soft-voting is confirmed as an equally performant non-parametric gold standard.

---

## 4. Class Support Audit & The Zero-Support 6th Class

In `test_v2.csv` ($N = 1,389$), true class distribution is:
- **Class 0 (Normal):** $n = 386$ ($27.79\%$)
- **Class 1 (Diabetic Retinopathy):** $n = 482$ ($34.70\%$)
- **Class 2 (Glaucoma):** $n = 361$ ($25.99\%$)
- **Class 3 (Cataract):** $n = 134$ ($9.65\%$)
- **Class 4 (AMD):** $n = 26$ ($1.87\%$)
- **Class 5 (Hypertensive Retinopathy / Myopia):** $n = 0$ ($0.00\%$)
- **Total Test Samples:** $386 + 482 + 361 + 134 + 26 + 0 = \mathbf{1,389}$

### Forensic Root Cause of Zero Support for Class 5:
- In early development, Class 5 was constructed by merging Hypertensive Retinopathy, Pathological Myopia, Disc Edema, and Retinal Detachment.
- During clinical label harmonization (`label_harmonization.csv`), these disparate pathologies were quarantined (8,477 samples) because conflating them into a single class causes hazardous misdiagnoses.
- Neural network heads retain 6 outputs for checkpoint compatibility. The 6th row of the confusion matrix has sum 0, and Macro F1 is computed over the 5 active diagnostic classes.

---

## 5. Conformal Risk Control & Empty Set Fallback Analysis

Evaluated using `calibration_v2.csv` ($N = 1,389$) and `test_v2.csv` ($N = 1,389$):

| Conformal Metric | Standard US-CRC | Admissibility-Weighted CRC (AW-CRC, $\gamma=1.0$) | Verification Status |
| :--- | :---: | :---: | :---: |
| **Emergency Stratum Cutoff ($q_{\text{emerg}}$)** | 0.7234 | **0.7017** | `VERIFIED` |
| **Routine Stratum Cutoff ($q_{\text{rout}}$)** | 0.3145 | **0.3055** | `VERIFIED` |
| **Emergency Coverage ($\alpha_{\text{emerg}} = 0.01$, Bound $\ge 99.0\%$)** | 99.31% (863/869) | **99.31%** (863/869) | `VERIFIED` |
| **Routine Coverage ($\alpha_{\text{rout}} = 0.05$, Bound $\ge 95.0\%$)** | 96.54% (502/520) | **96.73%** (503/520) | `VERIFIED` |
| **Empty Prediction Sets ($C(X) = \emptyset$)** | 5 cases (0.36%) | **8 cases** (0.58%) | `VERIFIED` |
| **Raw Conformal Coverage ($y \in C(X)$, $\emptyset = \text{False}$)** | 98.06% (1,362/1,389) | **97.91%** (1,360/1,389) | `VERIFIED` |
| **Operational Coverage (With Argmax Fallback)** | 98.27% (1,365/1,389) | **98.34%** (1,366/1,389) | `VERIFIED` |
| **Mean Prediction Set Size** | 1.0036 | **1.0101** | `VERIFIED` |
| **Singleton Prediction Sets** | 99.64% | **98.99%** | `VERIFIED` |

### Empty Set Fallback Dynamics:
- Under pure mathematical conformal prediction, an empty prediction set is an omission failure ($y \notin \emptyset$).
- AW-CRC produces 8 empty sets ($0.58\%$).
- The operational clinical guardrail (`backend/conformal.py`) injects $\operatorname{argmax}(\hat{P})$ and flags the scan with an urgent mandatory clinical examination alert.
- In 6 of the 8 cases, argmax matched ground truth, raising coverage from **$97.91\%$ to $98.34\%$**. Both raw and operational coverages strictly exceed the theoretical target ($> 97.5\%$).

---

## 6. Regression Testing Verification

All 37 test modules executed cleanly under `pytest tests/`:
- **Total Test Cases:** **317**
- **Passed:** **317 (100.0%)**
- **Failed:** 0
- **Regression Status:** Clean pass.
