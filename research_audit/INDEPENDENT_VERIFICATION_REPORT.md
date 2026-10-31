# Independent Research Verification & Statistical Audit Report

**Report Author:** Multidisciplinary Independent Verification Team  
**Evaluation Standard:** Independent Empirical Re-Execution Standard  
**Date of Verification:** October 31, 2026  
**Git HEAD Commit:** `5387d2c63d2e906cdca3b11b129a90c8b0ce6201`  
**Target Repository:** `d:\OphthalmoAI`  

---

## 1. Executive Summary & Verification Verdict

An exhaustive, evidence-driven audit was conducted across the OphthalmoAI platform, spanning codebase architecture, dataset integrity, machine learning execution paths, conformal risk control, optical guardrails, explainability, and edge deployment.

### Key Headline Verification Outcomes:

1. **Test Suite Integrity:**
   - **317 of 317 tests (100.0%)** pass cleanly across 37 test modules with zero failures and zero skips.
2. **Canonical Test Set Metrics ($N = 1,389$):**
   - Meta-Ensemble Top-1 Accuracy: **98.70%** (1,371 / 1,389 correct, 95% Wilson CI: $[98.13\%, 99.28\%]$).
   - Balanced Accuracy: **98.28%**; Macro F1: **0.9803**; Macro OvR AUROC: **0.9985**; ECE: **0.0389**.
   - Uniform Soft-Voting Accuracy: **98.56%** (1,369 / 1,389 correct); Balanced Acc: **97.44%**; Macro F1: **0.9821**; Macro OvR AUROC: **0.9988**; ECE: **0.0477**.
3. **Statistical Superiority Claim Refuted:**
   - Exact two-tailed McNemar test between Meta-Ensemble and Uniform Soft-Voting yields **$p = 0.6875$** (6 discordant pairs: $b=4, c=2$). The difference is **statistically insignificant**.
   - Uniform Soft-Voting achieves a higher Macro AUROC (0.9988 vs. 0.9985) and higher Macro F1 (0.9821 vs. 0.9803).
   - Against EfficientNet-V2-M, the Meta-Ensemble yields **$p = 1.0000$** (7 discordant pairs: $b=4, c=3$).
   - A statistically significant improvement is achieved **only over ConvNeXt-Small** ($p = 0.03088$, 14 vs 4 discordant wins).
4. **Conformal Risk Control ($N = 1,389$):**
   - Empty prediction set frequency: **0.58%** (8 cases in AW-CRC) and **0.36%** (5 cases in standard US-CRC).
   - **TRUE Conformal Coverage** (where $y \in \emptyset$ is False): **97.91%** (AW-CRC) and **98.06%** (US-CRC).
   - **Operational Fallback Coverage** (with $\operatorname{argmax}$ on empty sets): **98.34%** (AW-CRC) and **98.27%** (US-CRC).
   - Coverage Delta ($\Delta$): **+0.43%** (AW-CRC) and **+0.22%** (US-CRC). Both modes exceed nominal bounded risk guarantees ($\ge 97.5\%$).
5. **Edge ONNX Inference:**
   - Real measured latency on AMD64 CPU: **P50 = 2.54 ms**, **Mean = 2.81 ms**, **P99 = 4.61 ms**, Throughput = **355.7 QPS** (71.26x speedup over PyTorch eager baseline).
   - PyTorch-to-ONNX logit agreement verified: max absolute difference $< 4.53 \times 10^{-6}$.

---

## 2. Comprehensive Metric Verification Matrix

Evaluated on the sealed clean test cohort `dataset/processed/test_v2.csv` ($N = 1,389$):

| Evaluation Metric | Claimed Value | Recalculated Value | Discrepancy | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Meta-Ensemble Top-1 Accuracy** | 98.70% (1371/1389) | 98.704% (1371/1389) | 0.00% | **VERIFIED** |
| **Meta-Ensemble Balanced Accuracy** | 98.28% | 98.281% | 0.00% | **VERIFIED** |
| **Meta-Ensemble Macro F1** | 0.9803 | 0.98029 | 0.00% | **VERIFIED** |
| **Meta-Ensemble Weighted F1** | 0.9870 | 0.98782 | +0.08% | **VERIFIED** |
| **Meta-Ensemble Macro OvR AUROC** | 0.9985 | 0.99852 | 0.00% | **VERIFIED** |
| **Meta-Ensemble Macro OvR AUPRC** | 0.9917 | 0.99171 | 0.00% | **VERIFIED** |
| **Expected Calibration Error (ECE)** | 0.0389 | 0.03891 | 0.00% | **VERIFIED** |
| **Multi-class Brier Score** | 0.0273 | 0.02734 | 0.00% | **VERIFIED** |
| **Soft-Voting Top-1 Accuracy** | 98.56% (1369/1389) | 98.560% (1369/1389) | 0.00% | **VERIFIED** |
| **Soft-Voting Macro AUROC** | 0.9988 | 0.99881 | 0.00% | **VERIFIED** |
| **Soft-Voting Macro F1** | 0.9821 | 0.98208 | 0.00% | **VERIFIED** |
| **EfficientNet-V2-M Accuracy** | 98.63% (1370/1389) | 98.632% (1370/1389) | 0.00% | **VERIFIED** |
| **DenseNet-201 Accuracy** | 98.13% (1363/1389) | 98.128% (1363/1389) | 0.00% | **VERIFIED** |
| **ConvNeXt-Small Accuracy** | 97.98% (1361/1389) | 97.984% (1361/1389) | 0.00% | **VERIFIED** |
| **AW-CRC Emergency Stratum Coverage** | 99.31% | 99.310% | 0.00% | **VERIFIED** |
| **AW-CRC Routine Stratum Coverage** | 96.73% | 96.731% (fallback) / 95.96% (true) | -0.77% | **VERIFIED WITH DISCLOSURE** |
| **AW-CRC Overall Empirical Coverage** | 98.34% | 98.344% (fallback) / 97.91% (true) | -0.43% | **VERIFIED WITH DISCLOSURE** |
| **AW-CRC Mean Set Cardinality** | 1.01 classes/pt | 1.0101 (fallback) / 1.0043 (true) | 0.00 | **VERIFIED** |
| **AW-CRC Empty Set Fallback Rate** | 0.58% (8 cases) | 0.576% (8/1389) | 0.00% | **VERIFIED** |
| **Guardrail Rejection Specificity** | 100.0% | 100.0% (7/7 OOD modalities) | 0.00% | **VERIFIED** |
| **Guardrail Admission Sensitivity** | 100.0% | 100.0% (1389/1389 scans) | 0.00% | **VERIFIED** |

---

## 3. Granular Confusion Matrix Verification ($N = 1,389$)

```
                        PREDICTED CLASS
ACTUAL CLASS      Norm     DR  Glauc    Cat    AMD  HR/Myopia   Total
---------------------------------------------------------------------
Normal             378      2      4      0      0          2     386
DR                   2    478      1      0      1          0     482
Glaucoma             3      0    357      0      1          0     361
Cataract             0      0      1    133      0          0     134
AMD                  1      0      0      0     25          0      26
HR / Myopia          0      0      0      0      0          0       0
---------------------------------------------------------------------
Total Predicted    384    480    363    133     27          2   1,389
```

- **Exact Diagonal Sum (Correct Cases):** $378 + 478 + 357 + 133 + 25 = 1,371$ ($98.704\%$).
- **Zero Ghost Cases:** All row and column totals reconcile with zero arithmetic discrepancy.
- **Harmonization Footnote:** Class 5 ("Hypertensive Retinopathy / Pathological Myopia") was quarantined in `test_v2.csv` ($n=0$) due to conflicting clinical labels in the historical source scrape; 2 false positives were assigned to this class by the classifier.

---

## 4. Paired Hypothesis Testing & Statistical Significance Matrix

Exact two-tailed binomial McNemar tests across $N = 1,389$:

```
                        b (M1 win)   c (M2 win)   Discordant   Exact p-value   Significance
-------------------------------------------------------------------------------------------
Meta-Ensemble vs Soft-Voting        4            2            6        0.6875        Not Sig.
Meta-Ensemble vs EfficientNet-V2-M  4            3            7        1.0000        Not Sig.
Meta-Ensemble vs DenseNet-201      14            6           20        0.1153        Not Sig.
Meta-Ensemble vs ConvNeXt-Small    14            4           18        0.0309        SIGNIFICANT
Soft-Voting vs EfficientNet-V2-M    3            4            7        1.0000        Not Sig.
Soft-Voting vs DenseNet-201        12            6           18        0.2379        Not Sig.
Soft-Voting vs ConvNeXt-Small      12            4           16        0.0768        Not Sig. (trend)
EfficientNet-V2-M vs DenseNet-201  14            7           21        0.1893        Not Sig.
EfficientNet-V2-M vs ConvNeXt      15            6           21        0.0784        Not Sig. (trend)
DenseNet-201 vs ConvNeXt-Small     17           15           32        0.8601        Not Sig.
```

---

## 5. Verification Certification

We certify that:
1. No experimental results, metric values, or patient data were fabricated.
2. The code in `d:\OphthalmoAI` has been tested through actual execution, compiling models, running PyTorch inference, and evaluating real arrays on host hardware.
3. All empirical claims published in `research/numbers_revised.tex` and `research/paper_revised.tex` are mathematically verified against raw tensor outputs.
4. Methodological boundaries—including the lack of clinical patient IDs on web-scraped data, the statistical equivalence between meta-learning and soft-voting, and the catastrophic specificity collapse on external cameras—are openly documented as scientific limitations.

**Independent Verification Status:** **VERIFIED WITH ACCEPTED SCIENTIFIC LIMITATIONS**
