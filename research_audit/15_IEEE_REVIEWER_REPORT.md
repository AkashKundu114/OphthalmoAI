# Phase 14: IEEE Conference Peer Review Report

**Conference:** 2027 IEEE International Conference on Cognitive Computing and Engineering Evaluation (ICCECE 2027)  
**Track:** Biomedical Informatics, Cognitive Systems & Trustworthy AI  
**Manuscript Title:** Trustworthy Point-of-Care Retinal Disease Screening: Biophysical Optical Guardrails, Calibrated Vision Ensembles, and Admissibility-Weighted Conformal Sets  
**Reviewer Role:** Senior Biomedical AI Researcher, Clinical AI Safety Auditor & IEEE Reviewer  
**Audit & Review Date:** October 30, 2026  
**Manuscript Artifact Audited:** `research/manuscript.tex` (25 Pages, IEEE Two-Column Format, `ieeecolor.cls`, 1,557 lines)  
**Codebase & Data Repositories:** `D:\OphthalmoAI` (Engineering/Models) & `D:\OphthalmoAI-Paper` (Research Workspace)  

---

## 1. Formal Review Verdict & Overall Recommendation

| Evaluation Metric | Assessment |
| :--- | :--- |
| **Overall Recommendation** | **REJECT (Desk Reject / Strong Reject)** |
| **Reviewer Confidence** | **5 / 5 (Expert in Biomedical Vision, Conformal Prediction, and Clinical AI Safety)** |
| **Primary Rejection Drivers** | 1. **Fatal Format Violation:** Manuscript is **25 pages long** (with 21 figures and 15 tables), exceeding the strict ICCECE 2027 8-page conference limit by $>300\%$.<br>2. **Methodological Invalidation:** Demographic fairness parity and multimodal synergy evaluated on synthetic EHR features conditioned directly on ground-truth disease labels.<br>3. **Internal Mathematical Contradiction:** Table 6 per-class sensitivities imply only 1,871.4 correct cases (83.21% accuracy), directly contradicting the claimed 2,019 correct cases (89.77% accuracy) on line 824.<br>4. **Catastrophic External Specificity Collapse:** Specificity collapses to 16.42% on IDRiD and 1.05% on RIM-ONE DL.<br>5. **Unsupported Grounding Claims:** PASG-GradCAM, Pointing Game benchmark, and 10% ophthalmologist audit are absent from repository code; spatial descriptions are generated via `random.choice()`. |

---

## 2. Detailed Numerical Scoring Grid (1 to 10 Scale)

| Evaluation Dimension | Score (1–10) | Forensic Review Justification |
| :--- | :---: | :--- |
| **1. Originality** | **6 / 10** | Shifting OOD detection to pre-inference biophysical optical invariants (telecentric mask, chromophore ratio) and weighting conformal prediction by continuous optical admissibility are sound ideas. However, the core vision pipeline is standard ensembling and Platt scaling. |
| **2. Technical Depth** | **7 / 10** | Extensive mathematical formulations spanning temperature calibration, biophysical optics, conformal risk control, and epistemic uncertainty. However, theoretical depth is undermined by missing implementations in code. |
| **3. Methodological Correctness** | **3 / 10** | **Severe flaws:** (1) Tabular EHR features were synthesized from distributions conditioned on the ground-truth disease label, creating circular label leakage in multimodal and fairness benchmarks. (2) Evidential meta-classifier script trains on simulated Gaussian noise rather than extracted image features. |
| **4. Statistical Validity** | **4 / 10** | Rigorous use of Wilson intervals and DeLong tests is commended. However, Table 6 contains fatal internal arithmetic contradictions (Macro Average AUROC printed as 0.9914 vs. true arithmetic mean 0.9839; 147.6 ghost correct cases unaccounted for). Multi-seed and DCA scripts run on hardcoded nominal values. |
| **5. Dataset Quality & Provenance** | **4 / 10** | Isolating the 35.96% patient leakage in the naive split is commendable. However, 16,140 rows in the master manifest (`fundus_cfe_`) have patient IDs generated via an artificial modulo heuristic (`num // 2`) rather than verified hospital patient IDs. Messidor-2, EyePACS, and REFUGE raw data directories are missing from the repository. |
| **6. Experimental Completeness** | **7 / 10** | Vast breadth of experimental evaluations (ablation waterfall, RETFound comparison, multi-seed runs, IDRiD/RIM-ONE external cohorts). However, several key benchmarks (2,500 negative stress corpus, Pointing Game, DCA) are hardcoded or simulated. |
| **7. Code Reproducibility** | **5 / 10** | Checkpoints (`convnext_small.pth`, `densenet201.pth`, etc.) and the master evaluation script (`reproduce_evaluation.py`) execute out-of-the-box. However, evidential training, clinical battery generation, and saliency grounding scripts do not reproduce the manuscript from raw data. |
| **8. Clinical Safety & Readiness** | **3 / 10** | **Unacceptable for clinical deployment:** Catastrophic external specificity collapse (16.42% on IDRiD, 1.05% on RIM-ONE DL) would overwhelm secondary clinics with false alarms. AW-CRC yields 3.14% empty prediction sets ($\mathcal{C}(X) = \emptyset$) without a documented clinical triage policy. |
| **9. Writing & Presentation Quality** | **7 / 10** | Professionally written, high typographic quality, impressive figures and tables. However, the manuscript suffers from severe over-claiming, conflates proposed database schemas with empirical ML results, and dramatically exceeds conference length. |
| **10. Literature & Reference Quality** | **8 / 10** | Comprehensive citations (55 bibliographic entries) covering primary retinal AI literature (Gulshan, Ting, Abràmoff, RETFound), conformal prediction (Vovk, Angelopoulos), and OOD baselines (Hendrycks, Liang). |
| **Composite Weighted Score** | **5.4 / 10** | **Substandard for publication in current form; requires fundamental scientific remediation and structural condensation.** |

---

## 3. Comprehensive Review Summary

The manuscript presents "OphthalmoAI," an ambitious point-of-care retinal disease screening framework aimed at diagnosing six posterior pole conditions (Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, and Hypertensive Retinopathy / Pathological Myopia) from color fundus photography. The proposed system integrates five methodological pillars:
1. A Temperature-Calibrated Multi-Backbone Ensemble (TC-MBE: DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M).
2. A deterministic pre-GPU Optical Aperture and Chromophore Domain Guardrail (OAC-DG) designed to reject non-fundus OOD inputs in 1.8 ms on CPU.
3. An Admissibility-Weighted Conformal Risk Control (AW-CRC) framework that dynamically adjusts prediction set sizes based on continuous optical quality $\Phi(X)$.
4. An Epistemic-Aleatoric Dual-Uncertainty Decomposition utilizing Monte Carlo dropout and evidential Dirichlet vacuity.
5. Pixel-Aligned Saliency Grounding (PASG-GradCAM) to constrain visual explainability to anatomical regions of interest (macula, optic disc).

The research addresses critical bottlenecks in medical computer vision: model overconfidence, optical domain shift, silent failure on OOD inputs, and the prohibitive compute footprints of modern vision foundation models (RETFound).

**However, in its present form, the manuscript cannot be accepted.** The paper suffers from a fatal page budget violation (25 pages vs. 8 allowable pages), severe methodological flaws (circular synthetic EHR data used for fairness and multimodal claims), fatal internal arithmetic contradictions in primary diagnostic tables, catastrophic external specificity collapse, and theoretical explainability claims that are completely unsupported by the repository codebase.

---

## 4. Major Reviewer Concerns (Fatal Flaws Requiring Resolution)

### 4.1 Concern 1: Unacceptable Page Budget & Scope Sprawl (25 Pages vs. 8 Pages)
- **Finding:** ICCECE 2027 operates under a strict **6 to 8 page limit** for regular conference papers. The submitted manuscript is **25 pages long** in two-column format, featuring 21 figures and 15 tables.
- **Critique:** The paper attempts to be an algorithmic ML paper, an optical physics paper, a multi-cohort clinical trial report, an enterprise database specification (PostgreSQL Row-Level Security), a cybersecurity compliance manual (HIPAA/GDPR), and an FDA regulatory roadmap simultaneously.
- **Mandatory Action:** The paper must be drastically pruned to 8 pages or redirected to a specialized high-capacity journal (e.g., Springer Nature *Health Information Science and Systems* or *IEEE Transactions on Medical Imaging*).

### 4.2 Concern 2: Circular Label Leakage in Synthetic EHR Bio-Data & Demographic Fairness
- **Finding:** Section IV-G (Table 8) presents a demographic fairness audit claiming Disparate Impact Ratios $\text{DIR} \ge 0.966$ across age cohorts ($<50$, $50-65$, $>65$), and Section IV-I claims "Multimodal Diagnostic Synergy" boosting accuracy from 85.18% to 87.42% by fusing images with 12 EHR biometric variables (age, sex, HbA1c, SBP, DBP, IOP, LogMAR).
- **Forensic Code Audit:** Inspection of `scripts/generate_patient_biodata.py` (lines 107–150) reveals that all tabular EHR features in `test_patient_clean.csv` were synthetically generated by sampling from Gaussian distributions **parameterized directly by the ground-truth disease class**:
  ```python
  # generate_patient_biodata.py lines 53-67
  # If class == "Diabetic Retinopathy": hba1c ~ N(8.6, 1.45), dm_duration ~ N(13.4, 6.2)
  # If class == "Glaucoma": iop ~ N(24.8, 4.6)
  # If class == "Hypertensive Retinopathy": sbp ~ N(162.0, 17.5)
  # If class == "Normal": hba1c ~ N(5.2, 0.35), sbp ~ N(118.0, 10.5)
  ```
- **Scientific Impact:** The EHR variables directly leak the target label. Any multimodal gain is an artifact of circular feature injection. Furthermore, auditing fairness across age slices using a synthetic, label-correlated `age` column evaluates a random number generator, not clinical algorithmic equity.
- **Mandatory Action:** Permanently retract all multimodal synergy claims and Table 8 fairness claims from the primary manuscript.

### 4.3 Concern 3: Fatal Mathematical Self-Contradictions in Table 6
- **Arithmetic Mean Contradiction:** Table 6 (line 880) reports the "Macro Average" AUROC as `\textbf{0.9914}`. However, taking the unweighted arithmetic mean of the six per-class AUROCs in that exact table yields:
  $$\frac{0.9642 + 0.9754 + 0.9871 + 0.9962 + 0.9924 + 0.9879}{6} = 0.983867 \approx 0.9839$$
  Four of the six individual diagnostic categories have AUROCs well below 0.9914. Reporting 0.9914 as the Macro Average is mathematically invalid without disclosing a specific weighting formula.
- **Contingency Discrepancy (147.6 Ghost Correct Cases):** Section IV-B (line 824) asserts:
  $$\text{Accuracy} = \frac{2,019}{2,249} = 89.7732\% \approx 89.77\%$$
  However, calculating true positive counts from Table 6 sensitivities yields:
  $$430(0.852) + 485(0.838) + 406(0.916) + 147(0.942) + 374(0.798) + 407(0.712) = 1,871.39 \text{ correct cases}$$
  $$\text{Implied Overall Accuracy} = \frac{1,871.39}{2,249} = 83.21\%$$
  There is an unexplained discrepancy of **147.6 correct cases**. The per-class sensitivities in Table 6 cannot produce an overall accuracy of 89.77%.

### 4.4 Concern 4: Catastrophic External Specificity Collapse
- **Finding:** In Section IV-J, the model is evaluated on two external clinical cohorts:
  - **IDRiD ($N=413$, India):** Sensitivity is $99.64\%$, but Specificity collapses to **$16.42\%$** (112 false-positive referrals out of 134 normal scans; false-positive rate $83.58\%$).
  - **RIM-ONE DL ($N=485$, Spain):** Graham AUROC is $0.6175$ and Reinhard AUROC is $0.6577$. Specificity collapses to **$1.05\%$ (Graham)** and **$2.81\%$ (Reinhard)** (277 to 282 false positives out of 285 normal eyes).
- **Clinical Critique:** The model exhibits severe optical over-referral under camera hardware domain shift (Kowa and Nidek optics). In a real-world primary care deployment, referring $>97\%$ of healthy patients to specialist ophthalmologists would overwhelm hospital capacity, causing severe clinical harm.
- **Mandatory Action:** The authors must eliminate promotional claims of "clinical readiness" or "autonomous triage." This failure mode must be elevated as a central safety finding demonstrating the limitations of deep learning under optical domain shift.

### 4.5 Concern 5: Theoretical Grounding vs. Physical Code Reality (PASG-GradCAM)
- **Finding:** Formulation 5 and Table 9 claim "Pixel-Aligned Saliency Grounding (PASG-GradCAM)" using an automated U-Net segmentation pipeline to enforce macular and disc energy ratios ($\eta \ge 0.40$), validated against an ophthalmologist audit on a 10% sample (Dice $0.91 \pm 0.04$), achieving 88.4% Pointing Game accuracy.
- **Forensic Code Audit:** 
  1. The repository contains NO U-Net segmentation model or weights.
  2. The repository contains NO Pointing Game benchmark script.
  3. The repository contains NO records, data files, or IRB protocols regarding an ophthalmologist audit.
  4. In `backend/services/explanation_service.py` (lines 66–96), the anatomical lesion description is literally generated by:
     ```python
     quadrants = ["superotemporal macular arcade", "inferonasal quadrant near optic disc", "foveal center", ...]
     return f"microaneurysms, dot-blot hemorrhages... localized along the {random.choice(quadrants)}"
     ```
     Anatomical quadrant descriptions are generated via `random.choice()`.
- **Mandatory Action:** Table 9 and all claims of a verified U-Net grounding pipeline and ophthalmologist audit must be removed or explicitly disclosed as an unverified conceptual proposal.

### 4.6 Concern 6: Conformal Prediction Set Emptiness (3.14% Null Sets)
- **Finding:** Table 8 and Section IV-G celebrate that AW-CRC maintains $96.86\%$ singleton prediction sets while restoring coverage on Grade C media.
- **Forensic Audit:** `models/aw_crc_calibration.json` records:
  - `"mean_set_size": 0.9686450167973124`
  - `"singletons_pct": 96.86450167973123`
  - `"overall_cov": 0.9686450167973124`
- **Critique:** A mean set size strictly below 1.0 proves that **3.14% of patients receive an empty prediction set ($\mathcal{C}(X) = \emptyset$)**. In clinical screening, outputting an empty set leaves an attending clinician with zero diagnostic guidance. The manuscript conceals this empty set phenomenon.
- **Mandatory Action:** Disclose the 3.14% empty set rate and formalize the clinical triage protocol for null conformal sets (mandatory immediate escalation to manual human examination).

### 4.7 Concern 7: Aspirational Enterprise & Regulatory Over-Claiming
- **Finding:** Section III-C presents a SQL snippet claiming enterprise PostgreSQL Row-Level Security (RLS) enforcement for HIPAA/GDPR compliance.
- **Forensic Audit:** Zero RLS policies exist in Alembic migrations or backend code. The local deployment runs on SQLite, which does not support PostgreSQL RLS. Claiming certified HIPAA compliance for an experimental prototype violates scientific integrity standards.

---

## 5. Minor Reviewer Concerns

1. **Parameter Accounting Inconsistency:** Section IV-K (line 1141) states OphthalmoAI uses 120.5M parameters; Table 11 states 124.3M parameters. Reconcile the exact parameter volume across constituent backbones and meta-classifier.
2. **Monte Carlo Dropout Passes:** Algorithm 2 (line 492) specifies $S=20$ forward passes; `backend/uncertainty.py` (line 18) defaults to `n_passes: int = 8`. Reconcile this operational discrepancy.
3. **Hardcoded Benchmark Scripts:** `scripts/evaluate_extended_clinical_battery.py` and `scripts/evaluate_multi_seed.py` construct evaluations using hardcoded arrays and simulated beta distributions rather than executing live tensor passes. Refactor these scripts to evaluate genuine model outputs.
4. **Figure Color Gradients:** While the figures are visually stunning, several multi-line plots (e.g., Fig. 10 ROC curves) become illegible when printed in standard black-and-white. Ensure all curves have distinct dashing patterns.

---

## 6. Missing Experiments Required for Resubmission

1. **True Cross-Camera Domain Adaptation:** Evaluate unsupervised domain adaptation (e.g., AdaBN, CycleGAN, or contrastive sensor alignment) on IDRiD and RIM-ONE DL to determine if specificity can be restored from 16% to $>85\%$.
2. **Empirical Grounding Sanity Check:** Implement Adebayo et al.'s model parameter randomization test to verify whether the EfficientNet-B4 Grad-CAM heatmaps reflect learned disease features or merely edge detection.
3. **Empirical Empty-Set Escalation:** Formulate and benchmark an explicit conformal fallback rule ensuring $\mathcal{C}(X) \neq \emptyset$, evaluating its impact on singleton efficiency.
4. **Genuine Multimodal Clinical Study:** If multimodal synergy is to be claimed, evaluate a real-world clinical dataset with authentic, non-synthetic patient bio-data (e.g., UK Biobank retinal cohort).

---

## 7. Eight-Page Conference Template & Page Budget Restructuring

To transform this 25-page compendium into a competitive, rigorous **8-page IEEE conference paper for ICCECE 2027**, the authors must execute the following surgical structural reorganization:

### Page-by-Page Allocation Plan (Strict 8-Page IEEE Budget)

```
====================================================================================================
Page  Target Section & Content Scope                                     Figures / Tables Included
====================================================================================================
Page 1: Title, Abstract, Introduction & Operational Motivation            Fig. 1 (End-to-End Pipeline)
        - Frontline screening bottlenecks in low-resource settings.
        - Core contributions: Guardrail, Ensemble, AW-CRC.
----------------------------------------------------------------------------------------------------
Page 2: Related Work & Mathematical Problem Formulation                   None (Text & Formulas)
        - Related Work: IDx-DR, RETFound, Conformal Prediction.
        - Formulation 1: Temperature-Calibrated Multi-Backbone Ensemble.
        - Formulation 2: Optical Aperture & Chromophore Guardrail (OAC-DG).
----------------------------------------------------------------------------------------------------
Page 3: Conformal Risk Control & Dual Uncertainty Formulation             Alg. 1 (AW-CRC Inference)
        - Formulation 3: Admissibility-Weighted Conformal Risk Control.   Formula block (AW Quantiles)
        - Formulation 4: Epistemic-Aleatoric Uncertainty Decomposition.
----------------------------------------------------------------------------------------------------
Page 4: Benchmark Cohort, Leakage Audit & Experimental Setup              Table 1 (Clean vs Leaked Split)
        - Multi-source cohort composition (ODIR-5K, APTOS, etc.).         Fig. 2 (Architecture Pareto)
        - Contralateral patient-level isolation & leakage proof.
        - Training parameters, FP16 execution, hardware setup.
----------------------------------------------------------------------------------------------------
Page 5: Empirical Classification Benchmarks & Guardrail Evaluation        Table 2 (Model Benchmarks)
        - 6-class diagnostic results (89.77% Top-1, 0.9839/0.9914 AUROC). Table 3 (OOD Guardrail Rejection)
        - Guardrail evaluation: 97.84% OOD rejection, 1.8 ms CPU latency.
----------------------------------------------------------------------------------------------------
Page 6: Conformal Risk Control Benchmarks & RETFound Comparison           Table 4 (AW-CRC Coverage Tiers)
        - AW-CRC coverage restoration on Grade C media (97.59%).          Fig. 3 (RETFound Comparison)
        - Head-to-head empirical comparison vs. RETFound ViT-Large.
----------------------------------------------------------------------------------------------------
Page 7: External Validation (IDRiD / RIM-ONE) & Safety Analysis           Fig. 4 (External Transfer / Shift)
        - Transparent disclosure of external specificity collapse.        Table 5 (Component Ablation)
        - Role of conformal prediction in mitigating domain shift.
        - Stepwise component subtraction ablation waterfall.
----------------------------------------------------------------------------------------------------
Page 8: Discussion, Deployment Constraints, Conclusion & References       ~25 Core References
        - Edge inference economics (84 ms GPU / sub-50 ms WASM).
        - Limitations: Empty conformal sets, camera sensor dependence.
        - References 1–25 (rigorously selected, compressed format).
====================================================================================================
```

### Content Eviction Ledger (What Must Move to Supplementary Material)

1. **Evict Enterprise Database Architecture (Sec. III-C, Fig. 7):** Remove PostgreSQL Row-Level Security schemas and SQL listings completely from the conference paper.
2. **Evict Human-in-the-Loop Triage Diagrams (Sec. III-D, Fig. 8):** Move theoretical active learning workflows to supplementary materials.
3. **Evict PASG-GradCAM Benchmark (Sec. IV-H, Table 9):** Remove Table 9 and Pointing Game claims until U-Net segmentation code is physically committed.
4. **Evict Synthetic EHR / Multimodal Claims (Sec. IV-I, Table 8):** Completely delete multimodal fusion and demographic fairness claims derived from synthetic bio-data.
5. **Evict Decision Curve Analysis & Likelihood Ratio Battery (Sec. IV-I, Table 10, Fig. 18):** Move secondary epidemiological metrics (LR+, LR-, DOR, DCA net benefit) to supplementary materials.
6. **Evict Redundant Figures:** Reduce figures from 21 down to 4 high-impact figures (Pipeline Overview, Architecture Pareto, RETFound Comparison, External Domain Shift).

---

## 8. Final Reviewer Recommendation

**Verdict: REJECT (Desk Reject / Strong Reject).**  
While the paper demonstrates strong engineering capability and creative formulations around biophysical optical guardrails and conformal risk control, its fatal format violation (25 pages vs. 8 pages), circular synthetic EHR evaluation, internal mathematical contradictions in primary diagnostic tables, and unsupported explainability claims preclude publication at ICCECE 2027.

The authors are strongly encouraged to execute the restructuring plan outlined in Section 7, resolve the mathematical and data provenance issues, and resubmit either as a focused 8-page paper to a future IEEE conference or as a comprehensive full-length article to a specialized journal such as *Health Information Science and Systems* or *IEEE Transactions on Medical Imaging*.
