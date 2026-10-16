# OphthalmoAI: Clinical & Empirical Validation Protocol

**Document Version**: 1.0.0  
**Research Designation**: Prospective Multi-Cohort Evaluation Design for Investigational Screening  
**Status**: Academic Research Manuscript in Preparation (Non-Diagnostic Prototype)  

---

## 1. Objective & Scope

This protocol establishes the standardized, reproducible methodology for evaluating the diagnostic discrimination, post-hoc calibration, out-of-distribution robustness, and algorithmic fairness of OphthalmoAI's vision screening models across multi-institutional fundus photographic datasets.

> [!IMPORTANT]
> This protocol specifies an empirical research evaluation methodology. OphthalmoAI is not an FDA-cleared or CE-marked medical device. All findings must be corroborated by a licensed eye care professional.

---

## 2. Benchmark Dataset Cohorts

The evaluation protocol mandates validation over four independent clinical cohorts:

| Cohort Name | Source & Institution | Camera Hardware | Primary Indications | Inclusion Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **EyePACS** | EyePACS Foundation / Kaggle | Multi-camera field units | Diabetic Retinopathy, Normal | Posterior pole color fundus, $\ge 45^\circ$ FOV |
| **Messidor-2** | Messidor Consortium, France | Topcon TRC NW6 | Diabetic Retinopathy, Macular Edema | Centered on fovea, non-mydriatic |
| **APTOS 2019** | Aravind Eye Hospital, India | Multiple clinic cameras | Diabetic Retinopathy severity | Rural screening clinics across India |
| **IDRiD** | Eye Clinic, Nanded, India | Kowa VX-10 $\alpha$ digital | DR, Glaucoma, AMD | 50° field of view, mydriatic |

---

## 3. Data Hygiene & Patient-Level Splitting

To prevent data leakage and artificially inflated performance estimates:
1. **Subject-Level Partitioning**: All images originating from the same patient (both eyes, longitudinal visits) must be allocated strictly to either Train, Validation, or Test. Never split images of the same patient across splits.
2. **Perceptual Deduplication**: Execute perceptual hashing (pHash/dHash) across splits. Any pair with a Hamming distance $\le 4$ must be audited and isolated to a single partition.
3. **Disjoint Calibration Set**: Post-hoc Platt temperature scaling parameters must be learned exclusively on the holdout validation split and never on the test split.

---

## 4. Evaluation Metrics & Statistical Standards

### 4.1 Discrimination
- **Primary Metric**: Macro-averaged Area Under the Receiver Operating Characteristic (AUROC) with 1,000 bootstrap iterations for 95% Confidence Intervals.
- **Sensitivity & Specificity**: Evaluated at clinical operational thresholds (Youden index or fixed high-sensitivity screening threshold $\ge 95\%$).
- **Multi-Class Metrics**: Per-class precision, recall, and Macro-F1 score across target categories (`Normal`, `Diabetic Retinopathy`, `Glaucoma`, `Cataract`, `AMD`).

### 4.2 Calibration & Uncertainty
- **Expected Calibration Error (ECE)**: 15-bin equal-width reliability diagram integration before and after temperature scaling.
- **Brier Score**: Multi-class quadratic probability error.
- **Conformal Risk Control (AW-CRC)**: Adaptively weighted conformal prediction set evaluation targeting coverage guarantee $1 - \alpha = 0.95$. Report singleton prediction set proportion vs multi-label set proportion.

### 4.3 Fail-Safe Escalation Gate
- Evaluation of the automated human review gate (`requires_human_review: true`):
  - Triggered if confidence $< 75\%$.
  - Triggered if Monte Carlo epistemic variance $\sigma^2 > 0.15$.
  - Triggered if conformal prediction set cardinality $> 1$.
  - Triggered on non-canonical cropped views (e.g. RIM-ONE DL disc crops). Target escalation rate on out-of-distribution input: **$100.0\%$**.

---

## 5. Algorithmic Fairness & Subgroup Analysis

Models must be audited across demographic and hardware slices:
- Camera hardware sensor shifts (Topcon vs Kowa vs Canon).
- Exposure variances (low-light vs over-illuminated).
- Pupil dilation status (mydriatic vs non-mydriatic).
- Parity of False Negative Rates (FNR) across patient subgroups.
