# Second-Pass Independent Adversarial Review & Disproof Investigation

**Document ID:** `research_audit/SECOND_PASS_ADVERSARIAL_REVIEW.md`  
**Audit Date:** October 30, 2026  
**Auditor Role:** Independent External Reviewer & Adversarial AI Safety Inspector  
**Objective:** Disprove research claims, uncover methodological weaknesses, identify data leakage, detect benchmark contamination, and expose unsupported assumptions missed by the initial audit.  
**Audited Repositories:** [`D:\OphthalmoAI`](file:///d:/OphthalmoAI) (Engineering Codebase) & [`D:\OphthalmoAI-Paper`](file:///d:/OphthalmoAI-Paper) (Research Manuscript)  
**Evidence Standardization:** VERIFIED, PARTIALLY VERIFIED, UNVERIFIED, CONTRADICTED, INVALID, PROPOSED  

---

## 1. Executive Adversarial Synthesis

The initial research audit successfully uncovered critical vulnerabilities, including historical patient leakage in `test.csv` ($n = 938$), circular label leakage in synthetic tabular EHR features (`scripts/generate_patient_biodata.py`), phantom PASG-GradCAM explainability, and external specificity collapse on IDRiD and RIM-ONE DL.

However, the initial audit **failed to be sufficiently skeptical** in several critical domains. It accepted the premise that the revised split (`test_patient_clean.csv`, $n = 2,249$) was genuinely isolated, trusted the foundation model benchmark against RETFound, overlooked client-side silent diagnostic fallbacks, and accepted the mathematical formulation of Admissibility-Weighted Conformal Risk Control (AW-CRC) without exposing that prediction sets **never adaptively expand** and generate **3.14% empty sets**.

This second-pass adversarial review overturns several conclusions of the initial audit and establishes ten major previously overlooked vulnerabilities:

```
+------------------------------------------------------------------------------------------------------------------+
|                                    KEY ADVERSARIAL DISCOVERIES AT A GLANCE                                       |
+------------------------------------------------------------------------------------------------------------------+
| 1. Pseudo-Patient Grouping Artifact:                                                                            |
|    `scripts/split_patient_level.py` grouped images using `num // 2` on row indices of a randomly shuffled        |
|    dataframe, pairing unrelated images and scattering true bilateral eyes across train and test.                |
| 2. Pseudo-Split in Hugging Face Ingestion:                                                                       |
|    In `scripts/ingest_color_fundus_eye.py`, all 16,140 images (72.1% of the master corpus) were assigned        |
|    unique 1-to-1 dummy IDs (`pt_cfe_XXXXX`), converting the "patient split" into a pure random image split.     |
| 3. Cross-Split Exact Duplicates with Contradictory Labels:                                                       |
|    Cryptographic MD5 hashing discovered 157 EXACT DUPLICATE IMAGES between train and test in `clean` split;       |
|    many identical images have conflicting labels (e.g. Glaucoma in Train, AMD in Test).                         |
| 4. Disadvantaged Foundation Model Benchmark:                                                                     |
|    RETFound ViT-Large was trained on 224x224 images (vs 384x384 for OphthalmoAI; 2.94x fewer pixels) and       |
|    evaluated with a 1,000-sample bootstrap difference falsely labeled as a "DeLong test".                       |
| 5. Silent Heuristic Fallback in Frontend:                                                                       |
|    When ONNX WebAssembly fails in `frontend/src/edgeInference.js`, the app silently guesses diseases using     |
|    arbitrary RGB pixel ratios (e.g., contrastRatio > 1.12 -> Glaucoma, meanBlue > 80 -> Cataract).               |
| 6. Silent Fail-Open Exception Handling in Guardrail:                                                             |
|    `backend/routes/screening.py` catches any `validate_fundus_image` exception with a warning and allows        |
|    unvalidated inputs directly into the deep neural ensemble.                                                    |
| 7. Synthetic Vector Search CBMIR:                                                                                |
|    `backend/vector_search.py` populates case retrieval using random Gaussian noise (`np.random.randn(512)`),   |
|    not real clinical image embeddings.                                                                           |
| 8. Conformal Set Inelasticity & Empty Set Concealment:                                                           |
|    In `aw_crc_calibration.json`, overall coverage, mean set size, and singleton percentage are identical        |
|    (0.968645): sets never expand under uncertainty, 3.14% are empty, and figures were hardcoded to hide it.    |
| 9. Background Aperture Distortion in Reinhard Color Constancy:                                                  |
|    `apply_reinhard` includes black background border pixels in mean/std calculation, blowing out the retinal     |
|    center into overexposed noise and degrading IDRiD AUROC by -0.0775.                                           |
| 10. Residual 12-Class Prototype Remnants in Production Backend:                                                  |
|     `backend/routes/screening.py` hardcodes "Monolithic 12-Class Model" and "85.2% Test Accuracy".              |
+------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Step 1 — Challenging the Initial Audit Itself

| Initial Audit Finding | First-Audit Evidence | Adversarial Challenge & Re-Investigation | Revised Status |
|---|---|---|:---:|
| **Clean Patient Partitioning ($n=2,249$)** | `scripts/split_patient_level.py` reported zero patient intersection. | **INVALID.** `num // 2` was applied to indices of a shuffled dataframe (`prepare_fundus_dataset.py`), pairing unrelated images. In `ingest_color_fundus_eye.py`, 16,140 images were given unique dummy IDs (`pt_cfe_XXXXX`). 157 exact MD5 duplicate images cross between train and test. | **CONTRADICTED / INVALID** |
| **Superiority over RETFound (+6.74%)** | `models/retfound_comparison_report.json` reported 89.77% vs 83.03% ($p=2.29\times 10^{-15}$). | **PARTIALLY INVALID.** RETFound was evaluated at $224 \times 224$ resolution while OphthalmoAI used $384 \times 384$ ($2.94\times$ pixel advantage). The reported "DeLong test" was actually a 1,000-sample bootstrap difference. | **PARTIALLY VERIFIED (UNFAIR PROTOCOL)** |
| **AW-CRC Dynamic Adaptation** | Manuscript claimed prediction sets expand to size 2.18 under optical degradation. | **CONTRADICTED.** In `models/aw_crc_calibration.json`, `mean_set_size == singletons_pct == 0.968645`. Sets **never** expand to size $\ge 2$. Set size metrics in `generate_all_manuscript_figures.py` line 944 were hardcoded. | **CONTRADICTED / INVALID** |
| **Biophysical Guardrail Intercept** | 7/7 unit tests passed; 0.00% False Rejection Rate. | **PARTIALLY VERIFIED.** `backend/routes/screening.py` lines 205–207 catch exceptions in `validate_fundus_image` and silently fail open, bypassing the guardrail. | **PARTIALLY VERIFIED (FAIL-OPEN BYPASS)** |
| **Reinhard Color Constancy as Domain Adaptation** | Script documented $\Delta\text{AUROC} = +0.0402$ on RIM-ONE DL. | **FLAWED ALGORITHM.** `apply_reinhard` computes mean/std across the entire image including black border pixels, blowing out the retinal center and degrading IDRiD AUROC by $-0.0775$. | **INVALID IMPLEMENTATION** |

---

## 3. Step 2 — Overlooked Problems Ledger (10 Critical Discoveries)

### Discovery ADV-01: Pseudo-Patient Allocation on Randomly Shuffled Rows
- **Affected File:** [`scripts/split_patient_level.py#L71-L86`](file:///d:/OphthalmoAI/scripts/split_patient_level.py#L71-L86) & [`scripts/prepare_fundus_dataset.py#L182-L187`](file:///d:/OphthalmoAI/scripts/prepare_fundus_dataset.py#L182-L187)
- **Mechanism:** In `prepare_fundus_dataset.py`, the dataframe was shuffled:
  ```python
  balanced_df = pd.concat(dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
  for idx, row in balanced_df.iterrows():
      out_name = f"fundus_{row['source']}_{idx:05d}.jpg"
  ```
  Original patient filenames (`0_left.jpg`, `0_right.jpg`) were discarded and replaced with `idx:05d`. In `split_patient_level.py`, patient IDs were assigned via:
  ```python
  num = int(''.join(filter(str.isdigit, img_id)))
  patient_id = f"odir_patient_{num // 2}"
  ```
- **Consequence:** `num // 2` paired two unrelated images from the shuffled dataframe into a fictitious patient. The true bilateral eyes of original patients were assigned different row indices, given different dummy patient IDs, and scattered across train, val, and test splits.

---

### Discovery ADV-02: 157 Cross-Split Exact Duplicate Images with Conflicting Labels
- **Affected Manifests:** [`dataset/processed/train_patient_clean.csv`](file:///d:/OphthalmoAI/dataset/processed/train_patient_clean.csv) and [`dataset/processed/test_patient_clean.csv`](file:///d:/OphthalmoAI/dataset/processed/test_patient_clean.csv)
- **Mechanism:** Ingestion of `Peacein/color-fundus-eye` imported duplicated files across different pathology folders. Because each image was assigned a unique ID (`pt_cfe_XXXXX`), duplicates were placed in both train and test.
- **Empirical Evidence (Exact MD5 Collisions):**
  - **Total Cross-Split Collisions:** **157 exact MD5 duplicate images** between Train and Test; **153** between Train and Val.
  - **Conflicting Label Examples:**
    * MD5 `998c85f6dcaffe0ca8c66e4d77323db2`: `fundus_cfe_05680.jpg` in Train (**Glaucoma**) vs `fundus_cfe_12476.jpg` in Test (**AMD**).
    * MD5 `de44e34f54ad039f710801ebd499bcf3`: `fundus_cfe_06729.jpg` in Train (**Normal**) vs `fundus_cfe_14523.jpg` in Test (**Glaucoma**).
    * MD5 `8070fcda08d06a57b148cda3c256337e`: `fundus_cfe_03997.jpg` in Train (**Glaucoma**) vs `fundus_cfe_10197.jpg` in Test (**Hypertensive Retinopathy**).
    * MD5 `10ca761bc087e7ab4fa5bf54f78ff7b2`: `fundus_cfe_14301.jpg` in Train (**Glaucoma**) vs `fundus_cfe_09088.jpg` in Test (**AMD**).
- **Scientific Impact:** The test set is contaminated with identical training images labeled as different diseases, introducing label noise and evaluating memorized features.

---

### Discovery ADV-03: Arbitrary Color-Heuristic Diagnostic Fallback in Client-Side Frontend
- **Affected File:** [`frontend/src/edgeInference.js#L288-L301`](file:///d:/OphthalmoAI/frontend/src/edgeInference.js#L288-L301) and [`frontend/src/edgeHeuristic.js#L287-L295`](file:///d:/OphthalmoAI/frontend/src/edgeHeuristic.js#L287-L295)
- **Mechanism:** If the ONNX WebAssembly model fails to load or execute in the browser, the frontend silently executes `runHeuristicInference`:
  ```javascript
  const rawScores = {
    "Normal": 1.00 + (vascularGreenDominance >= 0.45 && vascularGreenDominance <= 0.60 && contrastRatio >= 0.95 && contrastRatio <= 1.10 ? 1.5 : 0),
    "Diabetic Retinopathy": 1.00 + (meanGreen < 68 && vascularGreenDominance < 0.45 ? 1.6 : 0),
    "Glaucoma": 1.00 + (contrastRatio > 1.12 ? 1.6 : 0),
    "Cataract": 1.00 + (meanBlue > 80 || (meanGreen > 95 && contrastRatio < 0.98) ? 1.7 : 0),
    "Age-related Macular Degeneration": 1.00 + (contrastRatio < 0.92 ? 1.5 : 0),
    "Hypertensive Retinopathy": 1.00 + (vascularGreenDominance < 0.38 ? 1.4 : 0),
  };
  ```
- **Scientific Impact:** Diagnosing glaucoma based on `contrastRatio > 1.12` or cataract based on `meanBlue > 80` is medically dangerous. Serving heuristic fallback diagnoses without a blocking error modal violates clinical safety.

---

### Discovery ADV-04: Silent Fail-Open Exception Handling in Backend Guardrail
- **Affected File:** [`backend/routes/screening.py#L179-L207`](file:///d:/OphthalmoAI/backend/routes/screening.py#L179-L207)
- **Mechanism:**
  ```python
  try:
      is_fundus, fundus_score, fundus_reason, fundus_metrics = validate_fundus_image(image)
      if not is_fundus:
          raise HTTPException(status_code=422, ...)
  except HTTPException:
      raise
  except Exception as val_err:
      logger.warning("predict.fundus_validator_error", error=str(val_err))
  ```
- **Consequence:** If an adversarial or corrupted input triggers any exception in `validate_fundus_image` (e.g. division by zero, OpenCV memory error, dimension overflow), execution enters the broad `except Exception` block, logs a warning, and **silently continues execution into the deep ensemble**. The guardrail fails open rather than failing closed.

---

### Discovery ADV-05: False DeLong Test Attribution in Foundation Model Benchmark
- **Affected File:** [`scripts/finetune_retfound_comparison.py#L27-L32`](file:///d:/OphthalmoAI/scripts/finetune_retfound_comparison.py#L27-L32)
- **Mechanism:**
  ```python
  def compute_delong_pvalue(preds1, preds2, targets):
      """
      Computes p-value for the difference between two ROC AUCs using DeLong's test.
      Fallback to a simple bootstrap if multi-class is tricky, but here we do 1vRest DeLong for macro.
      For simplicity in this comparison, we'll do bootstrap for macro AUC difference.
      """
      n_bootstraps = 1000
      ...
  ```
- **Consequence:** The manuscript reports: *"Statistically significant improvement over RETFound confirmed by DeLong's test ($p = 2.29 \times 10^{-15}$)."* In reality, the code executed an empirical 1,000-iteration bootstrap resampling difference and named the function `compute_delong_pvalue`. Misattributing statistical tests in peer-reviewed text is a research integrity violation.

---

### Discovery ADV-06: Resolution Disparity Disadvantaging RETFound ViT-Large
- **Affected File:** [`scripts/finetune_retfound_comparison.py#L133-L147`](file:///d:/OphthalmoAI/scripts/finetune_retfound_comparison.py#L133-L147)
- **Mechanism:** RETFound was instantiated with `timm.create_model('vit_large_patch16_224')` and fed images at **$224 \times 224$ resolution**, while OphthalmoAI's ensemble was fed images at **$384 \times 384$ resolution**.
- **Impact:** $384 \times 384$ contains **$2.94\times$ more pixels** than $224 \times 224$. Ophthalmic microaneurysms, hard exudates, and fine vessels are severely degraded at $224 \times 224$. Evaluating the competitor model at downsampled resolution while evaluating one's own model at high resolution creates an unfair baseline comparison.

---

### Discovery ADV-07: Black Border Distortion in Reinhard Color Constancy
- **Affected File:** [`scripts/evaluate_external_dataset.py#L70-L84`](file:///d:/OphthalmoAI/scripts/evaluate_external_dataset.py#L70-L84)
- **Mechanism:**
  ```python
  def apply_reinhard(img_np: np.ndarray) -> np.ndarray:
      lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
      src_means = np.mean(lab, axis=(0, 1))
      src_stds = np.maximum(np.std(lab, axis=(0, 1)), 1e-4)
      target_means = np.array([128.5, 142.0, 153.2], dtype=np.float32)
      ...
  ```
- **Flaw:** `src_means` and `src_stds` are computed across all pixels, including the **black circular background mask** ($L=0$). Images with larger black borders have artificially depressed mean luminance. Shifting the distribution to the hardcoded `target_means[0] = 128.5` severely overamplifies the central retina, blowing out vessels into saturated white noise. This explains why Reinhard normalization degraded IDRiD AUROC by $-0.0775$.

---

### Discovery ADV-08: Synthetic Random Vectors in CBMIR Case Retrieval
- **Affected File:** [`backend/vector_search.py#L83-L97`](file:///d:/OphthalmoAI/backend/vector_search.py#L83-L97)
- **Mechanism:**
  ```python
  def _generate_synthetic_class_embedding(class_index: int, dim: int = 512, seed: int | None = None) -> np.ndarray:
      s = (class_index * 1337 + 42) if seed is None else seed
      rng = np.random.RandomState(s)
      base = rng.randn(dim).astype(np.float32) * 0.15
      subspace_size = max(10, dim // len(TARGET_CLASSES))
      start = (class_index % len(TARGET_CLASSES)) * subspace_size
      base[start : start + subspace_size] += 6.0
      norm = float(np.linalg.norm(base))
      return base / max(norm, 1e-6)
  ```
- **Consequence:** The Content-Based Medical Image Retrieval (CBMIR) service does **not** index or search deep latent embeddings extracted from real patient fundus images. It computes cosine similarities against artificially constructed random Gaussian vectors. Claiming clinical decision support via deep visual case retrieval is unverified.

---

### Discovery ADV-09: Hardcoded Set Sizes Concealing Conformal Set Inelasticity
- **Affected Files:** [`models/aw_crc_calibration.json`](file:///d:/OphthalmoAI/models/aw_crc_calibration.json) vs [`scripts/generate_all_manuscript_figures.py#L944`](file:///d:/OphthalmoAI/scripts/generate_all_manuscript_figures.py#L944)
- **Mechanism:** In `aw_crc_calibration.json`:
  ```json
  "overall_cov": 0.9686450167973124,
  "mean_set_size": 0.9686450167973124,
  "singletons_pct": 96.86450167973123
  ```
  In reality:
  - Exactly 865 test images yielded singleton sets ($|\mathcal{C}(X)| = 1$).
  - Exactly 28 test images ($3.14\%$) yielded **empty sets** ($|\mathcal{C}(X)| = 0$).
  - **Zero images yielded sets of size 2, 3, or more.**
  In `generate_all_manuscript_figures.py` (Line 944), the script hardcoded:
  ```python
  mean_set_sizes = [1.18, 1.45, 2.18]  # Synthetic values for plotting
  ```
- **Consequence:** Prediction sets did not adaptively expand under optical haze. The reported expansion was hardcoded in the figure generator to conceal that ambiguous images produce null sets.

---

### Discovery ADV-10: Residual 12-Class Prototype Remnants in Production Backend
- **Affected File:** [`backend/routes/screening.py#L247-L258`](file:///d:/OphthalmoAI/backend/routes/screening.py#L247-L258)
- **Evidence:**
  ```python
  model_group_name = "Tri-Backbone Ensemble (85.2% Test Accuracy)"
  ...
  model_group_name = "Monolithic 12-Class Model"
  ```
- **Impact:** The production backend still contains hardcoded metadata referring to a historical 12-class prototype and a legacy 85.2% test accuracy baseline, demonstrating incomplete synchronization between research scripts and production routes.

---

## 4. Step 3 — Independent Recalculation of Critical Results

| Experimental Metric | Original Manuscript | First-Audit Finding | Second-Pass Recalculation | Root Cause & Adversarial Explanation |
|---|---|---|---|---|
| **Clean Test Accuracy** | 97.23% (claimed) | 89.77% ($n=2,249$) | **89.77%** (Raw ensemble) | First audit correctly verified clean test accuracy on $n=2,249$, but 157 images are exact duplicates of training images. |
| **Cross-Split Duplicate Count** | 0 (assumed disjoint) | 159 (master base images) | **157 exact MD5 collisions** | Duplicates in Hugging Face dataset were ingested with separate `pt_cfe_` IDs and split across train and test. |
| **Conflicting Label Pairs** | None acknowledged | 142 (master base images) | **41 conflicting image pairs** | The exact same physical fundus scans exist in train as Glaucoma and in test as AMD or Normal. |
| **RETFound ViT-Large Accuracy** | 83.03% | 83.03% | **83.03%** ($224\times 224$ input) | Disadvantaged by lower resolution ($224\times 224$ vs $384\times 384$) and limited fine-tuning epochs. |
| **Statistical Test vs RETFound** | DeLong test ($p=2.29\times 10^{-15}$) | DeLong test accepted | **1,000-iteration bootstrap** | Function `compute_delong_pvalue` is actually a bootstrap difference test, falsely named DeLong in text. |
| **AW-CRC Overall Mean Set Size** | $1.18 \pm 0.39$ | $0.9686$ | **0.968645** | Prediction sets never expand beyond size 1; 3.14% of patients receive empty sets ($\mathcal{C}(X) = \emptyset$). |
| **AW-CRC Grade C Mean Set Size** | $2.18 \pm 0.62$ | $0.9759$ | **0.9759** | Figure generation script hardcoded 2.18 to manufacture the appearance of adaptive set expansion. |
| **IDRiD Zero-Shot Specificity** | "100% human review trigger" | 16.42% ($22/134$) | **16.42%** | Severe false-positive collapse under dark periphery optical profiles; Reinhard drops it to 2.99%. |
| **RIM-ONE DL Zero-Shot Specificity**| "100% human review trigger" | 1.05% ($3/285$) | **1.05%** | 282 of 285 healthy eyes misdiagnosed as Glaucoma due to severe optic cup domain shift. |
| **Expected Calibration Error (ECE)**| 0.0263 | 0.0263 (Meta-Ensemble) | **0.0263** (Equal-width) / **0.0592** (Equal-freq) | Equal-width ECE hides miscalibration in lower confidence quantiles. Equal-frequency binning reveals true ECE is 0.0592 (+125%). |
| **Tri-Backbone vs Standalone DenseNet**| "Ensemble significantly superior" | Superiority assumed | **McNemar $p = 0.8180$**<br>**DeLong $p = 0.2174$** | Standalone DenseNet-201 achieves **89.98% accuracy** ($n=2,249$). The ensemble provides NO statistically significant improvement. |
| **Multi-Seed Variance Claim** | 89.77% $\pm$ 0.17% | 85.18% $\pm$ 0.17% | **Provenanced from $n=938$** | Standard deviation 0.17% was measured on the legacy 5-seed soft-voting model on the leaked $n=938$ split, not on the stacking classifier. |

### 4.1 Granular Per-Class Diagnostic Sensitivities & Specificities ($n = 2,249$)

Live GPU inference across `test_patient_clean.csv` using calibrated model weights yielded:

| Diagnostic Category | Support ($n$) | Meta-Classifier Sens | Meta-Classifier Spec | Calibrated Soft-Voting Sens | Calibrated Soft-Voting Spec |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal (Class 0)** | 430 | **81.86%** | 97.80% | 80.47% | 97.64% |
| **Diabetic Retinopathy (Class 1)** | 485 | **92.37%** | 98.75% | 94.02% | 98.02% |
| **Glaucoma (Class 2)** | 406 | **89.66%** | 95.33% | 90.39% | 95.60% |
| **Cataract (Class 3)** | 147 | **95.24%** | 99.71% | 95.92% | 99.62% |
| **AMD (Class 4)** | 374 | **92.51%** | 98.13% | 91.44% | 98.24% |
| **HR / Pathological Myopia (Class 5)**| 407 | **90.66%** | 97.77% | 88.94% | 98.10% |

**Key Takeaways:**
1. **Normal Sensitivity Deficit:** Normal fundi exhibit the lowest sensitivity (81.86%), meaning 18.14% of healthy eyes receive false-positive disease referrals on internal test data.
2. **External Specificity Collapse Mechanism:** On external datasets (IDRiD and RIM-ONE DL), `preds_bin = (preds_meta_6 != 0).astype(int)` binarizes any non-zero prediction as disease. Domain shift easily tilts predictions away from Normal, causing 94.4% of IDRiD and 99.3% of RIM-ONE DL scans to be classified as diseased, driving specificity down to 16.42% and 1.05%.

---

## 5. Step 4 — The 5 Strongest Reviewer Rejection Arguments

Below are the 5 strongest arguments an expert IEEE reviewer would use to reject the paper for ICCECE 2027:

```
====================================================================================================
ARGUMENT 1: FATAL PAGE BUDGET VIOLATION & SCOPE SPRAWL (25 PAGES VS. 8-PAGE LIMIT)
Categorization: VALID BUT REMEDIABLE
====================================================================================================
Reviewer Critique:
The submitted manuscript (research/manuscript.tex) is 25 pages long (1,557 lines) with 21 figures 
and 15 tables. ICCECE 2027 enforces an 8-page maximum for regular conference papers. Exceeding 
the page budget by >300% triggers an immediate administrative desk reject. The manuscript dilutes 
its core contributions by attempting to be an image processing paper, an optical physics paper, 
an enterprise database manual (PostgreSQL RLS), and an FDA regulatory compliance roadmap simultaneously.

Remediation: Surgically condense to exactly 8 pages using the Phase 17 blueprint, moving enterprise 
database schemas and regulatory text to supplementary documentation.
```

```
====================================================================================================
ARGUMENT 2: CIRCULAR LABEL LEAKAGE IN SYNTHETIC EHR BIO-DATA & DEMOGRAPHIC EQUITY
Categorization: VALID AND UNRESOLVED
====================================================================================================
Reviewer Critique:
Section IV-I claims "Multimodal Diagnostic Synergy" boosting accuracy from 85.18% to 87.42% via 
12 EHR biometric variables (HbA1c, SBP, IOP, LogMAR). Section IV-G (Table 8) claims demographic 
fairness parity (DIR >= 0.966) across age brackets.
Code inspection of scripts/generate_patient_biodata.py (lines 80-140) proves that all 12 EHR 
variables were synthetically generated by sampling from Gaussian distributions parameterized directly 
by the ground-truth disease class (e.g. HbA1c ~ N(8.6, 1.45) for DR vs N(5.2, 0.35) for Normal; 
IOP ~ N(24.8, 4.6) for Glaucoma vs N(15.2, 2.8) for Normal). The EHR variables are an explicit 
leak of the target label. Any multimodal gain is an artifact of circular feature injection, and 
auditing demographic fairness across age brackets using a synthetic label-correlated age column 
evaluates random generation, not clinical algorithmic equity.

Remediation: Permanently retract Section IV-I, Table 7, and Table 8. Frame the paper strictly as a 
vision-only retinal screening system.
```

```
====================================================================================================
ARGUMENT 3: THEORETICAL MATHINESS & APPLICABILITY MISMATCH IN ENSEMBLE VARIANCE PROOFS
Categorization: VALID AND UNRESOLVED
====================================================================================================
Reviewer Critique:
Proposition 1 (Canonical Ensemble Variance Bound) is presented as a novel theoretical contribution:
  Var(P_ensemble) <= sigma_max^2 * (1 + (M - 1)*rho_bar) / M
1. This formula is a classical, 70-year-old textbook identity for the variance of correlated random 
   variables (Markowitz 1952; Krogh & Vedelsby 1995). Publishing it as an original proposition with 
   a formal proof in an IEEE paper constitutes decorative "mathiness."
2. Proposition 1 applies strictly to uniform soft voting. The primary model claimed in the paper 
   is a non-linear learned MLP stacking meta-classifier to which Proposition 1 does not apply.
3. For the uniform soft-voting model where the proof holds, the authors' own empirical data 
   demonstrates that probability calibration suffers Calibration Inversion: ECE worsens from 
   0.0545 (DenseNet-201) to 0.0787 (Soft Voting).

Remediation: Reframe Proposition 1 as established background theory, clarify that the stacking 
meta-classifier operates as a non-linear calibration shrinkage operator, and explain why soft-voting 
exhibits calibration inversion.
```

```
====================================================================================================
ARGUMENT 4: CONFORMAL EMPTY-SET FLAW (3.14% NULL SETS) & METRIC CONCEALMENT
Categorization: VALID AND UNRESOLVED
====================================================================================================
Reviewer Critique:
1. The mathematical formulation of AW-CRC produces empty prediction sets (C_AW(X) = emptyset) 
   whenever top softmax confidence falls below 1 - q_hat / S(X)^gamma.
2. The empirical calibration file (models/aw_crc_calibration.json) documents an overall mean set 
   size of 0.9686 (< 1.0) and exactly 28 null sets (3.14%), with ZERO prediction sets expanding to 
   size 2 or 3.
3. The manuscript conceals this phenomenon and reports fabricated metrics in numbers.tex 
   (AWCRCMeanSetSize = 1.18 +- 0.39 and HazyScanMeanSetSize = 2.18 +- 0.62), which were hardcoded 
   in generate_all_manuscript_figures.py line 944.
In clinical triage, returning an empty set leaves clinicians with zero actionable hypothesis, 
invalidating the claim of "trustworthy selective triage."

Remediation: Transparently report the 3.14% empty set rate, formalize the non-empty fallback policy:
  C*_safe(X) = C_AW(X) if |C_AW(X)| >= 1 else {argmax P(c|X)} U {Flag: Urgent Specialist Escalation}
and delete all hardcoded figure generation values.
```

```
====================================================================================================
ARGUMENT 5: CATASTROPHIC EXTERNAL SPECIFICITY COLLAPSE & PHANTOM EXPLAINABILITY (PASG)
Categorization: VALID AND UNRESOLVED
====================================================================================================
Reviewer Critique:
1. External Generalization Failure: On external validation cohorts with different camera optics 
   (IDRiD, Kowa VX-10 optics, India, N=413; RIM-ONE DL, Nidek optics, Spain, N=485), specificity 
   completely collapses to 16.42% and 1.05%. In a primary screening setting, this would falsely 
   refer >83% to >98% of healthy patients to specialist clinics, causing clinical gridlock.
2. Phantom Explainability Claims: Formulation 5 and Table 9 claim "Pixel-Aligned Saliency Grounding 
   (PASG-GradCAM)" using an automated U-Net segmentation pipeline, an 88.4% Pointing Game benchmark, 
   and a 10% ophthalmologist audit (Dice 0.91 +- 0.04). Code inspection reveals:
   - No U-Net segmentation weights or code exist.
   - No Pointing Game evaluation script exists.
   - In backend/services/explanation_service.py line 80, spatial quadrant descriptions are selected 
     via random.choice(quadrants)!

Remediation: Retract Table 9 and claims of U-Net segmentation / ophthalmologist audits. Replace 
random.choice() with deterministic bounding-box logic. Reframe the external specificity collapse 
honestly as camera sensor shift requiring camera-specific calibration.
```

---

## 6. Step 5 — Novelty Deconstruction Against Prior Art

### 6.1 Biophysical Guardrail $\Phi(X)$
- **Claimed Novelty:** "Optical Aperture and Chromophore Domain Guardrail (OAC-DG)."
- **Prior Art:**
  - *Circular Aperture Masking:* Fleming et al. (2006), Lalonde et al. (2001), Usher et al. (2004) formalized circular fundus mask extraction and corner dark vignetting checks.
  - *Chromophore Ratio ($\rho_{\text{RB}} \ge 1.05$):* Niemeijer et al. (2006, *IEEE TMI*) and Hunter et al. (2011, *IOVS*) established color histogram profiling in retinal image quality assessment. Hemoglobin absorption in green/blue (400–550 nm) and melanin reflection in red (>600 nm) has been standard fundus preprocessing for 20 years.
  - *Contrast & Saturation Thresholds:* Shen et al. (2019, *EyeQuality*) and Köhler et al. (2013).
- **Novelty Verdict:** **ZERO METHODOLOGICAL NOVELTY.** It is an engineering assembly of 20-year-old OpenCV threshold heuristics. However, implementing it as a sub-2.0 ms CPU pre-filter before allocating GPU memory is solid systems engineering and should be presented as an *efficient clinical systems contribution*, not a foundational optical discovery.

### 6.2 Tri-Backbone Stacking Ensemble
- **Claimed Novelty:** Tri-backbone heterogeneous feature aggregation.
- **Prior Art:**
  - Wolpert (1992) and Breiman (1996) formalized stacking.
  - Kaggle Diabetic Retinopathy (2015, Ben Graham) and APTOS (2019) winning solutions deployed ensembles combining DenseNet, ResNeXt, and EfficientNet backbones with soft voting or meta-regression heads.
  - Ting et al. (2017, *JAMA*) deployed multi-backbone deep CNN ensembles specifically for multi-class retinal pathology.
- **Novelty Verdict:** **INCREMENTAL APPLIED MACHINE LEARNING.** Selecting off-the-shelf ImageNet models and stacking them with an MLP is standard practice.

### 6.3 Admissibility-Weighted Conformal Risk Control (AW-CRC)
- **Claimed Novelty:** Distribution-free risk-controlling prediction sets weighted by optical quality.
- **Prior Art:**
  - Modifying non-conformity scores by local difficulty or quality is the definition of *Locally Adaptive Conformal Prediction* (Papadopoulos et al. 2008) and *Adaptive Prediction Sets / RAPS* (Romano, Patterson, & Candès 2019; Angelopoulos et al. 2021).
  - In AW-CRC, multiplying by $S(X)^\gamma$ is a scalar heuristic re-weighting of the threshold: $\tau(X) = 1 - \hat{q}/S(X)^\gamma$. It does not alter the underlying conformal machinery, nor does it guarantee conditional coverage within quality strata (which requires Mondrian conformal prediction; Vovk 2003).
  - Furthermore, under optical covariate shift ($P_{\text{test}}(X) \neq P_{\text{cal}}(X)$), exchangeability is violated (Tibshirani et al. 2019).
- **Novelty Verdict:** **HEURISTIC EXTENSION.** Useful for practical engineering, but mathematically informal under distribution shift.

### Recommended Narrow, Defensible Scope for ICCECE 2027
**Reframed Research Question:**
> *"How Can Deterministic Biophysical Optical Gating and Quality-Conditioned Conformal Sets Mitigate Silent Deep Learning Failure Modes in Point-of-Care Retinal Disease Screening?"*

---

## 7. Step 6 — Verification of Remediations & Residual Risk Ledger

| Vulnerability ID | Root Cause | Required Correction | Experiment Needed | Implementation Status | Safe Verification Result | Residual Risk |
|---|---|---|---|:---:|---|---|
| **ISS-01** (25 Pages) | Scope sprawl across enterprise DB, FDA roadmap, and regulatory essays. | Surgically condense to 8 pages using Phase 17 blueprint. | Compile 8-page `D:\OphthalmoAI-Paper\main.tex`. | **IN PROGRESS** | Blueprint completed in `17_FINAL_RESEARCH_ROADMAP.md`. | Low; purely editorial. |
| **ISS-02** (Synthetic EHR) | Gaussian sampling conditioned on disease label in `generate_patient_biodata.py`. | Retract Section IV-I and Table 7; formulate as vision-only. | Vision-only benchmark verification. | **PENDING** | Verified that base vision models run independently without EHR inputs. | Zero risk once text is deleted. |
| **ISS-03** (Synthetic Fairness) | Demographic fairness evaluated on synthetic label-correlated age. | Retract Table 8; discuss fairness as future multi-center work. | None (retraction). | **PENDING** | Flagged for deletion in paper draft. | Zero risk once text is deleted. |
| **ISS-04** (Table 6 Discrepancy) | Confusion between unweighted OvR mean (0.9839) and global weighted AUROC (0.9914). | Update Table 6 to report 0.9839 unweighted and publish $6\times 6$ confusion matrix. | Confusion matrix extraction. | **PENDING** | Confusion matrix verified: 1,871.4 correct cases matches class supports. | Low; numbers reconciled. |
| **ISS-05** (Empty Sets) | Score cutoff $\max P(c \mid X) < 1 - \hat{q}/S(X)^\gamma$ leaves set empty. | Add non-empty fallback rule $\mathcal{C}^*_{\text{safe}}(X)$; disclose 3.14% null sets. | Fallback referral evaluation across $\alpha \in [0.01, 0.10]$. | **PENDING** | Verified 865 singletons and 28 null sets in `models/aw_crc_calibration.json`. | Low; formalizes triage fallback. |
| **ADV-01** (Pseudo-Patients) | `num // 2` applied to randomly shuffled row index in `prepare_fundus_dataset.py`. | Document the heuristic honestly; run de-duplicated patient hash split. | Re-split using MD5 hash clustering. | **PENDING** | Discovered 157 exact MD5 duplicate images crossing train and test. | Medium; requires disclosing split limitations. |
| **ADV-02** (Cross-Split Dups) | Ingestion of `color_fundus_eye` imported duplicate files across pathology folders. | Deduplicate master manifest by image MD5 before splitting. | Re-run evaluation excluding 157 contaminated test images. | **PENDING** | 157 duplicate hashes cataloged and isolated. | Medium; slight test set reduction ($n=2,092$). |
| **ADV-03** (Heuristic Fallback)| Silent fallback in `edgeInference.js` guesses diseases from RGB ratios. | Delete `edgeHeuristic.js` diagnostic guessing; show clear error modal on ONNX failure. | Test frontend error handling with disabled ONNX. | **PENDING** | Located in `frontend/src/edgeHeuristic.js` lines 287–295. | Low; frontend safety patch. |
| **ADV-04** (Fail-Open Guard) | `screening.py` catches all exceptions in `validate_fundus_image` with warning. | Change `except Exception:` to fail closed (`raise HTTPException(422)`). | Unit test throwing exception in validator. | **PENDING** | Located in `backend/routes/screening.py` lines 205–207. | Zero; 2-line code fix. |
| **ADV-05** (False DeLong) | 1,000-sample bootstrap difference labeled as DeLong test in script. | Change manuscript text to: "1,000-iteration bootstrap difference test ($p < 0.001$)." | None (text correction). | **PENDING** | Located in `scripts/finetune_retfound_comparison.py` lines 27–32. | Low; text alignment. |
| **ADV-07** (Reinhard Bug) | Reinhard normalizer includes black border in mean/std calculation. | Compute LAB mean/std strictly over foreground retinal mask. | Re-evaluate IDRiD with foreground-only Reinhard. | **PENDING** | Located in `scripts/evaluate_external_dataset.py` lines 70–84. | Low; improves domain adaptation. |
| **ADV-08** (Synthetic CBMIR)| `ClinicalCaseVectorIndex` uses random Gaussian vectors instead of deep embeddings. | Retract CBMIR claims or connect to actual ConvNeXt penultimate layer embeddings. | Extract true 512-dim embeddings. | **PENDING** | Located in `backend/vector_search.py` lines 83–97. | Low; prune claims from paper. |

---

## 8. Step 7 — Final Internal Research-Readiness Classification

### **CLASSIFICATION: REQUIRES MAJOR METHODOLOGICAL REVISION**

```
+------------------------------------------------------------------------------------------------------------------+
|                                    INTERNAL RESEARCH-READINESS CLASSIFICATION                                    |
+------------------------------------------------------------------------------------------------------------------+
| [X] NOT READY FOR SUBMISSION                                                                                     |
| [ ] REQUIRES MAJOR EXPERIMENTAL WORK                                                                             |
| [X] REQUIRES MAJOR METHODOLOGICAL REVISION                                                                       |
| [ ] SCIENTIFICALLY DEFENSIBLE BUT REQUIRES WRITING REVISION                                                      |
| [ ] READY FOR FINAL HUMAN REVIEW                                                                                 |
+------------------------------------------------------------------------------------------------------------------+
```

### Justification & Criteria for Progression
The project is **NOT READY** for immediate submission in its current state because:
1. The primary manuscript (`research/manuscript.tex`) is 25 pages long, guaranteeing an administrative desk reject at ICCECE 2027.
2. The primary manuscript contains claims based on circular synthetic EHR features, phantom PASG explainability, and fabricated figure set sizes.
3. The "clean" split (`test_patient_clean.csv`) contains 157 exact MD5 duplicate images from training, some with conflicting diagnostic labels.
4. Production and frontend routes contain silent heuristic fallbacks and fail-open exception handlers.

### Actionable Roadmap to Reach "READY FOR FINAL HUMAN REVIEW":
1. **Purge Synthetic Claims:** Delete Section IV-I (synthetic EHR multimodal synergy), Table 8 (synthetic demographic fairness), Table 9 (phantom PASG-GradCAM), and CBMIR case retrieval claims from the paper.
2. **Execute Clean-Split Deduplication:** Remove the 157 cross-split exact duplicates from `test_patient_clean.csv` to establish an uncontaminated test cohort ($n = 2,092$).
3. **Draft the 8-Page IEEE Manuscript:** Compile `D:\OphthalmoAI-Paper\main.tex` following the strict page budget established in Phase 17.
4. **Patch Code Safety Deficiencies:**
   - In `backend/routes/screening.py`, change `except Exception as val_err:` to fail closed (`HTTPException(422)`).
   - In `frontend/src/edgeInference.js`, remove `edgeHeuristic.js` disease guessing and replace with a standard error message.
   - In `scripts/evaluate_external_dataset.py`, mask out black border pixels before computing Reinhard LAB moments.
5. **Reconcile Conformal Reporting:** Transparently report the 3.14% empty set rate, formalize the non-empty fallback rule, and remove all hardcoded figure generation constants.

Once these five mandatory revisions are completed, OphthalmoAI will stand as a mathematically rigorous, experimentally reproducible, and clinically defensible paper ready for submission to **ICCECE 2027**.
