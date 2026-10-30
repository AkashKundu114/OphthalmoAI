# Phase 7 Audit Report: Optical Domain Guardrail & Out-of-Distribution (OOD) Detection

**Document Reference:** `research_audit/08_OOD_AND_DOMAIN_GUARDRAIL_AUDIT.md`  
**Auditor:** Senior Clinical AI Safety, Ophthalmic Imaging, and Medical AI Auditor  
**Date of Audit:** October 30, 2026  
**Repository Target:** OphthalmoAI (`backend/fundus_validator.py`, `scripts/red_team_guardrail_test.py`, `scripts/reproduce_evaluation.py`, `manuscript.tex`)  
**Audit Status:** COMPLETE  

---

## Executive Summary

The Optical Aperture and Chromophore Domain Guardrail (OAC-DG) is presented in the research manuscript as a deterministic, biophysically grounded pre-inference operator $\Phi(X) \in \{0, 1\}$ designed to intercept non-fundus imagery, severe optical artifacts, and adversarial/accidental out-of-distribution (OOD) inputs on the host CPU in $1.8\text{ ms}$, prior to GPU tensor allocation.

This audit conducted an uncompromising forensic verification of:
1. The mathematical formulation vs. executable implementation in the codebase.
2. The claimed 2,500-sample negative stress corpus across Tier 1 (near-OOD retinal degradations) and Tier 2 (far-OOD biomedical, dermoscopic, natural, and synthetic noise).
3. The comparative baseline evaluations against Maximum Softmax Probability (MSP, $\tau=0.70$) and ODIN ($T=1000, \epsilon=0.0014$).
4. The claimed 0.00% False Rejection Rate (FRR) across the $n=2,249$ patient-segregated clinical test cohort.
5. The CPU execution latency claims ($1.8\text{ ms}$ on CPU).

### Audit Verdict Summary Matrix

| Audit Target / Claim | Manuscript Claim | Codebase Reality | Evidence Label |
| :--- | :--- | :--- | :--- |
| **Guardrail Module Location** | `backend/services/guardrail_service.py` | Absent at specified path; implemented in [`backend/fundus_validator.py`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L1-L292) as `validate_fundus_image` | **PARTIALLY VERIFIED** |
| **Mathematical Gate Formulation** | Conjunctive gate: $\Phi(X) = \mathcal{G}_{\text{pre}}(X) \cdot \mathbb{I}(\sum \mathcal{S} \ge 0.50)$ | Faithful Python implementation of geometry, luminance variance, white ratio, spatial autocorrelation, circular aperture, and chromophore ratio | **VERIFIED** |
| **Optical Aperture Operator ($\mathcal{S}_{\text{aperture}}$)** | Corner mean $<45$, center/corner ratio $\ge 1.35$, active tissue $0.15 \le \frac{\|\Omega\|}{HW} \le 0.98$ | Implemented at [`backend/fundus_validator.py#L99-L126`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L99-L126); handles corner quadrants and central disc | **VERIFIED** |
| **Chromophore Spectral Ratio ($\rho_{RB}, \rho_{RG}$)** | Melanin & hemoglobin absorption: $\rho_{RB} \ge 1.05$, R/G/B ratios, BGR swap check | Implemented at [`backend/fundus_validator.py#L145-L190`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L145-L190); includes BGR inversion heuristic | **VERIFIED** |
| **Spatial Autocorrelation ($r_{\text{spatial}} \ge 0.35$)** | Mean lag-1 Pearson correlation over horizontal/vertical axes | Implemented at [`backend/fundus_validator.py#L72-L97`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L72-L97) using `np.corrcoef` on luminance | **VERIFIED** |
| **Vascular Bed Contrast ($C_{\text{vessel}}$)** | Gradient magnitude of green channel in active tissue | Implemented at [`backend/fundus_validator.py#L129-L143`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L129-L143) via `np.diff` on green channel | **VERIFIED** |
| **2,500-Sample Negative Stress Corpus** | 2,500 real negative images (CheXpert 400, ISIC 400, ImageNet 400, Noise 200, Retinal 1,100) | **NO image dataset exists on disk**. Evaluation in [`scripts/reproduce_evaluation.py#L127-L148`](file:///d:/OphthalmoAI/scripts/reproduce_evaluation.py#L127-L148) is a hardcoded print list | **UNVERIFIED / SIMULATED** |
| **Comparative Baselines (MSP & ODIN)** | MSP rejects 0.22%, ODIN rejects 68.4% with 11.20% FRR | **Zero code implementation**. Neither MSP nor ODIN exists in any Python script; numbers are ungrounded text assertions | **CONTRADICTED / UNVERIFIED** |
| **In-Domain Clinical FRR (0.00%)** | $0.00\%$ (0 / 2,249 rejected) on patient-clean test cohort | **Empirically confirmed**: Batch audit of all 2,249 test images against `validate_fundus_image` yielded 2,249 passes, 0 fails | **VERIFIED** |
| **CPU Execution Latency (1.8 ms)** | $1.8\text{ ms}$ on host CPU | Empirical benchmark on $384 \times 384$ array: Mean $6.94\text{ ms}$ (P95 $8.38\text{ ms}$); full disk pipeline averages $7.94\text{ ms}$ | **PARTIALLY VERIFIED / OVERSTATED** |

---

## 1. Architectural & Codebase Audit: Guardrail Implementation

### 1.1 File Path & Service Wiring
The audit protocol specified inspecting `backend/services/guardrail_service.py`.
- **Finding:** The directory [`backend/services/`](file:///d:/OphthalmoAI/backend/services/) contains only:
  - `explanation_service.py`
  - `image_preprocessor.py`
  - `inference_service.py`
  - `model_loader.py`
- **Resolution:** The optical domain guardrail is fully implemented in [`backend/fundus_validator.py`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L1-L292) and wired into the FastAPI screening pipeline in [`backend/routes/screening.py#L419`](file:///d:/OphthalmoAI/backend/routes/screening.py#L419):
  ```python
  from backend.fundus_validator import validate_fundus_image
  ...
  is_valid, guardrail_score, guardrail_reason, guardrail_metrics = validate_fundus_image(image_pil)
  if not is_valid:
      raise HTTPException(
          status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
          detail={
              "error": "Optical Guardrail Violation",
              "reason": guardrail_reason,
              "score": guardrail_score,
              "metrics": guardrail_metrics
          }
      )
  ```
- **Evidence Label:** **PARTIALLY VERIFIED** (Feature exists and is wired to return HTTP 422, but resides in `backend/fundus_validator.py` rather than `backend/services/guardrail_service.py`).

---

## 2. Mathematical Formalization vs. Code Realization

The manuscript formalizes the guardrail operator as:
$$\Phi(X) = \mathcal{G}_{\text{pre}}(X) \cdot \mathbb{I}\Big(\mathcal{S}_{\text{aperture}}(X) + \mathcal{S}_{\text{chromophore}}(X) + \mathcal{S}_{\text{autocorr}}(X) + \mathcal{S}_{\text{contrast}}(X) \ge \tau_{\text{guard}}\Big)$$

with veto gate:
$$\mathcal{G}_{\text{pre}}(X) = \mathbb{I}(\sigma_{\text{lum}} \ge 8.0) \cdot \mathbb{I}(r_{\text{white}} \le 0.65) \cdot \mathbb{I}(r_{\text{spatial}} \ge 0.35) \cdot \mathbb{I}(\mathcal{S}_{\text{aperture}} > 0) \cdot \mathbb{I}(\mathcal{S}_{\text{chromophore}} > 0)$$

We audited each individual condition in [`backend/fundus_validator.py`](file:///d:/OphthalmoAI/backend/fundus_validator.py):

### 2.1 Dimensional & Luminance Pre-Filters
1. **Minimum Dimensions:** `MIN_DIMENSION_PX = 128`, `MAX_ASPECT_RATIO = 3.5` ([`backend/fundus_validator.py#L11-L41`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L11-L41)). Rejects sub-diagnostic thumbnails and panoramic banners.
2. **Luminance Dispersion ($\sigma_{\text{lum}} \ge 8.0$):**
   - Luminance computed via ITU-R BT.601: $\text{lum} = 0.299R + 0.587G + 0.114B$ ([`backend/fundus_validator.py#L66`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L66)).
   - Condition `std_lum < MIN_LUMINANCE_STD (8.0)` triggers immediate veto: `failure_stage: "zero_variance"`. Rejects solid black/white frames.
3. **White Pixel Fraction ($r_{\text{white}} \le 0.65$):**
   - Condition `np.mean(lum > 238) > 0.65` triggers veto: `failure_stage: "document_pattern"`. Rejects medical reports, prescription scans, and text documents.
- **Evidence Label:** **VERIFIED**

### 2.2 Spatial Autocorrelation ($r_{\text{spatial}} \ge 0.35$)
- Code implementation ([`backend/fundus_validator.py#L72-L97`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L72-L97)):
  - Evaluates horizontal lag-1 correlation: $h_{\text{left}} = \text{lum}[:, :-1]$, $h_{\text{right}} = \text{lum}[:, 1:]$.
  - Evaluates vertical lag-1 correlation: $v_{\text{top}} = \text{lum}[:-1, :]$, $v_{\text{bot}} = \text{lum}[1:, :]$.
  - $r_{\text{spatial}} = \text{clip}\big(\frac{r_h + r_v}{2}, 0, 1\big)$ using Pearson correlation via `np.corrcoef`.
  - Condition `spatial_corr < 0.35` triggers immediate veto: `failure_stage: "random_noise"`.
  - Rejects synthetic white Gaussian noise, uniform static, and transmission corruption.
- **Evidence Label:** **VERIFIED**

### 2.3 Telecentric Circular Aperture Operator ($\mathcal{S}_{\text{aperture}}$)
- Code implementation ([`backend/fundus_validator.py#L99-L126`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L99-L126)):
  - Corners sampled as $8\% \times 8\%$ quadrants: $\bar{I}_{\text{corn}} = \frac{1}{4}(\bar{I}_{TL} + \bar{I}_{TR} + \bar{I}_{BL} + \bar{I}_{BR})$.
  - Center disc sampled at width/height radii of $22\%$: $\bar{I}_{\text{cent}}$.
  - Telecentric aperture verified if $\bar{I}_{\text{corn}} < 45.0 \land \bar{I}_{\text{cent}} > 55.0 \land \frac{\bar{I}_{\text{cent}}}{\bar{I}_{\text{corn}} + 10^{-4}} > 1.35$.
  - Active tissue mask: $\Omega_{\text{tissue}} = \{p : \text{lum}_p > \max(15.0, 1.15 \bar{I}_{\text{corn}})\}$.
  - Scoring: $+0.35$ if circular mask confirmed; $+0.20$ if active ratio $0.15 \le \frac{|\Omega|}{HW} \le 0.98$.
  - Immediate veto if active ratio $< 0.12$: `failure_stage: "dark_field"`.
- **Evidence Label:** **VERIFIED**

### 2.4 Chorioretinal Chromophore Spectral Ratio ($\mathcal{S}_{\text{chromophore}}$)
- Code implementation ([`backend/fundus_validator.py#L145-L190`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L145-L190)):
  - Chromophore ratios computed strictly over active tissue $\Omega_{\text{tissue}}$:
    $$\rho_{RB} = \frac{\bar{R}_{\text{tissue}}}{\bar{B}_{\text{tissue}} + \epsilon}, \quad \rho_{RG} = \frac{\bar{R}_{\text{tissue}}}{\bar{G}_{\text{tissue}} + \epsilon}$$
  - **BGR Inversion Detection:** An automated diagnostic heuristic detects if an OpenCV BGR array was passed uncorrected (`b_mean > r_mean * 1.15` with circular mask or vessel gradient) and swaps channels to preserve clinical throughput.
  - Veto rules:
    - Dominant blue rejection: $B > 1.25R \land B > 1.15G$ without circular mask.
    - Dominant green rejection: $G > 1.35R$ without circular mask.
    - Monochromatic/greyscale rejection: channel variance $< 3.5$ without circular mask.
  - Chromophore Scoring:
    - $\rho_{RB} \ge 1.18 \land \rho_{RG} \ge 1.02 \implies +0.35$
    - $0.94 \le \rho_{RB} \le 1.25 \land 0.92 \le \rho_{RG} \le 1.18 \implies +0.32$
    - $\rho_{RB} \ge 1.05 \implies +0.20$
    - Otherwise $\implies +0.08$
- **Evidence Label:** **VERIFIED**

### 2.5 Vascular Bed Contrast Operator ($\mathcal{S}_{\text{contrast}}$)
- Code implementation ([`backend/fundus_validator.py#L129-L143`](file:///d:/OphthalmoAI/backend/fundus_validator.py#L129-L143)):
  - $g_{\text{norm}} = G / 255.0$; gradients computed as $g_x = \text{diff}(g_{\text{norm}}, \text{axis}=1)$, $g_y = \text{diff}(g_{\text{norm}}, \text{axis}=0)$.
  - Gradient magnitude: $|\nabla G| = \sqrt{g_x^2 + g_y^2}$.
  - Averaged over active tissue mask: $C_{\text{vessel}} = \text{mean}(|\nabla G|_{\Omega_{\text{tissue}}})$.
  - Scoring combined with spatial autocorrelation:
    - $0.005 \le C_{\text{vessel}} \le 0.25 \land r_{\text{spatial}} \ge 0.65 \implies +0.30$
    - $0.004 \le C_{\text{vessel}} \le 0.25 \land r_{\text{spatial}} \ge 0.35 \implies +0.20$
    - $r_{\text{spatial}} \ge 0.50 \implies +0.15$
    - Otherwise $\implies +0.05$
- Decision threshold: $\Phi(X) = 1 \iff \mathcal{S} \ge 0.50$.
- **Evidence Label:** **VERIFIED**

---

## 3. Forensic Audit of the Claimed 2,500-Sample Negative Stress Corpus

Table III of the manuscript claims:
> "Evaluations were conducted across both a unified two-tier negative stress corpus ($n = 2,500$ non-fundus samples) and the verified in-domain clinical test cohort ($n = 2,249$). Rejection rate: Tier 1 (1,100 / 1,100, 100%), CheXpert (400 / 400, 100%), ISIC (400 / 400, 100%), Synthetic Noise (200 / 200, 100%), ImageNet (346 / 400, 86.5%). Overall: 97.84% (2,446 / 2,500)."

### 3.1 Dataset Availability on Disk
A recursive filesystem search of `dataset/`, `models/`, and project directories for CheXpert, ISIC, or ImageNet datasets returned **zero files**:
- No CheXpert radiography files exist anywhere in the repository.
- No ISIC dermoscopy images exist anywhere in the repository.
- No ImageNet sample directory exists anywhere in the repository.
- Total raw datasets present on disk: [`dataset/raw/aptos_dr`](file:///d:/OphthalmoAI/dataset/raw/aptos_dr), [`dataset/raw/color_fundus_eye`](file:///d:/OphthalmoAI/dataset/raw/color_fundus_eye), [`dataset/raw/odir5k`](file:///d:/OphthalmoAI/dataset/raw/odir5k), [`dataset/raw/preprocessed_fundus`](file:///d:/OphthalmoAI/dataset/raw/preprocessed_fundus). All are retinal fundus datasets.

### 3.2 Evaluation Script Analysis
We audited [`scripts/reproduce_evaluation.py`](file:///d:/OphthalmoAI/scripts/reproduce_evaluation.py#L123-L148) and [`scripts/red_team_guardrail_test.py`](file:///d:/OphthalmoAI/scripts/red_team_guardrail_test.py#L31-L49):
- In `scripts/red_team_guardrail_test.py`, only **6 programmatic synthetic test cases** are evaluated:
  1. Solid Blue Surface (`Image.new("RGB", (384, 384), (20, 40, 230))`) -> Blocked
  2. Solid White Document (`Image.new("RGB", (384, 384), (250, 250, 250))`) -> Blocked
  3. Pure Black Screen (`Image.new("RGB", (384, 384), (0, 0, 0))`) -> Blocked
  4. Synthetic White Noise (`np.random.randint(0, 255, (384, 384, 3))`) -> Blocked
  5. Under-resolved 64x64 (`Image.new("RGB", (64, 64), (180, 100, 30))`) -> Blocked
  6. Everyday Natural Scenery (synthetic split image: blue sky / green grass) -> Blocked
  7. 1 Real Color Fundus Scan -> Passed
- In `scripts/reproduce_evaluation.py` lines 127–148:
  ```python
  negative_stress_splits = [
      ("Near-OOD Retinal Artifacts (Flash/Haze)", 1100, 1100, "100.0%", "Spectral / pupil dynamic range violation"),
      ("CheXpert Chest X-Rays", 400, 400, "100.0%", "Absence of hemoglobin-melanin absorption"),
      ("ISIC 2019 Dermoscopic Lesions", 400, 400, "100.0%", "Non-telecentric aperture & chromatic disparity"),
      ("Synthetic Noise / Solids / Text", 200, 200, "100.0%", "Zero vascular contrast & dynamic range anomaly"),
      ("ImageNet Natural Images (Far-OOD)", 400, 346, "86.5%", "Diffuse natural scenery color distribution"),
  ]
  ```
  The script **does not execute any model or guardrail against 2,500 images**. It simply iterates over a hardcoded tuple list and prints the numbers.
- **Evidence Label:** **UNVERIFIED / SIMULATED**. The 2,500-sample negative stress corpus is not an actual image corpus preserved on disk, but a simulated demonstration list. While the physical rules (telecentric aperture, chromophore absorption) would theoretically reject greyscale X-rays and skin lesions, the reported empirical benchmark of 2,446 / 2,500 (97.84%) was not generated by executing code over real CheXpert, ISIC, or ImageNet datasets.

---

## 4. Audit of Comparative Baselines: MSP and ODIN

Table III claims:
- Maximum Softmax Probability (MSP, $\tau=0.70$): Overall OOD Rejection = 0.22% (Overconfident), Clinical In-Domain FRR = 25.91% (583 / 2,249).
- ODIN ($T=1000, \epsilon=0.0014$): Overall OOD Rejection = 68.4%, Clinical In-Domain FRR = 11.20% (252 / 2,249).

### 4.1 Implementation Search
A full-text search across all Python files in `backend/`, `scripts/`, `models/`, `benchmarks/`, and `tests/` for `ODIN`, `temperature scaling perturbation`, `epsilon`, and `MSP` baseline implementations revealed:
- **No implementation of ODIN exists in the repository.** ODIN requires computing input image gradient perturbations $\tilde{x} = x - \epsilon \cdot \text{sign}(-\nabla_x \log S_{\hat{y}}(x; T))$ via backpropagation through the model. There is no script or function in the repository performing input backpropagation for OOD detection.
- **No implementation of MSP OOD filtering exists in the repository.**
- **Evidence Label:** **CONTRADICTED / UNVERIFIED**. The comparative baseline figures for MSP and ODIN in Table III are ungrounded text assertions imported from external literature, not empirically measured against the OphthalmoAI backbones on a real test suite.

---

## 5. Empirical Verification: False Rejection Rate (FRR) on Clinical Fundus Scans

The manuscript claims that on the patient-segregated test cohort ($n = 2,249$), the guardrail achieves a False Rejection Rate (FRR) of **0.00% (0 / 2,249)**, meaning no genuine clinical fundus image is erroneously blocked.

### 5.1 Verification Experiment
We executed an automated live audit script running all 2,249 images from [`dataset/processed/test_patient_clean.csv`](file:///d:/OphthalmoAI/dataset/processed/test_patient_clean.csv) through `validate_fundus_image`:
- **Cohort:** 2,249 verified color fundus photographs.
- **Result:**
  - `Tested`: 2,249
  - `Passed`: 2,249
  - `Failed (FRR)`: **0**
  - `Missing files`: 0
  - `Empirical FRR`: **0.000% (0 / 2,249)**
- **Evidence Label:** **VERIFIED**. The claim of 0.00% false rejection rate on the in-distribution clinical test cohort is fully verified and mathematically sound.

---

## 6. Execution Speed & Latency Audit

The manuscript claims:
> "Operating entirely on raw RGB pixels on the host CPU in $1.8\text{ ms}$, the guardrail intercepts non-ocular entities before GPU memory allocation or neural execution begins..."

### 6.1 Benchmark Protocol
We benchmarked `validate_fundus_image` on host CPU (32-thread AMD/Intel CPU, Windows environment):
1. **In-Memory Array Benchmark (100 iterations, $384 \times 384 \times 3$ image):**
   - Mean latency: **6.94 ms**
   - Median latency: **6.75 ms**
   - P95 latency: **8.38 ms**
   - Min latency: **6.11 ms**
   - Max latency: **10.70 ms**
2. **In-Memory Array Benchmark ($256 \times 256 \times 3$ image):**
   - Mean latency: **3.74 ms**
   - Min latency: **3.17 ms**
3. **End-to-End Pipeline (Disk load + NumPy array conversion + Validation across 2,249 test images):**
   - Total time for 2,249 images: **17.86 seconds**
   - Mean throughput latency: **7.94 ms per image**

### 6.2 Latency Breakdown
The operations required by OAC-DG include:
- BT.601 perceptual luminance conversion: $0.299R + 0.587G + 0.114B$ ($\sim 0.8\text{ ms}$)
- Horizontal and vertical lag-1 Pearson correlations via `np.corrcoef` ($\sim 2.8\text{ ms}$)
- Aperture slicing and quadrant averaging ($\sim 0.6\text{ ms}$)
- Vessel gradient magnitude calculation via `np.diff` on green channel ($\sim 1.5\text{ ms}$)
- Chromophore mean calculation over active mask ($\sim 1.2\text{ ms}$)
Total computational cost on a standard $384 \times 384$ array is **6.5–7.5 ms**, not **1.8 ms**.
- **Evidence Label:** **PARTIALLY VERIFIED / OVERSTATED**. While the guardrail is indeed lightweight and executes entirely on host CPU without GPU intervention, the claimed latency of $1.8\text{ ms}$ is understated by a factor of $\sim 3.8\times$. The actual execution latency is approximately $6.9\text{ ms}$ in RAM and $7.9\text{ ms}$ including image preprocessing.

---

## 7. Mandatory Remediation Recommendations

1. **Clarify Stress Corpus Provenance in Manuscript:**
   - [PROPOSED] Replace the assertion that 2,500 real negative images from CheXpert, ISIC, and ImageNet were evaluated on GPU with an accurate description: *"The physical rejection criteria (telecentric circular masking, hemoglobin-melanin chromophore ratios, spatial continuity) were validated across programmatic boundary conditions and an automated red-team test suite (`scripts/red_team_guardrail_test.py`), achieving 0.00% False Rejection on $n=2,249$ clinical test images."*
2. **Remove or Qualify MSP/ODIN Experimental Baseline Claims:**
   - [PROPOSED] Table III's MSP and ODIN numbers should either be implemented in a reproducible script (`scripts/evaluate_ood_baselines.py`) or explicitly designated as literature-derived comparisons rather than direct empirical measurements on this specific dataset.
3. **Correct CPU Latency Claim:**
   - [PROPOSED] Update manuscript latency from "$1.8\text{ ms}$" to *"~6.9 ms on host CPU (sub-10 ms)"*, reflecting rigorous empirical measurement on standard $384 \times 384$ clinical inputs.
4. **Code Path Reference Update:**
   - [PROPOSED] In documentation and paper text, reference [`backend/fundus_validator.py`](file:///d:/OphthalmoAI/backend/fundus_validator.py) instead of the non-existent `backend/services/guardrail_service.py`.
