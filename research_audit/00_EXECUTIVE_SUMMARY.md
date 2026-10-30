# Autonomous Multidisciplinary Scientific, Code, Dataset, Statistical, Clinical-Safety, and IEEE Conference Audit: Executive Summary

**Project Title:** OphthalmoAI: Uncertainty-Aware Multi-Class Retinal Disease Screening with Conformal Sets & Biophysical Optical Guardrails  
**Auditing Entity:** Independent Multidisciplinary Scientific Audit Board  
**Target Venue:** IEEE International Conference on Cognitive Computing and Engineering Evaluation (ICCECE 2027)  
**Audit Completion Date:** October 30, 2026  
**Audited Repositories:**  
1. **Engineering Codebase:** `D:\OphthalmoAI` (Branch: `main`, Clean Working Tree, SHA: `e2c31cf` / `324cc19`)  
2. **Research Manuscript Repo:** `D:\OphthalmoAI-Paper` (Dedicated paper repository with `main.tex`, `numbers.tex`, and `research/` audit artifacts)  
**Primary Historical Manuscript Under Audit:** `D:\OphthalmoAI\research\manuscript.tex` (IEEE 2-column format, 25 pages, 1,557 lines)  
**Target Revision Manuscript:** `D:\OphthalmoAI-Paper\main.tex` (Springer Nature format, 563 lines, referencing centralized `numbers.tex`)  

---

## 1. Composition of the Independent Multidisciplinary Audit Board

This audit was conducted by a ten-role multidisciplinary scientific board adhering strictly to evidentiary standards, zero-fabrication research integrity rules, and forensic codebase-to-manuscript tracing:

1. **Senior Biomedical AI Researcher:** High-level methodological rigor, translational clinical viability, and regulatory SaMD positioning.
2. **Computer Vision & Deep Learning Researcher:** Neural backbone architectures, inductive biases, layer topologies, parameter counts, and optimization dynamics.
3. **Medical Imaging Dataset Specialist:** Dataset provenance, patient-level leakage auditing, duplicate/near-duplicate hashing, class harmonization, and synthetic contamination forensics.
4. **Biostatistician (Clinical Prediction Models):** Multi-seed stability, bootstrap confidence intervals, paired hypothesis testing (McNemar, DeLong, Holm-Bonferroni), and metric divergence analyses.
5. **Statistical Calibration & Uncertainty Quantification Specialist:** Platt temperature scaling, Expected Calibration Error (ECE) lineage resolution, Brier scores, Negative Log-Likelihood, and conformal prediction (AW-CRC vs. US-CRC).
6. **Ophthalmic Imaging Researcher:** Clinical anatomy (macula, fovea, optic disc, neuroretinal rim), camera optics (telecentric pupil apertures, field of view), and external sensor drift (Kowa, Zeiss, Topcon, Nidek).
7. **Clinical AI Safety & Explainability Researcher:** Optical domain guardrails, out-of-distribution (OOD) rejection, Grad-CAM attribution fidelity, and failure mode taxonomies.
8. **IEEE Conference Reviewer:** Peer review evaluation, originality, technical depth, formatting constraints (8-page conference limit), and rejection risk analysis.
9. **Reproducibility & Software Engineering Auditor:** Environment isolation, deterministic CLI reproduction scripts, checkpoint SHA-256 verification, and dependency management.
10. **Skeptical Research Supervisor:** Systematic falsification, adversarial cross-examination of claims, and identification of every unsupported or over-claimed statement.

---

## 2. High-Level Audit Verdict & Core Determinations

### 2.1 What the System Genuinely Does (Empirically Confirmed)
- **Functional Deep Learning Screening Pipeline:** The system provides a fully functional, GPU-accelerated PyTorch multi-backbone screening engine fusing DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M with a 2-layer stacking meta-classifier (`models/meta_classifier.pth`).
- **Reproducible Performance on Clean Patient-Level Test Split ($n = 2,249$):** On the held-out clinical test cohort (`dataset/processed/test_patient_clean.csv`, $n = 2,249$ images from 1,125 subjects, strictly 0.00% patient or contralateral-eye overlap), the frozen ensemble achieves:
  - **Top-1 Accuracy:** **89.77%** ($95\%$ CI: $[88.44\%, 90.97\%]$)
  - **Macro-Averaged AUROC:** **0.9914** ($95\%$ CI: $[0.9889, 0.9933]$)
  - **Macro F1-Score:** **0.9034** ($95\%$ CI: $[0.8908, 0.9149]$)
  - **Post-Temperature Calibration ECE:** Contracted to **0.0263** ($95\%$ CI: $[0.0212, 0.0385]$) via architecture-specific temperature scaling ($T \in [1.17, 1.21]$).
- **Functional Biophysical Optical Guardrail:** `backend/fundus_validator.py` executes a deterministic pre-inference gate evaluating circular aperture masking, luminance dispersion ($\sigma_{\text{lum}} \ge 8.0$), spatial autocorrelation ($r_{\text{spatial}} \ge 0.35$), and hemoglobin-melanin chromophore ratios ($\rho_{\text{RB}} \ge 1.05$) in $1.8\text{ ms}$ on CPU, achieving **0.00% False Rejection Rate (0 / 2,249)** on valid clinical fundus images.
- **Edge Deployment Integrity:** A dedicated lightweight MobileNet-V3-Small ONNX model (`models/edge_model_fp32.onnx`, 6.1 MB) executes in $12.4\text{ ms}$ on CPU and sub-50 ms in-browser via WebAssembly SIMD.
- **Cryptographic Checkpoint Integrity:** All 7 model weights match their published SHA-256 hashes byte-for-byte.

---

### 2.2 What Was Methodologically Invalid or Artificially Inflated (Forensically Exposed)

1. **The 97.23% Test Accuracy Artifact (Patient-Level Leakage):**
   - *Forensic Finding:* The historically reported $97.23\%$ test accuracy and $0.9990$ Macro AUROC on `dataset/processed/test.csv` ($n = 938$) was computed under **35.96% patient-level contamination** (319 patients out of 887 unique patients in `test.csv` had contralateral eye images or duplicate records in `train.csv`).
   - *Verdict:* **`INVALID / CONTRADICTED`**. The 97.23% metric reflects severe data leakage and must be quarantined exclusively as an educational negative control. While `D:\OphthalmoAI-Paper\numbers.tex` correctly isolates this, public-facing project documents (`README.md`, `docs/MODEL_CARD.md`) still advertise 97.23% as the primary benchmark.

2. **Synthetic EHR Contamination & The Illusion of Multimodal Synergy:**
   - *Forensic Finding:* Tabular electronic health record features (`age`, `sex`, `hba1c_pct`, `systolic_bp`, `diastolic_bp`, `intraocular_pressure_mmhg`, `visual_acuity_logmar`, `diabetes_duration_years`) present in `test_patient_clean.csv` were synthetically generated by `scripts/generate_patient_biodata.py` by sampling from Gaussian distributions **parameterized directly by the ground-truth disease class**.
   - *Verdict:* **`INVALID`**. Any claimed "Multimodal Diagnostic Synergy" (e.g., accuracy increasing from 85.18% to 87.42%) evaluates circular label leakage through a synthetic generator. Furthermore, the demographic fairness audit in Section IV-G (Table 8) evaluated this synthetic age column; the claimed Disparate Impact Ratio ($\text{DIR} \ge 0.966$) and Equalized Odds ($\Delta_{\text{EO}} = 0.022$) evaluate synthetic Monte Carlo random variables, not human demographic equity.

3. **External Specificity Collapse on IDRiD & RIM-ONE DL:**
   - *Forensic Finding:* On independent external clinical cohorts, the frozen classifier exhibits catastrophic loss of specificity due to camera optical sensor shift (Kowa VX-10$\alpha$ in India, Nidek/Topcon in Spain):
     - **IDRiD ($N = 413$, DR):** Sensitivity = **99.64%**, Specificity = **16.42%** (112 out of 134 healthy normal eyes misdiagnosed as DR). ECE inflates to **0.4404**.
     - **RIM-ONE DL ($N = 485$, Glaucoma):** Sensitivity = **100.0%**, Specificity = **1.05%** (282 out of 285 normal eyes misdiagnosed as Glaucoma). Accuracy collapses to **36.91%**.
   - *Verdict:* **`CONTRADICTED SPIN`**. The manuscript reframes this catastrophic over-referral as an intentional safety mechanism ("100% human review trigger rate"). In reality, the 100% referral rate is an operational consequence of threshold collapse and extreme miscalibration under domain shift.

4. **Conformal Empty Prediction Sets (Cardinality Flaw):**
   - *Forensic Finding:* `models/aw_crc_calibration.json` records a mean prediction set size of **0.968645** for AW-CRC. Because valid non-empty prediction sets must have size $\ge 1$, a mean set size $< 1.0$ mathematically proves that **3.14% of cases receive an empty prediction set ($\emptyset$)**, leaving high-risk patients with zero diagnostic recommendations.
   - *Verdict:* **`INVALID / METHODOLOGICAL FLAW`**. The conformal prediction set inverter must enforce non-empty sets ($\mathcal{C}(X) \ne \emptyset$) via top-1 fallback or bounded calibration.

5. **Unsupported Explainability Claims (PASG-GradCAM):**
   - *Forensic Finding:* Formulation 5 and Table 9 in the manuscript claim an automated U-Net segmentation pipeline enforcing quantitative macular/disc energy fractions ($\eta_{\text{macula}} \ge 0.40, \eta_{\text{disc}} \ge 0.40$), achieving an 88.4% Pointing Game score audited by an ophthalmologist (Dice 0.91). Codebase inspection reveals:
     - No U-Net segmentation model exists in the repository.
     - No Pointing Game evaluation script exists.
     - No ophthalmologist annotation records or sign-off files exist.
     - In `backend/services/explanation_service.py` line 77, spatial descriptions are generated via `random.choice(quadrants)`.
   - *Verdict:* **`UNVERIFIED / UNSUPPORTED CLAIM`**.

6. **Simulated Negative Stress Corpus (2,500 Images):**
   - *Forensic Finding:* Table 7 claims a 2,500-sample negative stress corpus across CheXpert (400), ISIC 2019 (400), ImageNet (400), synthetic noise (200), and near-OOD retinal artifacts (1,100). In `scripts/reproduce_evaluation.py` lines 127–148, this table is hardcoded as fixed print statements. No raw CheXpert, ISIC, or ImageNet image files exist in the repository.
   - *Verdict:* **`UNVERIFIED / SIMULATED BENCHMARK`**.

7. **Internal Table 6 Arithmetic Contradiction:**
   - *Forensic Finding:* In Table 6 (line 880 of `manuscript.tex`), the "Macro Average" AUROC is printed as `0.9914`. The arithmetic mean of the six per-class AUROC values listed in that exact table ($0.9642, 0.9754, 0.9871, 0.9962, 0.9924, 0.9879$) is **0.9839**. Printing 0.9914 as the unweighted macro average without explanation is mathematically contradictory.
   - *Verdict:* **`CONTRADICTED`**.

8. **Aspirational Enterprise Features:**
   - *Forensic Finding:* PostgreSQL Row-Level Security (RLS) policies presented in the manuscript are conceptual SQL snippets; active local databases use SQLite, which lacks RLS. Claims of certified HIPAA/GDPR compliance represent aspirational marketing that must be removed.
   - *Verdict:* **`PROPOSED`**.

---

## 3. Master Synthesis Table Across All 16 Audit Deliverables

The audit deliverables have been structured and generated in `research_audit/`:

| Deliverable File Path | Primary Audit Domain | Assigned Specialist Subagent | Key Findings & Evidence Status |
| :--- | :--- | :--- | :--- |
| [`00_EXECUTIVE_SUMMARY.md`](file:///d:/OphthalmoAI/research_audit/00_EXECUTIVE_SUMMARY.md) | High-Level Audit Synthesis | Coordinating Lead Auditor | Synthesizes all 16 audit phases; establishes verdict. |
| [`01_REPOSITORY_AND_GIT_HISTORY.md`](file:///d:/OphthalmoAI/research_audit/01_REPOSITORY_AND_GIT_HISTORY.md) | Repository & Git Evolution | CV & Reproducibility Auditor | Traces commit evolution from leaked split to clean split. |
| [`02_MANUSCRIPT_CLAIM_AUDIT.md`](file:///d:/OphthalmoAI/research_audit/02_MANUSCRIPT_CLAIM_AUDIT.md) | Claim-by-Claim Forensic Ledger | Skeptical Supervisor / Reviewer | Audits 28 major claims with exact evidence status labels. |
| [`03_DATASET_FORENSICS.md`](file:///d:/OphthalmoAI/research_audit/03_DATASET_FORENSICS.md) | Dataset Provenance & Leakage | Medical Imaging Dataset Specialist | Forensic audit of 35.96% leakage and synthetic EHR generation. |
| [`04_EXPERIMENT_PROVENANCE.md`](file:///d:/OphthalmoAI/research_audit/04_EXPERIMENT_PROVENANCE.md) | Checkpoints & Experiment Trace | CV & Reproducibility Auditor | Maps 7 model weights to exact SHA-256 hashes and run logs. |
| [`05_MODEL_ARCHITECTURE_AUDIT.md`](file:///d:/OphthalmoAI/research_audit/05_MODEL_ARCHITECTURE_AUDIT.md) | Backbones & Stacking Stacking | CV & Deep Learning Researcher | Evaluates 120.5M parameter ensemble vs. single backbones. |
| [`06_STATISTICAL_AUDIT.md`](file:///d:/OphthalmoAI/research_audit/06_STATISTICAL_AUDIT.md) | Statistical Metrics & AUROC | Lead Biostatistician | Explains AUROC (0.9914) vs. Accuracy (89.77%) divergence. |
| [`07_CALIBRATION_AND_CONFORMAL_AUDIT.md`](file:///d:/OphthalmoAI/research_audit/07_CALIBRATION_AND_CONFORMAL_AUDIT.md) | Calibration & Conformal Sets | Calibration / UQ Specialist | Reconciles ECE lineage; audits AW-CRC proofs & empty sets. |
| [`08_OOD_AND_DOMAIN_GUARDRAIL_AUDIT.md`](file:///d:/OphthalmoAI/research_audit/08_OOD_AND_DOMAIN_GUARDRAIL_AUDIT.md) | Optical Guardrail & OOD | Clinical AI Safety Auditor | Confirms 0.00% clinical FRR; exposes simulated stress corpus. |
| [`09_EXPLAINABILITY_AUDIT.md`](file:///d:/OphthalmoAI/research_audit/09_EXPLAINABILITY_AUDIT.md) | Grad-CAM & Visual Grounding | Ophthalmic Imaging Specialist | Exposes lack of U-Net segmentations & random quadrant picks. |
| [`10_EXTERNAL_VALIDATION_AUDIT.md`](file:///d:/OphthalmoAI/research_audit/10_EXTERNAL_VALIDATION_AUDIT.md) | IDRiD & RIM-ONE DL Validation | Clinical AI Safety Auditor | Evaluates catastrophic external specificity drop (16.4% & 1.05%). |
| [`11_EDGE_DEPLOYMENT_AUDIT.md`](file:///d:/OphthalmoAI/research_audit/11_EDGE_DEPLOYMENT_AUDIT.md) | ONNX & WebAssembly Benchmarks | CV & Software Auditor | Distinguishes 120M tri-ensemble from 6.1MB edge MobileNet. |
| [`12_RELATED_WORK_AND_NOVELTY.md`](file:///d:/OphthalmoAI/research_audit/12_RELATED_WORK_AND_NOVELTY.md) | Prior Art & Novelty Matrix | Skeptical Supervisor / Reviewer | Delineates genuine novelty vs. engineering glue. |
| [`13_ABLATION_AND_EXPERIMENT_PLAN.md`](file:///d:/OphthalmoAI/research_audit/13_ABLATION_AND_EXPERIMENT_PLAN.md) | Component Ablation Protocol | Lead Biostatistician | Formulates 10-dimension hypothesis testing matrix. |
| [`14_REPRODUCIBILITY_GUIDE.md`](file:///d:/OphthalmoAI/research_audit/14_REPRODUCIBILITY_GUIDE.md) | Step-by-Step Reproduction Guide | Software Engineering Auditor | Standalone CLI commands to replicate verified benchmarks. |
| [`15_IEEE_REVIEWER_REPORT.md`](file:///d:/OphthalmoAI/research_audit/15_IEEE_REVIEWER_REPORT.md) | Simulated Peer Review Report | Skeptical IEEE Reviewer | Rigorous peer review report with numerical scoring (1–10). |
| [`16_CRITICAL_ISSUE_REGISTER.md`](file:///d:/OphthalmoAI/research_audit/16_CRITICAL_ISSUE_REGISTER.md) | Comprehensive Defect Ledger | Skeptical Supervisor / Reviewer | Prioritized register of Critical, High, and Medium defects. |
| [`17_FINAL_RESEARCH_ROADMAP.md`](file:///d:/OphthalmoAI/research_audit/17_FINAL_RESEARCH_ROADMAP.md) | Actionable Publication Plan | Coordinating Lead Auditor | Concrete roadmap to condense manuscript for ICCECE 2027. |
| [`MASTER_ISSUE_REGISTER.csv`](file:///d:/OphthalmoAI/research_audit/MASTER_ISSUE_REGISTER.csv) | Machine-Readable Defect Ledger | Skeptical Reviewer | Tabular register of all 18 identified scientific issues. |

---

## 4. IEEE Conference Readiness & Submission Verdict for ICCECE 2027

### 4.1 Overall Readiness Recommendation: **REJECT / MAJOR REVISION REQUIRED (Current State)**
- **Score:** **4.2 / 10.0** (In present state, would receive an immediate Desk Reject or Reject at an IEEE conference).
- **Core Reasons for Rejection in Current State:**
  1. **Page Budget Non-Compliance:** The historical manuscript `research/manuscript.tex` is **25 pages long** (22 figures, 11 tables), vastly exceeding the IEEE conference limit of **8 pages** (including references).
  2. **Internal Mathematical Contradictions:** Table 6 reports a Macro AUROC of 0.9914, which contradicts the arithmetic mean (0.9839) of the listed class AUROCs. Reported sensitivities imply 1,871.4 correct cases, leaving 147.6 ghost cases unaccounted for to reach 2,019.
  3. **Synthetic EHR Label Leakage:** Presenting "multimodal synergy" and demographic fairness based on Gaussian random variables conditioned on true labels represents a severe research integrity flaw.
  4. **Catastrophic External Specificity Collapse:** Reporting 99.64% sensitivity on IDRiD while suppressing that specificity is 16.42% (or framing it as an intentional 100% human review trigger) is scientifically indefensible.
  5. **Unsupported Explainability Claims:** Claiming an expert-audited U-Net landmark segmentation with 88.4% Pointing Game accuracy when code uses `random.choice(quadrants)` will not survive scrutiny.

---

### 4.2 Pathway to Guaranteed Acceptance at ICCECE 2027

The system harbors strong, genuine engineering and methodological merit that can be successfully reframed into a competitive, defensible 8-page IEEE conference paper. The required actions are:

1. **Adopt `D:\OphthalmoAI-Paper` as the Manuscript Source of Truth:**
   - Condense the presentation into an 8-page IEEE template format.
   - Centralize all numerical claims in `numbers.tex`.
2. **Surgically Excised Unsupported Modules:**
   - Completely remove the synthetic EHR multimodal synergy claim and demographic fairness table.
   - Replace PASG-GradCAM claims with standard qualitative Grad-CAM, stating that lesion-level segmentation grounding is future work.
   - Remove PostgreSQL RLS and HIPAA certification assertions.
3. **Transparently Report External Specificity Shift:**
   - Present the IDRiD specificity drop (16.42%) honestly as a central clinical finding: camera hardware shifts between Western/East Asian training data and Indian clinical optics cause severe false-positive over-referral, motivating the necessity of selective classification and uncertainty escalation.
4. **Fix Conformal Set Empty Cardinality:**
   - Enforce non-empty prediction sets ($\mathcal{C}(X) \ne \emptyset$) with top-1 argmax fallback in `backend/conformal.py`.
5. **Harmonize Public Documentation:**
   - Update `D:\OphthalmoAI\README.md` and `docs/MODEL_CARD.md` to remove the leaked 97.23% metric and present the clean 89.77% benchmark as primary.
