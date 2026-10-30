# Phase 16: Critical Scientific Issue Register & Vulnerability Ledger

**Audit Date:** October 30, 2026  
**Auditor:** Skeptical Research Supervisor, Senior Biomedical AI Researcher & Lead Reproducibility Auditor  
**Repositories:** `D:\OphthalmoAI` (Engineering/Models) and `D:\OphthalmoAI-Paper` (Research Manuscript)  
**Target Submission:** ICCECE 2027 / Target Biomedical Informatics Journal  

---

## 1. Executive Summary & Issue Severity Definitions

This register catalogs all scientific, statistical, methodological, algorithmic, and presentation vulnerabilities identified across the OphthalmoAI codebase, dataset manifests, and research manuscript (`research/manuscript.tex`).

### Severity Classification Scheme
- **`CRITICAL`**: Fundamental scientific, statistical, or methodological flaw that invalidates published claims, violates ethical research standards (e.g., circular label leakage, data fabrication/simulation passed off as empirical data), or represents a fatal disqualifier (e.g., severe format violation leading to immediate desk rejection).
- **`HIGH`**: Significant empirical discrepancy, unverified major technical claim, or severe clinical safety hazard (e.g., catastrophic external false-positive collapse, empty conformal prediction sets, ghost classifications) that must be resolved before peer-reviewed publication.
- **`MEDIUM`**: Inconsistent parameter documentation, hardcoded test harness constants, or architectural over-claiming that degrades reproducibility and scientific rigor.
- **`LOW / COSMETIC`**: Typographic inconsistencies, minor rounding variances, or non-critical formatting issues.

---

## 2. Master Synthesis of Identified Scientific Issues

| Issue ID | Severity | Category | Issue Title & Brief Summary | Affected Manuscript Section | Affected Repository File | Scientific & Clinical Impact | Mandatory Resolution |
| :---: | :---: | :--- | :--- | :---: | :--- | :--- | :--- |
| **ISS-01** | **CRITICAL** | Scope / Format | **Fatal 25-Page Length Exceeding 8-Page Conference Limit** | Entire Manuscript (25 Pages) | `research/manuscript.tex`<br>`research/manuscript.pdf` | Immediate desk rejection at ICCECE 2027 for formatting violation. Compendium conflates ML algorithms, clinical trials, hardware silicon, and database schemas. | Execute surgical structural condensation to 8 pages following the page budget allocation, or redirect to a high-capacity journal venue. |
| **ISS-02** | **CRITICAL** | Methodology | **Circular Label Leakage via Synthetic EHR Bio-Data** | Sec. IV-I (L1071–1109)<br>Sec. I (L85) | `scripts/generate_patient_biodata.py`<br>`dataset/processed/test_patient_clean.csv` | Synthetic EHR features (HbA1c, BP, IOP, LogMAR) were sampled from distributions conditioned directly on ground-truth disease classes. Claimed multimodal synergy (+2.24% acc) is completely invalid. | Permanently retract all multimodal synergy claims and remove synthetic EHR evaluations from the primary manuscript. |
| **ISS-03** | **CRITICAL** | Fairness / Ethics | **Demographic Subgroup Fairness Audited on Synthetic Data** | Sec. IV-G (L971–1008)<br>Table 8 (L982) | `scripts/generate_patient_biodata.py`<br>`dataset/processed/test_patient_clean.csv` | Table 8 claims demographic parity ($\text{DIR} \ge 0.966$, $\Delta_{\text{EO}} = 0.022$) across age brackets ($<50$, $50-65$, $>65$). Auditing fairness on a synthetic, label-correlated `age` column evaluates a random generator, not human patient equity. | Retract Table 8 and all demographic fairness parity claims from the primary paper. Disclose synthetic nature of biodata in supplementary materials. |
| **ISS-04** | **CRITICAL** | Mathematics | **Arithmetic Mean Contradiction in Table 6 Macro AUROC** | Sec. IV-C<br>Table 6 (Line 880) | `research/manuscript.tex`<br>`models/extended_clinical_battery_report.json` | Table 6 prints Macro Average AUROC as 0.9914, whereas the unweighted arithmetic mean of the six per-class AUROCs in that table is 0.9839. Four of the six classes are well below 0.9914. | Amend Table 6 Macro Average to report the true arithmetic OvR mean (0.9839), or explicitly define and justify the macro-weighting formula. |
| **ISS-05** | **CRITICAL** | Mathematics | **Contingency Sum Mismatch & 147.6 Ghost Correct Cases** | Sec. IV-B (L822–825)<br>Sec. IV-C (Table 6) | `research/manuscript.tex`<br>`models/extended_clinical_battery_report.json` | Manuscript asserts 2,019 correct diagonal cases (89.77% acc), but multiplying Table 6 sensitivities by class supports yields only 1,871.4 correct cases (83.21% acc)—an unexplained deficit of 147.6 correct cases. | Reconcile the exact contingency table and confusion matrix diagonal values with reported per-class sensitivities. |
| **ISS-06** | **CRITICAL** | Transparency | **Dual-Reporting of Leaked 97.23% Metric in Public Repo** | Sec. III-F (L583–599)<br>Public README / Card | `README.md` (Badge L13, Table L46)<br>`docs/MODEL_CARD.md` (Line 59) | While the manuscript correctly reports the clean 89.77% benchmark, public repository files still boast the leaked 97.23% figure without disclosing 35.96% patient overlap, misleading external researchers. | Update all public repository documentation and badges to display the clean 89.77% benchmark and explicitly document the 97.23% partition as a quarantined negative control. |
| **ISS-07** | **HIGH** | Clinical Safety | **Catastrophic External Specificity Collapse (IDRiD & RIM-ONE)** | Sec. IV-J (L1110–1124)<br>Fig. 20 | `scripts/run_external_validation.py`<br>`models/external_validation_report.json` | Specificity collapses to 16.42% on IDRiD and 1.05% on RIM-ONE DL due to camera optical domain shift. In real screening, referring >97% of normal eyes would overwhelm clinical capacity and cause severe harm. | Eliminate claims of autonomous clinical readiness; elevate this failure mode as a central clinical safety finding demonstrating the necessity of selective prediction and local calibration. |
| **ISS-08** | **HIGH** | Clinical Safety | **Conformal Set Emptiness Flaw in AW-CRC (3.14% Null Sets)** | Sec. IV-G (L1010–1012)<br>Table 8 | `backend/conformal.py`<br>`models/aw_crc_calibration.json` | AW-CRC mean set size is 0.9686, proving that 3.14% of patients receive an empty prediction set ($\mathcal{C}(X) = \emptyset$). Producing empty sets in clinical screening leaves clinicians with zero diagnostic guidance. | Transparently disclose the 3.14% empty set rate; formalize and benchmark an explicit clinical fallback rule (mandatory immediate escalation to human ophthalmic review). |
| **ISS-09** | **HIGH** | Reproducibility | **Simulated Script Execution in Extended Clinical Battery** | Sec. IV-I (L1071–1109)<br>Table 10, Fig. 18 | `scripts/evaluate_extended_clinical_battery.py`<br>`models/extended_clinical_battery_report.json` | `evaluate_extended_clinical_battery.py` hardcodes nominal sensitivities/specificities and simulates confidences using `rng.beta(9.5, 1.8)` rather than executing genuine model forward passes over image tensors. | Refactor `evaluate_extended_clinical_battery.py` to compute diagnostic odds ratios, likelihood ratios, and DCA net benefit directly from real checkpoint predictions. |
| **ISS-10** | **HIGH** | Reproducibility | **Evidential Meta-Classifier Trained on Synthetic Random Noise** | Sec. III-D (L471–486) | `scripts/train_evidential_meta.py`<br>`backend/evidential.py` | `train_evidential_meta.py` generates training data via `torch.randn(2000, 18) * 0.5 + shift` rather than extracting true multi-backbone feature representations from the clinical image dataset. | Re-train the Dirichlet evidential meta-classifier on true penultimate embeddings extracted from the patient training cohort. |
| **ISS-11** | **HIGH** | Explainability | **Unsupported Saliency Grounding Claims (PASG-GradCAM)** | Sec. III-E (L513–524)<br>Sec. IV-H (Table 9) | `backend/services/explanation_service.py` | Formulation 5 and Table 9 claim a U-Net segmentation pipeline, 88.4% Pointing Game benchmark, and 10% ophthalmologist audit (Dice 0.91). None of these components exist in the repository code. | Remove Table 9 and point-game claims from the primary paper, or commit the physical U-Net segmentation weights, evaluation scripts, and anonymized clinician audit sheets. |
| **ISS-12** | **HIGH** | Explainability | **Random Quadrant Selection in Anatomical Descriptions** | Sec. III-E (Line 524) | `backend/services/explanation_service.py`<br>Lines 66–96 | `generate_spatial_description()` selects anatomical quadrant strings using `random.choice(quadrants)`. Saliency descriptions are literally random strings, not grounded visual attention. | Replace `random.choice()` with deterministic spatial centroid extraction mapping high-activation Grad-CAM bounding boxes to true anatomical quadrant coordinates. |
| **ISS-13** | **MEDIUM** | Reproducibility | **Hardcoded Negative Stress Corpus in Guardrail Evaluation** | Sec. IV-F (L937–963)<br>Table 7 | `scripts/reproduce_evaluation.py`<br>Lines 127–148 | The 2,500 negative stress corpus (CheXpert, ISIC 2019, ImageNet) reported in Table 7 is hardcoded as fixed integers in `reproduce_evaluation.py`; raw CheXpert and ISIC files are missing from the repo. | Commit sample subsets of the negative stress corpus and implement live execution of the biophysical guardrail over these physical files. |
| **ISS-14** | **MEDIUM** | Data Provenance | **Artificial Modulo Heuristic for Patient ID Clustering** | Sec. III-F (L583–599)<br>Table 5 | `scripts/split_patient_level.py`<br>`dataset/processed/master_patient_manifest.csv` | Patient IDs for 16,140 rows (`fundus_cfe_`) were assigned using `num // 2` on filenames rather than verified hospital patient IDs. True bilateral independence across these images is unconfirmed. | Document the exact provenance and limitations of the Kaggle CFE dataset; perform visual perceptual hash audits to confirm bilateral pairing accuracy. |
| **ISS-15** | **MEDIUM** | Systems / Scope | **Conflation of Edge Model (MobileNet) vs. Heavy Ensemble** | Sec. III-B (L237–242)<br>Sec. IV-E (L899–933) | `scripts/benchmark_latency.py`<br>`frontend/public/models/edge_model.onnx` | Manuscript reports 89.77% accuracy and in-browser WASM latency (<50 ms) in the same breath, obscuring that the 89.77% accuracy was achieved by a 120.5M ensemble (84 ms GPU), not the edge model. | Clearly delineate the dual-tier architecture: heavy tri-backbone ensemble for centralized GPU inference vs. lightweight distilled MobileNetV3-Small for in-browser client triage. |
| **ISS-16** | **MEDIUM** | Governance | **Aspirational PostgreSQL Row-Level Security in Paper** | Sec. III-C (L681–704)<br>Fig. 7 | `scripts/init_db.sql`<br>`backend/db.py`<br>`alembic/versions/` | Manuscript presents PostgreSQL RLS SQL listings and claims HIPAA/GDPR compliance. Zero RLS policies exist in Alembic or backend code; local DB runs on SQLite (which lacks RLS support). | Remove PostgreSQL RLS listings and regulatory certification claims from the academic screening paper. Document RLS as an enterprise infrastructure guideline in deployment docs. |

---

## 3. Detailed Forensic Deep-Dives by Issue Category

### 3.1 Category 1: Experimental Validity & Data Leakage (ISS-02, ISS-03, ISS-06)
- **Forensic Diagnosis:** The repository contains two distinct manifestations of data contamination:
  1. *Patient-Level Overlap (Naive Split):* Between `train.csv` and `test.csv`, 319 patients out of 887 unique test patients (35.96%) crossed split boundaries. This produced an artificial test accuracy of 97.23% (AUROC 0.9990). While the manuscript authors recognized this and created `test_patient_clean.csv` (89.77% accuracy), public project files still advertise 97.23%.
  2. *Circular Synthetic EHR Injection:* In `generate_patient_biodata.py`, 12 tabular features were sampled from Gaussian distributions whose parameters were conditioned directly on the ground-truth disease class. When these tabular features were evaluated in multimodal fusion and demographic fairness audits, the neural networks evaluated an explicit label leak.
- **Scientific Impact:** Completely discredits the multimodal synergy narrative (+2.24% accuracy) and invalidates Table 8's demographic fairness audit ($\text{DIR} \ge 0.966$, $\Delta_{\text{EO}} = 0.022$).

### 3.2 Category 2: Internal Mathematical & Statistical Consistency (ISS-04, ISS-05, ISS-09, ISS-10)
- **Forensic Diagnosis:** Primary diagnostic tables contain direct mathematical contradictions:
  - *Macro AUROC Arithmetic Contradiction:* Table 6 reports Macro Average AUROC as 0.9914, but the arithmetic mean of the six per-class AUROCs in that exact table (0.9642, 0.9754, 0.9871, 0.9962, 0.9924, 0.9879) is **0.9839**.
  - *Contingency Discrepancy:* Section IV-B claims 2,019 correct classifications out of 2,249 ($89.77\%$), but multiplying Table 6 sensitivities by class supports yields only 1,871.4 correct cases ($83.21\%$), leaving **147.6 ghost cases** unaccounted for.
  - *Simulation-Driven Reporting:* Both `evaluate_extended_clinical_battery.py` (DCA net benefit, likelihood ratios) and `train_evidential_meta.py` execute on hardcoded nominal constants or random Gaussian noise rather than real extracted image representations.

### 3.3 Category 3: Clinical Safety & Generalization Hazards (ISS-07, ISS-08)
- **Forensic Diagnosis:**
  - *External Specificity Collapse:* On IDRiD (India), specificity drops to 16.42% (83.58% false-positive rate). On RIM-ONE DL (Spain), specificity drops to 1.05% under Graham and 2.81% under Reinhard normalization. The model over-refers nearly all healthy external eyes.
  - *Empty Conformal Sets:* In `aw_crc_calibration.json`, mean set size is 0.9686, mathematically proving that 3.14% of cases produce an empty prediction set ($\mathcal{C}(X) = \emptyset$). In clinical screening, outputting an empty set leaves a frontline health worker with no actionable triage guidance.

### 3.4 Category 4: Explainability & Technical Rigor (ISS-11, ISS-12, ISS-13, ISS-14, ISS-15, ISS-16)
- **Forensic Diagnosis:**
  - *Phantom Explainability:* Formulation 5 claims "Pixel-Aligned Saliency Grounding (PASG-GradCAM)", a U-Net segmentation pipeline, an 88.4% Pointing Game benchmark, and a 10% ophthalmologist audit (Dice 0.91). None of these components exist in code; `explanation_service.py` executes vanilla Grad-CAM and selects quadrant descriptions via `random.choice(quadrants)`.
  - *Aspirational Enterprise Features:* Presenting PostgreSQL Row-Level Security policies and HIPAA certifications in an academic screening paper when the local codebase runs on SQLite and lacks RLS migrations represents severe over-claiming.

---

## 4. Prioritized Remediation Action Plan

```
====================================================================================================
PRIORITY 1: IMMEDIATE SCIENTIFIC RETRACTIONS & PURGING (Pre-Submission Mandate)
====================================================================================================
1. Retract Multimodal Synergy: Permanently delete Section IV-I multimodal claims from manuscript.
2. Retract Demographic Fairness Audit: Remove Table 8; disclose synthetic biodata in supplementary.
3. Reconcile Table 6 Mathematics: Update Macro Average to 0.9839; resolve 147.6 ghost cases.
4. Align Public Repositories: Replace 97.23% badge/claims in README.md with clean 89.77% benchmark.
5. Disclose Empty Conformal Sets: Document 3.14% empty set rate and formalize human triage fallback.

====================================================================================================
PRIORITY 2: STRUCTURAL CONDENSATION & RE-SUBMISSION PREPARATION (8-Page IEEE Fit)
====================================================================================================
6. Condense Manuscript to 8 Pages: Follow the Phase 14 8-page budget allocation.
7. Evict Enterprise Relational Database Schemas: Remove PostgreSQL RLS listings and HIPAA claims.
8. Remove Theoretical Explainability: Remove Table 9 PASG benchmark until code is committed.
9. Fix Saliency Text Generation: Replace random.choice() in explanation_service.py with deterministic logic.

====================================================================================================
PRIORITY 3: EXPERIMENTAL CODE HARDENING & REPRODUCIBILITY (Post-Submission Validation)
====================================================================================================
10. Refactor Clinical Battery Scripts: Compute DCA and likelihood ratios from live model checkpoints.
11. Re-Train Evidential Meta-Classifier: Train Dirichlet network on true image embeddings.
12. Formalize Negative Stress Corpus: Commit physical test images for CheXpert and ISIC guardrail checks.
====================================================================================================
```
