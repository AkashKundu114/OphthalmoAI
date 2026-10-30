# Phase 0 & 2: Forensic Manuscript Claim Audit & Repository Reconciliation

**Audit Date:** October 30, 2026  
**Auditor:** Skeptical Research Supervisor, Senior Biomedical AI Researcher & Lead Reproducibility Auditor  
**Repository:** `https://github.com/AkashKundu114/OphthalmoAI`  
**Manuscript Under Audit:** `research/manuscript.tex` (25 Pages, IEEE Two-Column Format, 1,557 lines)  
**Target Conference:** IEEE International Conference on Cognitive Computing and Engineering Evaluation (ICCECE 2027)  

---

## 1. Executive Summary & Audit Methodology

This document executes an uncompromising, evidence-driven forensic audit of all scientific, clinical, mathematical, and algorithmic claims in the primary research manuscript (`research/manuscript.tex`) against the physical source code, dataset manifests, trained model checkpoints, and experimental run artifacts in the OphthalmoAI repository.

Every major claim has been cross-referenced against:
1. Ground-truth code implementations in `backend/` and `scripts/`.
2. Binary model checkpoints in `models/` verified by SHA-256 cryptographic hashes.
3. Raw and processed dataset manifests in `dataset/processed/`.
4. Live evaluation outputs, calibration registers, and benchmark logs (`models/*.json`, `REMEDIATION/logs/*.log`).

### Evidence Classification Taxonomy
Each claim is audited and assigned exactly one forensic status label:
- **`VERIFIED`**: The claim is completely supported by executable code, verified checkpoints, and authentic empirical data in the repository matching reported figures within rounding tolerance.
- **`PARTIALLY VERIFIED`**: The underlying mechanism or empirical result exists, but reported numbers have minor discrepancies, missing configuration documentation, or qualified operational constraints not disclosed in the text.
- **`UNVERIFIED`**: The claim describes an experiment, benchmark, human expert audit, or external dataset evaluation for which no source code, raw data, or verifiable execution log exists in the repository.
- **`CONTRADICTED`**: The repository contains direct, executable empirical evidence that refutes the manuscript's claim, or the manuscript contains internal mathematical self-contradictions.
- **`INVALID`**: The experimental methodology used to generate the result is scientifically flawed (e.g., severe data leakage, synthetic label injection, circular feature synthesis), rendering the claim scientifically invalid.
- **`PROPOSED`**: The claim describes architectural capabilities, regulatory certifications, or enterprise database schemas (e.g., PostgreSQL Row-Level Security, HIPAA multi-tenancy) that are presented as functioning system components but exist only as theoretical proposals or aspirational text snippets.

---

## 2. Complete Manuscript Claim Reconciliation Table (28 Major Claims)

| Claim ID | Manuscript Claim Statement | Section & Line | Current Implementation Evidence | Historical Experiment Evidence | Status Label | Forensic Finding & Critique |
| :---: | :--- | :---: | :--- | :--- | :---: | :--- |
| **CLM-01** | **Overall Model Diagnostic Performance**<br>"FP16 tri-backbone ensemble with learned stacking meta-classification achieves 89.77% Top-1 Accuracy, 0.9914 Macro AUROC, and ECE 0.0263." | Sec. IV-A<br>Lines 812–813;<br>Sec. IV-D, L887–898;<br>Table 6, L880 | `backend/services/inference_service.py`<br>`models/meta_classifier.pth` (SHA-256: `211cf76b...`) | `models/evaluation_meta_ensemble.json`<br>`models/calibration.json` | **VERIFIED** | Checkpoint execution on `test_patient_clean.csv` ($n=2,249$) yields exactly 89.7732% accuracy, 0.9914 Macro AUROC, and 0.0263 post-temperature ECE. |
| **CLM-02** | **Leaked Split (97.23%) vs. Clean Split (89.77%)**<br>"Naive unconstrained image split yielded 97.23% accuracy due to 36.0% patient leakage; clean patient-partitioned cohort yields 89.77%." | Sec. III-F<br>Lines 583–599;<br>Inventory L20–21;<br>Model Card L59 | `scripts/split_patient_level.py`<br>`dataset/processed/test.csv`<br>`dataset/processed/test_patient_clean.csv` | `models/evaluation_meta_ensemble.json`<br>`docs/MODEL_CARD.md`<br>`README.md` (Badge L13, Table L46) | **CONTRADICTED / INVALID** | While the paper claims to have eliminated leakage, public repository documents (`README.md`, `docs/MODEL_CARD.md`) still advertise the leaked 97.23% figure without disclosing that 319 patients (35.96%) overlapped between `train.csv` and `test.csv`. |
| **CLM-03** | **Zero-Leakage Contralateral Patient Partition**<br>"Master corpus of 58,166 samples partitioned into 53,667 train, 2,250 val, and 2,249 test with 0.00% patient leakage." | Sec. III-F<br>Lines 554, 573,<br>581, 595–599;<br>Table 5 | `dataset/processed/train_augmented_40k.csv` ($N=53,667$)<br>`val_patient_clean.csv` ($N=2,250$)<br>`test_patient_clean.csv` ($N=2,249$) | `scripts/split_patient_level.py`<br>`dataset/processed/master_patient_manifest.csv` ($N=22,388$) | **PARTIALLY VERIFIED** | Total sample counts ($53,667 + 2,250 + 2,249 = 58,166$) match row counts. However, patient IDs for 16,140 rows (`fundus_cfe_`) were generated by an unverified heuristic (`num // 2`) rather than genuine clinical patient linking. |
| **CLM-04** | **Cohort Attrition and Data Provenance**<br>"Initial pool of 33,807 raw images from ODIR-5K, APTOS-2019, Messidor-2, EyePACS-1, REFUGE, STARE reduced by 6,834 exclusions to 26,973 pristine base images." | Sec. III-F<br>Lines 563–573,<br>580–582;<br>Table 5 | `dataset/raw/` contains only `aptos_dr`, `color_fundus_eye`, `odir5k`, `preprocessed_fundus`. Messidor-2, EyePACS, REFUGE, STARE raw directories missing. | `scripts/prepare_fundus_dataset.py`<br>`scripts/ingest_color_fundus_eye.py` | **PARTIALLY VERIFIED** | Unaugmented clean corpus on disk totals 22,388 images (not 26,973). The referenced external cohorts (Messidor-2, EyePACS-1, REFUGE, STARE) were ingested from a pre-packaged third-party Kaggle archive without individual accession tracking. |
| **CLM-05** | **Biophysical Domain Guardrail Interception Rate**<br>"Pre-inference optical guardrail achieves 97.84% OOD rejection (2,446 / 2,500) and 0.00% clinical False Rejection Rate (0 / 2,249) in 1.8 ms on host CPU." | Sec. IV-F<br>Lines 937–959;<br>Table 7, L953 | `backend/fundus_validator.py`<br>`scripts/red_team_guardrail_test.py` | `scripts/reproduce_evaluation.py` lines 127–148 | **PARTIALLY VERIFIED** | Code in `backend/fundus_validator.py` executes in ~1.8 ms CPU. However, the 2,500 negative stress corpus (CheXpert, ISIC 2019, ImageNet) is hardcoded as fixed integer constants in `reproduce_evaluation.py`; raw CheXpert and ISIC files are not present in the repository. |
| **CLM-06** | **AW-CRC Emergency & Routine Conformal Coverage**<br>"AW-CRC achieves 98.80% coverage on sight-threatening emergencies, 95.16% on routine conditions, and restores Grade C coverage from 78.31% to 97.59%." | Sec. IV-G<br>Lines 1010–1012;<br>Table 8, L1014 | `backend/conformal.py`<br>`models/aw_crc_calibration.json` | `scripts/run_aw_crc_calibration.py`<br>`models/aw_crc_calibration.json` | **VERIFIED** | Quantiles ($\hat{q}_{\text{emerg}}=0.1296, \hat{q}_{\text{rout}}=0.1210$) and coverage rates match `aw_crc_calibration.json` exactly: emergency 98.8038%, routine 95.1579%, Grade C 97.5904%. |
| **CLM-07** | **Conformal Prediction Set Cardinality Flaw**<br>"AW-CRC maintains 96.86% singleton prediction sets while guaranteeing distribution-free coverage." | Sec. IV-G<br>Line 1012;<br>Table 8 | `backend/conformal.py`<br>`models/aw_crc_calibration.json` | `models/aw_crc_calibration.json` line 33: `"mean_set_size": 0.968645` | **CONTRADICTED / INVALID** | In `models/aw_crc_calibration.json`, `mean_set_size` is 0.9686. Because valid non-empty prediction sets must have size $\ge 1$, a mean set size $< 1.0$ proves that **3.14% of cases receive an empty prediction set ($\emptyset$)**, leaving critical patients with zero diagnostic candidates. |
| **CLM-08** | **Demographic Subgroup Fairness Parity**<br>"Disparate Impact Ratio DIR $\ge 0.966$ [0.925, 1.000] across age brackets, pigmentation tiers, and sensors; Equalized Odds disparity $\Delta_{\text{EO}} = 0.022 \le 0.05$." | Sec. IV-G<br>Lines 979–1008;<br>Table 8, L982 | `dataset/processed/test_patient_clean.csv`<br>`scripts/generate_patient_biodata.py` | `models/extended_clinical_battery_report.json` | **INVALID** | The demographic features (age, sex) in `test_patient_clean.csv` were synthetically generated by `generate_patient_biodata.py` by sampling from distributions conditioned directly on the true disease label! The fairness audit evaluates a synthetic generator with built-in label leakage. |
| **CLM-09** | **Multimodal Clinical Synergy Claim**<br>"Fusing fundus imagery with 12-dimensional EHR tabular bio-data improves classification accuracy from 85.18% to 87.42%." | Sec. IV-I<br>Lines 1071–1109;<br>Docs Gap L120 | `scripts/generate_patient_biodata.py`<br>`scripts/evaluate_extended_clinical_battery.py` lines 310–325 | `models/extended_clinical_battery_report.json` | **INVALID** | All 12 EHR features (HbA1c, BP, IOP, LogMAR, DM duration) are completely synthetic, generated by sampling Gaussian distributions parameterized by ground-truth disease class. Any reported multimodal gain is an artifact of circular label leakage. |
| **CLM-10** | **Epistemic-Aleatoric Dual-Uncertainty Decomposition**<br>"Dual-threshold policy uses Monte Carlo dropout ($S=20$, $p=0.20$) and Dirichlet vacuity $u_{\text{vacuity}} = K / S_{\text{dir}}$ to bifurcate clinical uncertainty." | Sec. III-D<br>Lines 464–510;<br>Algorithm 2, L492 | `backend/uncertainty.py`<br>`backend/evidential.py`<br>`scripts/train_evidential_meta.py` | `backend/uncertainty.py` line 18: `n_passes: int = 8`<br>Line 10: `DEFAULT_VACUITY_THRESHOLD = 0.40` | **PARTIALLY VERIFIED** | Core mathematical logic exists in `backend/uncertainty.py`. However, production default is $S=8$ passes (not $S=20$), vacuity threshold is $0.40$ (not $0.65$), and `CRITICAL_DIAGNOSES` in code is hardcoded to `{"Uveitis", "Jaundice", "Keratitis"}` (anterior conditions, not retinal fundus). |
| **CLM-11** | **Evidential Meta-Classifier Training Provenance**<br>"DirichletMetaClassifier trained with Asymmetric Clinical-Cost Loss (AC-HDL) on multi-backbone representations." | Sec. III-D<br>Lines 471–486 | `backend/evidential.py`<br>`scripts/train_evidential_meta.py` | `scripts/train_evidential_meta.py` lines 52–60 | **INVALID** | `train_evidential_meta.py` trains the evidential network on purely random synthetic vectors (`synthetic_inputs = torch.randn(2000, 18) * 0.5 + shift`), not on genuine image embeddings extracted from the patient cohort. |
| **CLM-12** | **Pixel-Aligned Saliency Grounding (PASG-GradCAM)**<br>"PASG-GradCAM constrains visual heatmaps, achieving Pointing Game Accuracy of 88.4% and Biomarker Energy Fraction of 62.8% on macular and disc ROIs." | Sec. III-E<br>Lines 513–524;<br>Sec. IV-H, L1048–1068;<br>Table 9 | `backend/services/explanation_service.py` | Codebase search: `pointing_game`, `energy_macula`, and U-Net segmentation code are completely absent. | **UNVERIFIED / PROPOSED** | `explanation_service.py` only implements standard unconstrained Grad-CAM from `pytorch_grad_cam`. There is no U-Net segmentation model, no pointing game evaluation script, and no quantitative energy calculation code in the repository. |
| **CLM-13** | **Expert Ophthalmologist Annotation Audit**<br>"To verify landmark fidelity, an expert ophthalmologist audited a 10% representative random sample of the cohort (Dice similarity 0.91 +/- 0.04)." | Sec. III-E<br>Line 524;<br>Sec. IV-H, L1049 | None | No clinical audit protocol, anonymized scoring sheets, or ophthalmologist sign-off files exist in `research/` or `docs/`. | **UNVERIFIED** | Entirely unverified claim. No clinician credentials, IRB protocol identifier, annotation logs, or inter-rater reliability records exist anywhere in the repository. |
| **CLM-14** | **Spatial Anatomical Description Generation**<br>"Generates grounded spatial anatomical descriptions linking visual activations to specific retinal quadrant landmarks." | Sec. III-E<br>Line 524 | `backend/services/explanation_service.py`<br>Lines 66–96 | `backend/services/explanation_service.py` line 77: `random.choice(quadrants)` | **CONTRADICTED** | Spatial descriptions are not grounded in visual attention; `generate_spatial_description()` randomly selects an anatomical quadrant from a hardcoded list using `random.choice(quadrants)`. |
| **CLM-15** | **Zero-Shot External Validation on IDRiD**<br>"Zero-shot external validation on IDRiD ($N=413$) demonstrates 99.64% sensitivity and 16.42% specificity." | Sec. IV-J<br>Lines 1110–1124;<br>Fig. 20 | `scripts/run_external_validation.py`<br>`scripts/evaluate_external_dataset.py` | `models/external_validation_report.json` lines 15–21 | **VERIFIED** | Exactly verified in `external_validation_report.json` ($N=413$, Sens: 99.64%, Spec: 16.42%). Confirms catastrophic external specificity collapse on Indian clinical camera optics (Kowa VX-10 $\alpha$). |
| **CLM-16** | **External Validation on RIM-ONE DL**<br>"Zero-shot validation on RIM-ONE DL ($N=485$) yields AUROC 0.6175, improving to 0.6577 under Reinhard color normalization." | Sec. IV-J<br>Lines 1121–1124 | `scripts/run_external_validation.py` | `models/external_validation_report.json` lines 192, 254 | **VERIFIED** | Exactly verified: Graham AUROC is 0.6175, Reinhard AUROC is 0.6577. Confirms near-zero external specificity for glaucoma screening (1.05% Graham, 2.81% Reinhard). |
| **CLM-17** | **Head-to-Head Comparison with RETFound**<br>"OphthalmoAI decisively outperforms fine-tuned RETFound ViT-Large (89.77% vs. 83.03%, DeLong p = 2.29e-15)." | Sec. IV-K<br>Lines 1125–1142;<br>Fig. 21 | `scripts/finetune_retfound_comparison.py` | `models/retfound_comparison_report.json` lines 8, 47 | **VERIFIED** | Script and JSON report confirm RETFound Top-1 accuracy is 83.03% (95% CI: [81.41%, 84.62%]), Macro AUROC is 0.9809, and DeLong $p$-value is $2.2866 \times 10^{-15}$. |
| **CLM-18** | **Edge Latency and Inference Throughput**<br>"ONNX FP16 runtime executes in 84.2 ms on edge GPU, with MobileNet-V3-Small client-side inference executing in <50 ms in-browser WASM." | Sec. III-B<br>Lines 237–242;<br>Sec. IV-E, L899–933;<br>Table 11, L1196 | `scripts/benchmark_latency.py`<br>`scripts/benchmark_onnx.py`<br>`frontend/public/models/edge_model.onnx` | `REMEDIATION/logs/benchmark_latency.log`<br>Tri-Ensemble CUDA: 77.51 ms median, 136.29 ms p95.<br>ONNX Edge: 1.17 ms CPU. | **PARTIALLY VERIFIED** | The lightweight MobileNetV3-Small ONNX model executes in ~1.2 ms on CPU / sub-50 ms in browser. However, the 89.77% accuracy was achieved by the 120.5M parameter tri-ensemble, NOT the lightweight edge model. |
| **CLM-19** | **Internal Table 6 Arithmetic Self-Contradiction**<br>"Macro Average AUROC in Table 6 is reported as 0.9914, representing the macro average across the six diagnostic conditions." | Sec. IV-C<br>Table 6, Line 880 | None (Direct mathematical contradiction in LaTeX source) | Arithmetic mean of per-class AUROCs in Table 6: `(0.9642 + 0.9754 + 0.9871 + 0.9962 + 0.9924 + 0.9879) / 6 = 0.983867` | **CONTRADICTED** | The arithmetic mean of the six per-class AUROCs in Table 6 is **0.9839**. Printing 0.9914 in the "Macro Average" row is a mathematical impossibility. Four of the six classes have AUROCs well below 0.9914. |
| **CLM-20** | **Discrepancy in Contingency Correct Classifications**<br>"Stacking meta-classifier produces 2,019 correct diagonal classifications on the held-out test cohort (2019 / 2249 = 89.77%)." | Sec. IV-B<br>Lines 822–825 | `test_patient_clean.csv` ($N=2,249$) | Sum of diagonal correct classifications from Table 6 sensitivities: $430(0.852) + 485(0.838) + 406(0.916) + 147(0.942) + 374(0.798) + 407(0.712) = 1,871.4$ | **CONTRADICTED** | Table 6 sensitivities imply only **1,871.4 correct classifications (83.21% accuracy)**, leaving **147.6 ghost correct cases** unaccounted for to reach the claimed 2,019 correct cases. |
| **CLM-21** | **Extended Clinical Battery Provenance (DCA, LR+, DOR)**<br>"Exhaustive clinical battery evaluates diagnostic likelihood ratios, DOR, and Decision Curve Analysis net benefit on n=2,249." | Sec. IV-I<br>Lines 1071–1109;<br>Table 10 | `scripts/evaluate_extended_clinical_battery.py` | `models/extended_clinical_battery_report.json` | **INVALID** | `evaluate_extended_clinical_battery.py` hardcodes the nominal sensitivity and specificity arrays, synthesizes probabilities using `rng.beta(9.5, 1.8)` simulations, and writes the report from simulated data rather than model forward passes. |
| **CLM-22** | **Multi-Seed Stability and McNemar Significance**<br>"Multi-seed accuracy across 5 seeds is 85.18% +/- 0.17%; McNemar test vs. DenseNet-201 yields chi^2 = 51.97, p = 5.63e-13." | Sec. IV-A<br>Table 4;<br>Docs Inventory L73 | `scripts/evaluate_multi_seed.py` | `models/multi_seed_statistical_report.json` | **INVALID** | `evaluate_multi_seed.py` hardcodes seed accuracy arrays (`[0.8518, 0.8539, ...]`) and manually constructs 2x2 contingency vectors with hardcoded cell counts on the old $n=938$ split rather than computing real predictions. |
| **CLM-23** | **PostgreSQL Multi-Tenant Row-Level Security (RLS)**<br>"Enterprise multi-hospital tenancy is enforced directly at the PostgreSQL database engine layer via Row-Level Security policies." | Sec. III-C<br>Lines 681–704;<br>Fig. 7 | `alembic/versions/` (Only `0001_initial_schema.py` exists)<br>`backend/db.py` | `scripts/init_db.sql` enables only `uuid-ossp` and `pgcrypto`. Zero RLS policy definitions exist. | **PROPOSED** | RLS code block in the manuscript is an illustrative SQL snippet. No RLS migrations or tenant isolation policies exist in Alembic or the backend codebase; active local DB is SQLite, which cannot execute PostgreSQL RLS. |
| **CLM-24** | **HIPAA & GDPR Enterprise Regulatory Compliance**<br>"Architecture complies with HIPAA and GDPR standards through cryptographic audit logging and role-based access control." | Sec. III-C<br>Lines 681–682;<br>Sec. V-F, L1288–1296 | `backend/security.py`<br>`backend/audit.py` | `REMEDIATION/CLAIMS_AUDIT.md` CLM-17: "All text claiming 'clinical-grade', 'FDA approved', or 'HIPAA certified' is removed or replaced with neutral scientific standards." | **CONTRADICTED / PROPOSED** | While security middleware and hash audits are implemented in `security.py`, claiming certified HIPAA/GDPR compliance for an experimental research codebase is an unsubstantiated regulatory overstatement directly contradicting the remediation audit. |
| **CLM-25** | **Content-Based Medical Image Retrieval (CBMIR)**<br>"Penultimate 512-dimensional EfficientNet-B4 embeddings indexed via FAISS enable sub-5 ms cosine similarity retrieval of historical reference cases." | Sec. III-E<br>Lines 525–535;<br>Fig. 3 | `backend/vector_search.py` | `backend/vector_search.py` implements cosine similarity search on embeddings. | **PARTIALLY VERIFIED** | Vector similarity logic exists in `vector_search.py`. However, production uses in-memory NumPy/Torch dot products rather than an inverted-file FAISS index, and the pre-computed embedding database contains synthetic test fixtures. |
| **CLM-26** | **Human-in-the-Loop Active Learning Triage**<br>"Conformal prediction sets with cardinality >1 or elevated vacuity trigger automated referral tickets securely streamed into an active learning reservoir." | Sec. III-D<br>Lines 705–714;<br>Fig. 8 | `backend/uncertainty.py` (`build_review_payload`) | `backend/routes/telehealth.py` | **PROPOSED** | The backend flags review triggers in JSON response payloads (`requires_human_review: true`), but the active learning reservoir, clinician feedback streaming loop, and model retraining pipeline are conceptual proposals not implemented in code. |
| **CLM-27** | **Quantization Impact on Accuracy**<br>"Graph optimization and FP16 half-precision quantization reduce latency by 2.4x with minimal accuracy degradation (-0.06% accuracy, -0.0004 AUROC)." | Sec. III-B<br>Lines 651–680;<br>Table 11, L1196 | `scripts/export_edge_model.py` | `models/evaluation_meta_ensemble_bf16.json` (Acc: 0.8971, AUROC: 0.9910) | **VERIFIED** | Checkpoint evaluation under FP16/BF16 demonstrates accuracy of 89.71% (-0.06% from 89.77%) and AUROC of 0.9910 (-0.0004 from 0.9914), confirming minimal numerical degradation. |
| **CLM-28** | **Paired McNemar Significance of AW-CRC vs. US-CRC**<br>"A paired McNemar test on the test cohort confirms AW-CRC successfully captured ground truth in 21 cases where US-CRC failed (b=0, c=21, p=1.0e-6)." | Sec. IV-G<br>Line 1046 | `scripts/run_aw_crc_calibration.py`<br>`models/aw_crc_calibration.json` | `models/aw_crc_calibration.json` lines 10, 31 | **VERIFIED** | Under US-CRC, Grade C emergency coverage is lower ($78.31\%$), missing 21 cases that AW-CRC captures with widened quantiles, satisfying exact one-sided McNemar binomial probability $p = (1/2)^{21} \approx 4.77 \times 10^{-7} < 1.0 \times 10^{-6}$. |

---

## 3. Deep-Dive Forensic Analyses of Core Problem Areas

### 3.1 The 97.23% vs. 89.77% Dual-Reporting & Patient Leakage Audit
- **The Core Flaw:** The naive split (`train.csv` $n=4,374$, `test.csv` $n=938$) was generated by unconstrained random image splitting. Forensic cross-referencing against `dataset/processed/master_patient_manifest.csv` reveals that **319 patients out of 887 unique patients in `test.csv` (35.96%) also had images in `train.csv`**.
- **Impact on Reported Metrics:**
  - Random Leaked Split ($n=938$): **97.23% Top-1 Accuracy**, **0.9990 Macro AUROC**, ECE 0.0884.
  - Strict Patient-Clean Split ($n=2,249$): **89.77% Top-1 Accuracy** (Learned Meta) / **89.42%** (Soft Voting), **0.9914 Macro AUROC**, ECE 0.0263.
- **Repository Conflict:** While `manuscript.tex` correctly reports the clean 89.77% figure, public-facing project files (`README.md` badge line 13, summary table line 46, and `docs/MODEL_CARD.md` line 59) still showcase the leaked **97.23%** metric as the primary internal benchmark. This creates a severe transparency and reproducibility conflict between the public repository and the scientific manuscript.

### 3.2 Synthetic EHR Contamination & The Illusion of Multimodal Synergy
- **The Core Flaw:** The tabular features in `test_patient_clean.csv` (`age`, `sex`, `hba1c_pct`, `systolic_bp`, `diastolic_bp`, `intraocular_pressure_mmhg`, `visual_acuity_logmar`, `diabetes_duration_years`, `smoking_status`) were created by `scripts/generate_patient_biodata.py`.
- **Mechanism of Circularity:** In `generate_patient_biodata.py` lines 107–150, the feature generator accepts `disease_class` as input and queries `CLINICAL_PRIORS[disease_class]`:
  - If `disease_class == "Diabetic Retinopathy"`, `hba1c` is sampled from $\mathcal{N}(8.6, 1.45)$ and `diabetes_duration_years` from $\mathcal{N}(13.4, 6.2)$.
  - If `disease_class == "Glaucoma"`, `iop` is sampled from $\mathcal{N}(24.8, 4.6)$.
  - If `disease_class == "Hypertensive Retinopathy"`, `systolic_bp` is sampled from $\mathcal{N}(162.0, 17.5)$.
  - If `disease_class == "Normal"`, `hba1c` is sampled from $\mathcal{N}(5.2, 0.35)$ and `systolic_bp` from $\mathcal{N}(118.0, 10.5)$.
- **Scientific Impact:**
  1. Any claim of "Multimodal Diagnostic Synergy" (e.g., accuracy increasing from 85.18% to 87.42% in Section IV-I) is an experimental artifact of evaluating a neural network on features that directly leak the target class.
  2. The demographic subgroup fairness audit in Section IV-G (Table 8) evaluated age cohorts ($<50$, $50-65$, $>65$) using this synthetic `age` column. The claimed Equalized Odds disparity ($\Delta_{\text{EO}} = 0.022$) and Disparate Impact Ratio ($\text{DIR} \ge 0.966$) evaluate synthetic Monte Carlo distributions, not human patient equity.

### 3.3 Internal Mathematical & Statistical Inconsistencies
- **Table 6 Macro AUROC Arithmetic Impossibility:**
  In Table 6 (line 880 of `manuscript.tex`), the "Macro Average" AUROC is printed as `\textbf{0.9914}`. However, the arithmetic mean of the six per-class AUROC values listed in that exact table is:
  $$\frac{0.9642 + 0.9754 + 0.9871 + 0.9962 + 0.9924 + 0.9879}{6} = \frac{5.9032}{6} = 0.983867 \approx 0.9839$$
  Four of the six individual diagnostic categories have AUROCs well below 0.9914. Reporting 0.9914 as the Macro Average without explicit weighting disclosure is mathematically contradictory.
- **Contingency Table vs. Sensitivity Mismatch (147.6 Ghost Correct Cases):**
  Section IV-B (line 824) asserts:
  $$\text{Accuracy} = \frac{2,019}{2,249} = 89.7732\% \approx 89.77\%$$
  However, multiplying the exact per-class sensitivities from Table 6 by their class sample sizes yields:
  - Normal: $430 \times 0.852 = 366.36$
  - Diabetic Retinopathy: $485 \times 0.838 = 406.43$
  - Glaucoma: $406 \times 0.916 = 371.90$
  - Cataract: $147 \times 0.942 = 138.47$
  - AMD: $374 \times 0.798 = 298.45$
  - HR / Myopia: $407 \times 0.712 = 289.78$
  $$\text{Total Diagonal Correct} = 366.36 + 406.43 + 371.90 + 138.47 + 298.45 + 289.78 = 1,871.39 \text{ cases}$$
  $$\text{Implied Overall Accuracy} = \frac{1,871.39}{2,249} = 83.21\%$$
  There is an unexplained deficit of **$2,019 - 1,871.4 = 147.6$ correct cases**. The per-class sensitivities in Table 6 cannot produce an overall accuracy of 89.77%.

### 3.4 Conformal Prediction (AW-CRC) Empty Set Cardinality Flaw
- **The Empirical Evidence:** In `models/aw_crc_calibration.json`, the calibrated results for AW-CRC record:
  - `"mean_set_size": 0.9686450167973124`
  - `"singletons_pct": 96.86450167973123`
  - `"overall_cov": 0.9686450167973124`
- **Clinical Implication:** In conformal prediction, set cardinality is an integer $k \in \{0, 1, 2, \dots, K\}$. If 96.86% of sets have size 1 and the mean set size is exactly 0.9686, then **all remaining sets (3.14% of the cohort, or ~71 patients) have size 0 ($\mathcal{C}(X) = \emptyset$)**.
- An empty conformal set indicates that no class satisfied the non-conformity threshold. While mathematically valid under conformal risk control, in a clinical screening workflow, producing an empty set leaves an attending clinician with zero diagnostic guidance. The manuscript claims 96.86% singleton coverage but conceals that 3.14% of predictions are empty sets.

### 3.5 External Specificity Collapse (IDRiD & RIM-ONE DL)
- **IDRiD ($N=413$, Diabetic Retinopathy):**
  - Sensitivity: **99.64%** (278 / 279 DR cases detected)
  - Specificity: **16.42%** (Only 22 / 134 normal scans correctly classified; 112 false positives)
  - False Positive Rate: **83.58%**
- **RIM-ONE DL ($N=485$, Glaucoma):**
  - Graham preprocessing: Sensitivity 100.0%, Specificity **1.05%** (3 / 285 normal scans correct; 282 false positives; AUROC 0.6175)
  - Reinhard normalization: Sensitivity 99.38%, Specificity **2.81%** (8 / 285 normal scans correct; 277 false positives; AUROC 0.6577)
- **Clinical Diagnosis:** The model suffers from severe optical domain shift when exposed to external camera hardware (Kowa VX-10 $\alpha$ on IDRiD; Nidek AFC-210 / Topcon TRC-NW400 on RIM-ONE DL). While sensitivity is high, the collapse in specificity means the system would refer $>97\%$ of healthy community patients to specialist ophthalmologists, overwhelming healthcare capacity.

### 3.6 Explainability: Theoretical Grounding vs. Code Reality
- **Manuscript Formulation 5 & Table 9 Claim:**
  "Pixel-Aligned Saliency Grounding (PASG-GradCAM)" using an automated U-Net segmentation pipeline to enforce energy constraints ($\eta_{\text{macula}} \ge 0.40, \eta_{\text{disc}} \ge 0.40$), audited by an expert ophthalmologist on a 10% sample (Dice $0.91 \pm 0.04$), achieving 88.4% Pointing Game accuracy.
- **Repository Code Reality:**
  1. `backend/services/explanation_service.py` executes vanilla `GradCAM` from the open-source `pytorch_grad_cam` package on the penultimate layer of `EfficientNet-B4`.
  2. No U-Net segmentation weights or inference routines exist.
  3. No Pointing Game evaluation script exists.
  4. In `explanation_service.py` lines 66–96, the anatomical lesion localization string is generated by:
     ```python
     quadrants = ["superotemporal macular arcade", "inferonasal quadrant near optic disc", "foveal center", ...]
     return f"microaneurysms, dot-blot hemorrhages... localized along the {random.choice(quadrants)}"
     ```
     Anatomical quadrant descriptions are selected at random from a Python list using `random.choice()`.

---

## 4. Summary of Claim Status Distribution

```
Total Distinct Claims Audited: 28
=====================================================
[VERIFIED]            :  8 claims (28.6%)
[PARTIALLY VERIFIED]  :  6 claims (21.4%)
[UNVERIFIED]          :  2 claims ( 7.1%)
[CONTRADICTED]        :  4 claims (14.3%)
[INVALID]             :  5 claims (17.9%)
[PROPOSED]            :  3 claims (10.7%)
=====================================================
Uncompromised Scientific Integrity Index: 50.0% (14 / 28 Verified or Partially Verified)
Methodological Risk Index: 50.0% (14 / 28 Require Mandatory Revision, Retraction, or Reframing)
```

---

## 5. Mandatory Authorship Remediation Directives

1. **Purge Leaked 97.23% Metric:** Immediately update `README.md` and `docs/MODEL_CARD.md` to display the verified patient-clean metric (**89.77% Top-1 Accuracy**, 0.9914 Macro AUROC). Retract all promotional claims of 97.23% accuracy.
2. **Retract Multimodal & Demographic Fairness Claims:** Remove Section IV-I (Multimodal Synergy) and Table 8's demographic fairness claims from the main diagnostic evaluation. Explicitly disclose in supplementary materials that tabular EHR data was synthetically generated.
3. **Reconcile Table 6 Mathematics:**
   - Correct the Macro Average row in Table 6 to report the true arithmetic OvR mean (**0.9839**), or clearly define and document the macro-weighting formula used to reach 0.9914.
   - Resolve the discrepancy between Table 6 per-class sensitivities (which sum to 1,871.4 correct cases, or 83.21%) and the overall 2,019 correct cases (89.77%).
4. **Disclose Empty Conformal Sets:** Amend Section IV-G to acknowledge that AW-CRC yields 3.14% empty prediction sets ($\mathcal{C}(X) = \emptyset$), and document the clinical fallback policy (mandatory human triage) for empty sets.
5. **Frame External Specificity Collapse Transparently:** Reposition the IDRiD and RIM-ONE DL external validation results as a central clinical safety finding, demonstrating why selective prediction and confidence escalation are required when deploying across diverse optical sensors.
6. **Align Explainability Claims with Code:** Reframe PASG-GradCAM and U-Net energy ratios as a proposed theoretical framework, or remove Table 9 until the U-Net segmentation pipeline and ophthalmologist audit data are fully implemented and committed to the repository. Replace `random.choice()` in `explanation_service.py` with deterministic bounding-box attention.
7. **Prune Proposed Enterprise Features:** Remove theoretical SQL snippets for PostgreSQL Row-Level Security, active learning streaming loops, and formal HIPAA/GDPR certifications from the peer-reviewed conference paper. Focus strictly on the algorithmic screening pipeline.
