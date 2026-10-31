# OphthalmoAI Remaining Risks and Open Scientific Limitations

**Document ID:** `research_audit/REMAINING_RISKS.md`  
**Execution Date:** October 30, 2026  
**Auditor / Risk Engineer:** Clinical AI Safety Engineer & Biomedical Risk Analyst  
**Remediation Baseline Commit:** `373bc4c263559b1184f293a14fb2cde15327039d`

---

## 1. Executive Risk Posture

OphthalmoAI has completed its engineering and statistical remediation: synthetic circular label leakage was eradicated, duplicate clusters were purged, models were re-evaluated on a sealed clean test set (n=1,389), optical guardrails fail closed, and conformal prediction guarantees empirical coverage.

However, in accordance with clinical AI safety best practices, technical fixes do not eliminate real-world biological and translational limitations. This document records all remaining risks and irreducible constraints that must be disclosed in research publications and regulatory documentation.

---

## 2. Risk Taxonomy and Registry

### Risk R-01: True Patient-Level Bilateral Independence on `color_fundus_eye`
- **Severity:** MEDIUM (Controlled)
- **Domain:** Data Provenance / Statistical Generalization
- **Finding:** The public source `Peacein/color-fundus-eye` ($n = 16,140$ raw images) lacks authenticated hospital EHR identifiers or bilateral eye tags (`_OD`/`_OS`).
- **Remediation Implemented:** Executed cryptographic SHA-256 and perceptual hash clustering (`scripts/forensic_deduplication_v2.py`) to group near-identical images and exclude conflicting duplicates. Partitions were generated using GroupKFold to guarantee that no duplicate cluster spans splits.
- **Accepted Limitation:** While identical and near-identical images are strictly quarantined, true patient-level contralateral independence cannot be proven without authentic hospital metadata for this sub-cohort. It is formally classified as `DUPLICATE_CLUSTER_SEPARATED` rather than `VERIFIED_PATIENT_ID`.

### Risk R-02: Camera Optical Domain Shift on External Validation Cohorts
- **Severity:** HIGH (Inherent Clinical Challenge)
- **Domain:** Generalization / Translation
- **Finding:** External validation on IDRiD (Kowa VX-10 alpha, India) and RIM-ONE DL (Nidek AFC-210, Spain) reveals substantial sensitivity/specificity trade-offs under distribution shift. When evaluated under naive multiclass argmax, specificity on non-disease controls dropped to 16.42% (IDRiD) and 1.05% (RIM-ONE DL).
- **Remediation Implemented:** 
  1. Patched `domain_adaptation.py` to apply foreground retinal masking (`gray > 15.0`) for color constancy normalization.
  2. Evaluated datasets under disease-specific binary decision thresholds rather than naive argmax.
  3. AW-CRC conformal triage correctly identifies out-of-distribution scans and triggers clinician review on 100% of shifted scans (`human_review_trigger_rate: 1.0`).
- **Accepted Limitation:** OphthalmoAI cannot be claimed to be plug-and-play generalizable across all fundus camera manufacturers without local calibration or site-specific fine-tuning.

### Risk R-03: Lack of Prospective Randomized Clinical Trial
- **Severity:** MEDIUM
- **Domain:** Regulatory / Translational Readiness
- **Finding:** All evaluations are conducted on retrospective public benchmark repositories.
- **Accepted Limitation:** Retrospective benchmark accuracy does not equal clinical diagnostic efficacy. OphthalmoAI must be described strictly as an *"experimental research screening prototype, not a certified medical device"* in all user-facing documentation, UI notices, and research manuscripts.

### Risk R-04: Non-Clinical Synthetic Tabular Features (Permanently Quarantined)
- **Severity:** RESOLVED & MONITORED
- **Domain:** Scientific Validity / Data Leakage
- **Status:** Synthetic biodata generation (`scripts/generate_patient_biodata.py`) has been quarantined with explicit warnings. All v2 manifests and production models operate in a 100% vision-only regime. The multimodal fusion model is archived as a negative study in the paper integration queue.

### Risk R-05: Saliency Map Clinical Interpretation
- **Severity:** LOW (Controlled)
- **Domain:** Explainability / Attribution
- **Finding:** Saliency maps (Grad-CAM) highlight convolutional feature activations, not histopathologic lesion boundaries.
- **Remediation Implemented:**
  1. Passed Adebayo parameter randomization sanity check ($r = -0.0818 \pm 0.1581$, proving parameter dependence).
  2. Excised `random.choice(quadrants)` and replaced anterior segment hallucinations in `backend/biomarker_extractor.py` with genuine posterior pole features and scientific disclaimers.
- **Accepted Limitation:** Saliency maps must never be presented to clinicians as segmented lesion masks or causal explanations of disease etiology.

---

## 3. Summary of Operational Guardrails

| Guardrail Subsystem | Mechanism | Fail-Safe Behavior | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **Optical Domain Validator** | Biophysical tissue spectroscopy & geometry check | HTTP 422 Fail-Closed (no inference) | `test_predict_fails_closed_on_validator_exception_with_422` PASSED |
| **Non-Fundus Rejection** | Multi-channel variance & autocorrelation | 100% Rejection of X-ray, OCT, documents | `guardrail_stress_benchmark.json` (100% Spec) |
| **Conformal Prediction Triage** | AW-CRC with urgency stratification | Dynamic set expansion / Argmax fallback | `aw_crc_empirical_benchmark.json` (99.31% Emerg Cov) |
| **Edge / Server Decoupling** | MobileNetV3-Small ONNX vs Heavy Ensemble | Standalone low-latency WASM pipeline | `onnx_parity_benchmark.json` (1.57 ms, 636.7 QPS) |
| **Multi-Tenancy** | Request header validation | Fail-closed zero rows on empty tenant | `test_boundary_conditions.py` PASSED |
