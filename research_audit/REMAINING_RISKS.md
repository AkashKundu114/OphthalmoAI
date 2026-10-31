# Remaining Risks, Clinical Boundaries & Research Limitations

**Document Version:** 2.0  
**Audit Cycle:** Phase 10 Autonomous Remediation Loop  
**Date:** October 31, 2026  
**Status:** Canonical Clinical Risk Assessment for Peer-Reviewed Submission  

---

## 1. Executive Clinical Risk Summary

While the software engineering defects (ISS-19 through ISS-25) have been surgically resolved and verified with 100% test pass rates across the 317-test suite, several foundational scientific, clinical, and data-domain limitations remain intrinsic to retrospective ophthalmic research. 

This document provides a transparent, adversarial ledger of open risks that cannot be eliminated solely through code patches.

---

## 2. Intrinsic Scientific & Clinical Risks

### Risk 1: Unverified Patient-Level Bilateral Independence on Web-Scraped Cohorts (CFE & ODIR)
- **Severity:** HIGH (Methodological Limitation)
- **Evidence:** Source datasets downloaded from public web archives (`Peacein/color-fundus-eye` and ODIR-5K) permanently lack hospital electronic health record (EHR) accession numbers. In legacy scripts (`split_patient_level.py`), synthetic subject IDs were generated via `num // 2` on sequential image filenames.
- **Remediation Status:** In `scripts/split_patient_level_v2.py`, identity confidence is downgraded to `UNKNOWN` / `IMAGE_LEVEL_ONLY`. Perceptual hash clustering (dHash Hamming distance $\le 2$) is now enforced via Union-Find (DSU), eliminating near-duplicate visual overlap.
- **Remaining Risk:** Contralateral eyes (right eye OD and left eye OS) or longitudinal photographs of the same patient from different clinic visits may have distinct perceptual hashes (>2 Hamming distance) and cross partition boundaries, creating subtle, undetectable latent patient-level leakage.
- **Clinical Governance:** The paper must explicitly designate partitions as **Perceptual-Cluster-Stratified** rather than clinically verified patient-level splits.

---

### Risk 2: Catastrophic Specificity Collapse Under Unseen Fundus Cameras (IDRiD & RIM-ONE DL)
- **Severity:** CRITICAL (Clinical Safety & False Positive Burden)
- **Evidence:** External validation on out-of-distribution camera cohorts (`models/external_validation_report.json`) revealed severe specificity breakdown:
  - **IDRiD (Kowa VX-10 camera, India):** Specificity = **16.42%** (112 out of 134 normal scans misclassified as diseased).
  - **RIM-ONE DL (Nidek AFC-210 / Topcon TRC-NW400, Spain):** Specificity = **1.05%** (282 out of 285 normal scans misclassified as glaucoma; 99% false alarm rate).
  - **Conformal Coverage:** Dropped to **0.00%** on both external cohorts!
- **Fallacious Claim Refutation:** The manuscript previously asserted that *"a 100% human referral rate proves external clinical safety."* In clinical reality, a 1.05% specificity causes acute alert fatigue, floods referral clinics with hundreds of panicked, healthy patients, and would paralyze public health screening programs.
- **Mandatory Clinical Protocol:** The system must never be deployed on an uncalibrated camera model. Pre-deployment site acceptance testing (local few-shot calibration or unsupervised domain adaptation) is strictly mandatory before clinical operation.

---

### Risk 3: Biophysical Optical Guardrail Discard Bias (Pathology Exclusion Risk)
- **Severity:** HIGH (Diagnostic Selection Bias)
- **Evidence:** Tier-0 optical admissibility (`backend/fundus_validator.py`) deterministically rejects low-contrast, low-luminance, or blurry images ($S(X) < 0.50$, Grade C). Under severe overexposure (gain $\times 2.2$), false rejection reached **36.7%**.
- **Clinical Hazard:** Optical media opacities (dense cataracts, vitreous hemorrhage from proliferative DR, corneal edema from acute angle-closure glaucoma) directly degrade fundus image quality. Silently rejecting Grade C scans risks systematically discarding the patients in most urgent need of ophthalmic intervention.
- **Mitigation:** The clinical routing engine must categorize any biophysically rejected scan as *"Uninterpretable — Suspected Media Opacity (Mandatory Slit-Lamp / Indirect Ophthalmoscopy Referral)"*, rather than a benign upload error.

---

### Risk 4: Low Prevalence Support for Age-Related Macular Degeneration (AMD)
- **Severity:** MEDIUM (Statistical Precision)
- **Evidence:** In `dataset/processed/test_v2.csv` ($N = 1,389$), following the exclusion of diagnostic conflations (CSCR, Macular Scar, Retinitis Pigmentosa), ground-truth AMD prevalence is only $n = 26$ (1.87% of test samples).
- **Statistical Implication:** While Top-1 sensitivity on AMD is 96.15% (25/26), the 95% Wilson binomial confidence interval spans $[81.1\%, 99.3\%]$.
- **Governance:** The manuscript must explicitly disclose the wide confidence interval for AMD and avoid claiming definitive screening parity with common conditions like Diabetic Retinopathy ($n = 482$).

---

### Risk 5: Dual-Tier Architecture Model Conflation
- **Severity:** MEDIUM (Systems Transparency)
- **Evidence:** The repository maintains two completely separate neural architectures:
  1. **Server Tri-Backbone Ensemble:** DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M (124.3M parameters, 484MB weights, requires server GPU or multi-core host).
  2. **Edge ONNX Model:** MobileNetV3-Small (2.5M parameters, 5.82MB FP32 / 1.62MB INT8, 2.54 ms P50 latency on CPU).
- **Governance:** The two models must never be conflated. Performance metrics (98.70% accuracy, 0.9985 AUROC) belong strictly to the 124M server ensemble, while latency metrics (< 3 ms, 355 QPS) belong strictly to the 2.5M edge model. Performance claims must never be transferred between them.

---

### Risk 6: Retrospective In-Silico Evaluation vs. Prospective Diagnostic Readiness
- **Severity:** CRITICAL (Regulatory & Medical Scope)
- **Evidence:** All evaluations in this repository are retrospective benchmarks on static, curated photographic datasets.
- **Regulatory Truth:** OphthalmoAI is an **investigational decision-support research prototype**, NOT a certified medical device (FDA Class II/III, EU MDR Class IIa/IIb, or CDSCO Class C).
- **Prohibition:** Under no circumstances should this system be described as clinically validated for autonomous medical diagnosis. It is strictly a triage and secondary reader decision-support prototype.

---

## 3. Summary Risk Register

| Risk ID | Hazard Description | Root Cause | Impact | Residual Status |
| :---: | :--- | :--- | :--- | :---: |
| **RSK-01** | Contralateral Eye Split Leakage | Kaggle CFE lacks patient IDs | Latent inflation of test metrics | Managed via dHash DSU; documented as limitation |
| **RSK-02** | External Specificity Collapse | Optical shift on unseen cameras | 99% false alarm rate on external cohorts | Managed via 100% referral; local calibration required |
| **RSK-03** | Optical Discard Bias | Media opacities rejected as bad images | Severe cataracts/bleeding dropped | Mitigated via mandatory specialist referral protocol |
| **RSK-04** | Small AMD Test Support ($n=26$) | Clinical harmonization exclusions | Wide binomial confidence intervals | Disclosed in paper_revised.tex Table 3 |
| **RSK-05** | Dual-Tier Conflation | Conflating edge vs server models | Misleading capability claims | Distinctly separated in Section III and Table 8 |
| **RSK-06** | Premature Diagnostic Claims | Retrospective in-silico benchmarks | Regulatory / patient safety risk | Framed strictly as triage decision-support prototype |

---

*This document must accompany all manuscript submissions, code releases, and clinical evaluation protocols.*
