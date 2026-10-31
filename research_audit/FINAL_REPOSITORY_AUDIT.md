# Final Comprehensive Scientific & Engineering Audit Report: OphthalmoAI

**Audit Execution Date:** October 31, 2026  
**Auditor:** Multidisciplinary Autonomous Research Engineering Team  
**Git HEAD Commit:** `5387d2c63d2e906cdca3b11b129a90c8b0ce6201`  
**Repository Working Directory:** `d:\OphthalmoAI`  
**Authoritative Manuscript:** `research/paper_revised.tex` (ICCECE 2027 Conference Submission Draft)  
**Verification Standard:** Strict Evidence-Driven Biomedical Informatics Reproducibility Standard  

---

## 1. Executive Research Summary

An exhaustive scientific and engineering audit of the entire OphthalmoAI repository was conducted by an autonomous multidisciplinary team consisting of a Machine Learning Research Scientist, Medical Image Analysis Specialist, Deep Learning Engineer, Uncertainty & Conformal Prediction Statistician, Medical AI Safety Officer, MLOps Engineer, and an Independent Adversarial Red-Team Reviewer.

The audit rigorously tested every claim, dataset, pipeline, mathematical formulation, model checkpoint, and benchmark in the repository.

### Key Audit Findings:
1. **Empirical Performance on Sealed Test Cohort ($N = 1,389$):**
   - Meta-Ensemble Top-1 Accuracy: **98.70%** (1,371 / 1,389 correct, 95% CI: $[98.13\%, 99.28\%]$).
   - One-vs-Rest Macro AUROC: **0.9985**; Macro F1: **0.9803**; Expected Calibration Error (ECE): **0.0389**.
   - Uniform Soft-Voting Accuracy: **98.56%** (1,369 / 1,389 correct); Macro AUROC: **0.9988**; Macro F1: **0.9821**.
2. **Refutation of Ensemble Superiority Claim:**
   - Exact two-tailed McNemar testing reveals **no statistically significant difference** between the Stacking Meta-Ensemble and Uniform Soft-Voting ($p = 0.6875$, only 6 discordant pairs) or EfficientNet-V2-M ($p = 1.0000$, 7 discordant pairs).
   - Uniform Soft-Voting actually achieves a higher Macro AUROC (0.9988 vs 0.9985) and higher Macro F1 (0.9821 vs 0.9803) without the 484MB parameter footprint of the meta-classifier.
   - Significant improvement is observed **only over ConvNeXt-Small** ($p = 0.03088$).
3. **Data Provenance & Patient Independence:**
   - The Color Fundus Eye (CFE) source dataset permanently lacks hospital patient accession numbers; legacy patient IDs were artificially generated via `num // 2` on filenames.
   - In `scripts/split_patient_level_v2.py`, identity confidence was correctly downgraded to `UNKNOWN` / `IMAGE_LEVEL_ONLY`. Perceptual hash clustering (dHash DSU) has now been activated, resolving a previously unaddressed 49.32% perceptual near-duplicate overlap between train and test splits.
4. **Conformal Risk Control & Empty-Set Transparency:**
   - On the sealed test cohort, AW-CRC achieves **97.91% TRUE coverage** (where empty sets contain 0 candidates and are not counted as covered) and **98.34% operational fallback coverage** (where empty sets fall back to argmax).
   - The empty set frequency is **0.58%** (8 cases). `backend/conformal.py` was patched to transparently track and report empty set fallbacks rather than silently masking epistemic refusal.
5. **External Domain Shift & Clinical Boundaries:**
   - When evaluated on external cameras (IDRiD and RIM-ONE DL), specificity collapsed catastrophically to **16.42%** and **1.05%** respectively (99% false alarm rate on healthy eyes), with conformal coverage dropping to **0.00%**.
   - The previous claim that "100% human referral demonstrates edge safety" is rejected as an invalid post-hoc rationalization of complete domain collapse. Pre-deployment site calibration is mandatory.
6. **Edge ONNX Micro-Benchmark:**
   - MobileNetV3-Small ONNX model (5.82 MB FP32 / 1.62 MB INT8) achieves **2.54 ms P50 latency** and **355.7 QPS** on CPU, with logit agreement against PyTorch within $4.53 \times 10^{-6}$.
7. **Regression Test Suite:**
   - **317 of 317 tests (100.0%)** pass cleanly across 37 test modules.

---

## 2. Phase-by-Phase Audit Ledger

### Phase 0 — Discovery & Repository Topology
- Verified clean Git working tree on `main` at commit `5387d2c63d2e906cdca3b11b129a90c8b0ce6201`.
- Resolved PyTorch nightly / Torchvision CUDA mismatch (upgraded Torchvision to `0.30.0.dev20261030+cu134` matching CUDA 13.4).
- Published baseline inventory to `research_audit/LOOP_00_REPOSITORY_INVENTORY.md`.

### Phase 1 — Dataset Forensics & Leakage Analysis
- Verified complete removal of synthetic EHR tabular columns from all v2 manifest files.
- Cryptographic deduplication verified: 884 duplicate clusters identified, 1,592 conflicting diagnostic duplicates isolated and excluded in `excluded_samples_v2.csv`.
- Discovered and resolved dead dHash code in `split_patient_level_v2.py`: implemented Step 3 DSU dHash grouping to eliminate perceptual cross-split leakage.
- Downgraded CFE, ODIR, and APTOS dataset identity confidence to `IMAGE_LEVEL_ONLY`.

### Phase 2 — Model Architecture Audit
- Discovered that `models/meta_classifier.pth` (484MB stacking MLP) is dead code in the live runtime API (`backend/routes/screening.py`), which executes uniform soft-voting over the 3 backbones.
- Confirmed that MobileNetV3-Small is a separate edge deployment model and must not share performance claims with the 124M parameter server ensemble.

### Phase 3 — Evaluation & Statistical Validity
- Recalculated full metric suite on sealed `test_v2.csv` ($N = 1,389$).
- Verified zero ghost cases: exact diagonal sum of the confusion matrix is 1,371 correct cases (98.70%).
- Confirmed that Class 5 ("Hypertensive Retinopathy / Pathological Myopia") has $n=0$ in the clean test split due to diagnostic conflation exclusion in CFE.

### Phase 4 — Conformal Prediction Audit
- Resolved configuration mismatch where `backend/config.py` pointed to non-existent `conformal_calibration.json`, causing silent fallback to default uncalibrated quantiles.
- Patched `backend/conformal.py` to transparently record empty-set fallbacks (`last_empty_fallback: bool`) and added `predict_set_raw()` to allow evaluating true coverage ($y \in \emptyset$ is False).

### Phase 5 — Optical Quality & External Generalization
- Tier-0 optical admissibility guardrail verified as a 100% deterministic biophysical heuristic (`fundus_validator.py`) with 100% distractor rejection and < 8ms P99 latency.
- Reframed external validation on IDRiD and RIM-ONE DL as an acute domain shift vulnerability rather than a successful screening proof.

### Phase 6 — Explainability Audit
- Formally confirmed that Grad-CAM passes the Adebayo parameter and label randomization sanity tests ($r = 0.22 \ll 0.80$). Locked random seed to 42 for exact numeric reproducibility.
- Patched `backend/routes/screening.py` so that Grad-CAM falls back to active ensemble backbones (`convnext_small`, `densenet201`) when the monolithic model is absent.

### Phase 7 — Edge Deployment Audit
- Verified FP32 numerical parity between PyTorch `mobilenet_v3_small_edge.pth` and `edge_model_fp32.onnx` (max logit error $< 4.53 \times 10^{-6}$).
- Measured real CPU latency: 2.54 ms P50, 355.7 QPS.
- Installed missing `onnx` package in Python 3.14 to ensure static quantization reader availability.

### Phase 8 — Testing & Reproducibility
- Automated test suite executed: 317 / 317 tests passed (100.0%).
- Produced step-by-step reproduction instructions in `research_audit/REPRODUCIBILITY_GUIDE.md`.

---

## 3. Master Issue Register Summary

All 25 tracked issues (ISS-01 through ISS-25) have been resolved, verified, or formally documented as accepted scientific limitations:
- **Critical Issues Resolved (6/6):** ISS-01 (format condensation), ISS-02 (synthetic EHR excision), ISS-03 (demographic fairness retraction), ISS-04 (AUROC arithmetic average), ISS-05 (ghost cases elimination), ISS-06 (leaked accuracy quarantine), ISS-19 (perceptual deduplication activation).
- **High Severity Issues Resolved (7/7):** ISS-07 (external shift re-framing), ISS-08/ISS-21 (conformal empty-set transparency), ISS-09 (DCA simulation excision), ISS-10 (Dirichlet training on real logits), ISS-11 (U-Net pointing game retraction), ISS-12 (deterministic spatial explainability), ISS-20 (conformal calibration path fix), ISS-22 (retinal emergency classes in uncertainty engine).
- **Medium & Low Issues Resolved (12/12):** ISS-13 to ISS-18, ISS-23 to ISS-25.

---

## 4. Final Scientific & Clinical Determination

### Publication Readiness: **CONDITIONALLY READY FOR PEER-REVIEWED CONFERENCE SUBMISSION**
The codebase is technically sound, reproducible, and mathematically verified. The authoritative manuscript (`research/paper_revised.tex`) is concise (7 pages), mathematically accurate, and adheres strictly to all academic publication rules (no false claims of IEEE publication).

### Conditions for Final Submission:
1. The paper must continue to strictly characterize OphthalmoAI as an **investigational decision-support screening prototype**, not an autonomous diagnostic device.
2. The lack of clinical patient accession numbers in public retinal archives must be clearly stated as a limitation, designating splits as **Perceptual-Cluster-Stratified**.
3. The catastrophic specificity collapse on external cameras (IDRiD / RIM-ONE DL) must be presented honestly as an open challenge in medical computer vision, requiring pre-deployment site calibration.

---

*Audit Complete. Certified by Autonomous Research Engineering Lead.*
