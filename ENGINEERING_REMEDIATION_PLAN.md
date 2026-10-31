# OphthalmoAI Comprehensive Engineering Remediation Plan

**Document ID:** `ENGINEERING_REMEDIATION_PLAN.md`  
**Execution Context:** Autonomous Multi-Agent Engineering Audit & Remediation Loop  
**Git HEAD:** `b744414`  
**Lead Architect:** Software Architect & Remediation Orchestrator  
**Status:** COMPLETE & FULLY EXECUTED  

---

## 1. Executive Remediation Architecture

The remediation strategy for OphthalmoAI addressed 22 audited issues across data provenance, model architecture, calibration mathematics, clinical safety, API security, and academic reproducibility.

```mermaid
flowchart TD
    subgraph Phase0["Phase 0: Forensic Quarantine"]
        Quarantine["Quarantine 8,477 Leaked / Conflicting Samples<br/>(excluded_samples_v2.csv)"]
        ExciseEHR["Excise Synthetic EHR Columns<br/>(manifest_v2.csv & splits)"]
    end

    subgraph Phase1["Phase 1: Pipeline Hardening"]
        DSU["Activate DSU dHash Grouping<br/>(split_patient_level_v2.py)"]
        ZeroLeakage["Verify Zero Split Overlap<br/>(test_splits_zero_overlap PASSED)"]
    end

    subgraph Phase2["Phase 2: Mathematical & Code Remediation"]
        CalibPath["Fix Calibration Path & Fallback<br/>(aw_crc_calibration.json)"]
        EmptySet["Implement Empty-Set Tracking<br/>(last_empty_fallback & triage)"]
        CUDA["Resolve PyTorch / Torchvision CUDA Mismatch<br/>(CUDA 13.4 nightly)"]
        ONNX["Install & Wire ONNX Engine<br/>(edge_model_fp32.onnx, 1.48 ms)"]
    end

    subgraph Phase3["Phase 3: Scientific & Clinical Alignment"]
        McNemar["Concede McNemar Stacking Parity<br/>(p = 0.6875 vs Soft-Voting)"]
        DomainShift["Disclose External Camera Collapse<br/>(16.42% IDRiD, 1.05% RIM-ONE)"]
        AMD["Disclose AMD Sample Limitation<br/>(n = 29, 95% Wilson CI)"]
    end

    subgraph Phase4["Phase 4: Full Verification"]
        Pytest["317 / 317 Regression Tests PASSED"]
        RedTeam["Survive Hostile Red-Team Review"]
    end

    Phase0 --> Phase1 --> Phase2 --> Phase3 --> Phase4
```

---

## 2. Itemized Remediation Action Plan

### Remediation Pillar A: Dataset Decontamination & Partition Hygiene (Issues ISS-02, ISS-03, ISS-10, ISS-14, ISS-19, ISS-21)
1. **Quarantine Conflicting-Label Duplicates:**
   - Authored a 27-rule clinical harmonization matrix (`label_harmonization.csv`).
   - Quarantined 1,592 conflicting-label duplicate images and 6,885 clinically incompatible mergers (CSCR, RP, macular scar, retinal detachment) to `dataset/processed/excluded_samples_v2.csv`.
2. **Purge Synthetic Data Leakage:**
   - Excised all 8 label-conditioned Gaussian features (`hba1c_pct`, `systolic_bp`, etc.) from `train_v2.csv`, `validation_v2.csv`, `calibration_v2.csv`, `test_v2.csv`, and `manifest_v2.csv`.
   - Verified by automated unit test `test_no_synthetic_ehr_in_v2_manifests` (PASSED).
3. **Activate DSU Perceptual Near-Duplicate Clustering:**
   - Modified `scripts/split_patient_level_v2.py` to merge identical 64-bit difference hashes into unified `group_id`s before splitting.
   - Enforced `StratifiedGroupKFold` on unified group IDs, ensuring zero sample, zero group, and zero hash overlap across splits (`test_splits_zero_overlap` PASSED).
4. **Honest Patient Identity Classification:**
   - Updated `split_patient_level_v2.py` and manuscript to designate the CFE sub-cohort as `UNKNOWN / IMAGE_LEVEL_ONLY`, formally labeling the partition as Duplicate-Cluster-Separated.

---

### Remediation Pillar B: ML Engineering & Inference Runtime (Issues ISS-08, ISS-15, ISS-17, ISS-18, ISS-22)
1. **PyTorch / Torchvision CUDA Compatibility:**
   - Upgraded torchvision to `0.30.0.dev20261030+cu134` to match PyTorch CUDA 13.4.
   - Resolved runtime DLL crashes.
2. **ONNX Package & Edge Runtime Wiring:**
   - Installed `onnx-1.23.2`.
   - Connected `models/edge_model_fp32.onnx` as `EDGE_ONNX_MODEL_PATH` in `backend/onnx_inference.py`.
   - Corrected default input tensor shape in `run_comprehensive_benchmark` from `(1, 3, 380, 380)` to `(1, 3, 224, 224)` matching MobileNetV3-Small native geometry.
3. **Conformal Calibration Path & Empty-Set Fallback:**
   - Updated `backend/config.py` and `backend/conformal.py` to load `models/aw_crc_calibration.json`.
   - Implemented `last_empty_fallback: bool` tracking in `AWConformalPredictor` and exposed `predict_set_raw()` to distinguish true mathematical coverage ($97.91\%$) from fallback coverage ($98.34\%$).
4. **Parameter Volume & Checkpoint Provenance Reconciliation:**
   - Documented exact parameter counts in `MODEL_AND_TRAINING_AUDIT.md`: 120.43M fine-tuned server ensemble (ImageNet base 124.3M), 2.54M edge MobileNetV3-Small.
   - Updated `research/RESULTS_PROVENANCE.md` to reference the active checkpoint `models/meta_classifier_gpu.pth` (462.42 MB, SHA-256 `06452a81...`).

---

### Remediation Pillar C: Statistical & Clinical Integrity (Issues ISS-04, ISS-05, ISS-07, ISS-09, ISS-11, ISS-16)
1. **Contingency Matrix & Ghost Case Eradication:**
   - Recalculated live confusion matrix on sealed test set ($N = 1,389$): diagonal sum ($1,371$) + errors ($18$) = $1,389$ total. Zero ghost cases.
2. **McNemar Stacking Parity Concession:**
   - Conducted paired two-tailed exact binomial McNemar test between Meta-Ensemble and Uniform Soft-Voting: $b=4, c=2, p = \mathbf{0.6875}$.
   - Conceded empirical parity in `paper_revised.tex` and recommended soft-voting as a robust non-parametric gold standard.
3. **Cross-Camera Domain Shift Disclosure:**
   - Documented catastrophic specificity collapse ($16.42\%$ on IDRiD, $1.05\%$ on RIM-ONE DL) in Table VI.
   - Framed 100% human referral rate honestly as a safe fail-closed alarm bell under distribution shift, not generalizable autonomous screening.
4. **AMD Sample Uncertainty Disclosure:**
   - Reported exact 95% Wilson score interval for AMD sensitivity ($79.31\%$ [61.61%, 90.15%], $n=29$) and designated AMD coverage guarantees as provisional.
5. **PostgreSQL RLS Clean-Up:**
   - Excised theoretical SQL RLS listings from the manuscript; verified application-level multi-tenancy logic via automated tests.

---

### Remediation Pillar D: Explainability & Bibliography Quality (Issues ISS-01, ISS-12, ISS-13, ISS-20)
1. **Deterministic Grad-CAM Explainability:**
   - Replaced `random.choice` in `backend/services/explanation_service.py` with deterministic spatial centroid extraction mapping heatmap activations to retinal quadrants.
   - Locked random seed (`torch.manual_seed(42)`) in `scripts/verify_gradcam_sanity.py`, locking spatial Pearson correlation to $r = -0.0818 \pm 0.1581 \ll 0.80$.
2. **Citation Key Mismatch Resolution:**
   - Corrected line 70 in `research/paper_revised.tex` from `\cite{shen2021eyequality}` to `\cite{fu2019eyequality}` (Fu et al., *IEEE TMI* 2019).
   - Removed 4 duplicate BibTeX entries from `research/references.bib`, leaving 73 clean entries with zero missing key warnings.
3. **Manuscript Condensation (ICCECE 2027):**
   - Condensed 25-page narrative draft into a rigorous 7-page, double-blind conference paper (`research/paper_revised.tex`) strictly conforming to IEEE formatting standards.
