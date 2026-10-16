# Changelog & Verification Record: Project B — OphthalmoAI

All notable changes made to OphthalmoAI during the October 16, 2026 Deep Audit are documented here.

---

## [2.1.0-audit] - 2026-10-16

### Policy & Compliance Neutralization

- **Academic Publication Claims Harmonization (`research/`):**
  - Surgically sanitized 11 auxiliary research documents that contained premature claims of publication in IEEE or fabricated DOIs:
    - `research/supplementary/gavideo_captions.srt`: replaced IEEE publication claims with neutral preprint terminology.
    - `research/supplementary/video_abstract_production_package.md`: updated targets to "peer-reviewed biomedical informatics manuscript (in preparation)".
    - `research/reproducibility/CODE_OCEAN_REPRODUCIBILITY.md`: removed placeholder DOI `10.1109/JBHI.2026.XXXXXXX`.
    - `research/reproducibility/code_ocean_capsule/metadata/metadata.json`: updated status to "In Preparation" and removed publisher IEEE tag.
    - `research/reproducibility/zenodo.json`: removed fake DOI and updated metadata.
    - `research/reproducibility/code_ocean_capsule/code/reproduce_all.py`: neutral CLI description.
    - `research/reproducibility/code_ocean_capsule/code/run.sh`: updated runner header.
    - `research/scripts/build_expanded_manuscript.py`: stripped placeholder acceptance dates and DOIs.
    - `research/supplementary/README_DATASETS.txt`, `README_SUPPLEMENTARY.txt`, and `graphical_abstract_caption.txt`: updated headers.

### Security Fixes (P0 & P1)

- **Zero-Egress Data Leak Prevention (`frontend/src/apiConfig.js`):**
  - Removed default fallback to `'https://api.trycloudflare.com'`. Default is now `import.meta.env.VITE_FALLBACK_TUNNEL_URL || ''`, ensuring patient scans and metadata never leave local networks without explicit operator configuration.
- **Appointment Account Contamination Hardening (`backend/main.py`):**
  - Updated unauthenticated appointment scheduling and history queries in `/appointments` and `/appointments/history` to bind strictly to a dedicated demo account (`demo@ophthalmoai.org`) with hashed bcrypt credentials, preventing privilege escalation into real patient or administrator accounts.

### Explainability & UI Fixes (P2)

- **Grad-CAM None-Guard (`backend/main.py`):**
  - Added `if GRADCAM_AVAILABLE and MONOLITHIC_MODEL is not None:` in `/predict` to prevent `AttributeError` when model weights are not loaded.
- **1:1 Retinal Heatmap Pixel Alignment (`backend/main.py`, `frontend/src/App.jsx`):**
  - Modified `/predict` to return `preprocessed_image` (the 384x384 circular aperture cropped canvas).
  - Updated `App.jsx` to pass `result.preprocessed_image || previewUrl` to `SplitSenseSlider`, eliminating spatial jumping and coordinate misalignment when comparing raw fundus images with Grad-CAM saliency heatmaps.

### Regression Tests Added

- `tests/backend/test_security_and_auth.py`:
  - `test_appointment_unauthenticated_isolation`: verifies unauthenticated appointment history queries do not leak administrator or other patient appointments.

### Verification

- 283 Backend Pytest tests passed.
- 23 Frontend Vitest tests passed.
- Production Vite frontend build succeeded in 5.99s.

---

## [2.2.0-reliability] - 2026-10-16

### Deserialization Security Hardening (P1)
- **Universal `weights_only=True` Enforcement across Research Pipeline:**
  - Audited all 25 checkpoints in `models/*.pth` (verified to be pure tensor state dicts).
  - Enforced `weights_only=True` across all 21 `torch.load()` invocations in `scripts/*.py` (`evaluate_ensemble.py`, `train_ensemble.py`, `calibrate_models.py`, `evaluate_models.py`, `train_model.py`, `finetune_retfound_comparison.py`, `fine_tune_external_ensemble.py`, `run_conformal_calib.py`, `run_statistical_tests.py`), preventing arbitrary code execution via untrusted pickle files.

### Offline Air-Gap & Zero-Egress Hardening (P1)
- **Air-Gapped Font Stack Implementation (`frontend/src/index.css`):**
  - Removed external Google Fonts `@import url('https://fonts.googleapis.com/...')`.
  - Replaced with high-DPI native OS system font stacks (`system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`), ensuring 100% silent, offline operation with zero external CDN dependencies or network error logs in air-gapped clinical deployments.

### Parity & Benchmarking Reliability (P1)
- **Dynamic ONNX Input Resolution Auto-Adaptation (`backend/onnx_inference.py`, `scripts/benchmark_onnx.py`):**
  - Fixed dimension mismatch error where the benchmark engine hardcoded `(1, 3, 380, 380)` against 224x224 edge models, which previously triggered fallback to synthetic mode.
  - Implemented dynamic input shape extraction from session metadata (`session.get_inputs()[0].shape`), enabling real measured micro-benchmarks on `models/edge_model_fp32.onnx` (`1.97 ms p50`, `504.2 QPS`, `is_synthetic: false`).
  - Added model auto-discovery and optional `--input-shape` parameter to `scripts/benchmark_onnx.py`.

### Reproducibility & Pipeline Independence (P2)
- **Pandas-Independent DataLoader Fallback (`scripts/prepare_dataset.py`):**
  - Refactored `RetinalFundusDataset` to support Python's standard library `csv.DictReader` when pandas is not installed, eliminating import failures in lean runtime environments.
- **End-to-End Live Evaluation CLI (`scripts/reproduce_evaluation.py`, `scripts/evaluate_ensemble.py`):**
  - Added `--live` / `--live-eval` flags to `scripts/reproduce_evaluation.py` to trigger live neural forward passes over [`dataset/processed/test.csv`](file:///d:/OphthalmoAI/dataset/processed/test.csv).
  - Verified live GPU execution of the Tri-Backbone Meta-Ensemble on the held-out test split ($n = 938$), empirically measuring 97.23% Top-1 Accuracy, 0.9990 Macro AUROC, 0.9704 Macro F1, and 0.0884 ECE.

### Regression Test Suite Added
- `tests/backend/test_onnx_parity_and_security.py`:
  - `test_pytorch_to_onnx_fp32_numerical_parity`: verifies max absolute discrepancy $< 10^{-4}$ (measured: $3.81 \times 10^{-6}$) and cosine similarity $> 0.9999$ across random inputs and clinical test samples.
  - `test_weights_only_deserialization_security`: asserts all `.pth` files in `models/` load safely with `weights_only=True`.
  - `test_air_gapped_zero_egress_font_integrity`: asserts zero external Google Fonts or CDN references exist in frontend assets.

### Automated Test Verification
- **Backend Tests:** **286 passed** in 70.16s (100% pass rate).
- **Frontend Tests:** **23 passed** in 2.66s (100% pass rate).
- **Production Vite Build:** Cleanly transformed 2,011 modules in 6.93s.
- **Total Tests:** **309 Automated Tests Passing Cleanly**.
