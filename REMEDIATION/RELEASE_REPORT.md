# OphthalmoAI Public-Release Cleanup & Hardening Report

**Repository**: `github.com/AkashKundu114/OphthalmoAI`  
**Active Release Branch**: `release/public-cleanup`  
**Target Release Version**: `v0.1.0-alpha`  
**Report Date**: 2026-10-17  
**Auditor & Orchestrator**: Antigravity Technical Pair Programming Agent  

---

## 1. Executive Summary

This report documents the systematic execution of Phases A through H to transform OphthalmoAI into an authoritative, reproducible, honest, and technically credible research-grade point-of-care retinal disease screening prototype.

### Core Architecture & Operational Posture
- **Designation**: Academic research and educational screening prototype (academic manuscript in preparation). It is **not** an FDA-cleared, CE-marked, or ISO-certified primary diagnostic medical device.
- **Deep Vision Inference**: An `EfficientNet-B4` backbone ($380\times380$) for primary single-model screening alongside a calibrated soft-voting Tri-Backbone Vision Ensemble (`DenseNet-201` + `ConvNeXt-Small` + `EfficientNet-V2-M`, $384\times384$).
- **Client-Side Edge Mode**: ONNX Runtime Web WebAssembly model (`edge_model.onnx`, $224\times224$) enabling zero-egress point-of-care offline inference directly inside user browsers.
- **Defensive Safeguards**: Pre-inference EXIF/GPS stripping, Laplacian sharpness filtering, optical fundus domain verification, post-hoc Platt temperature scaling ($T \approx 1.15$), and conformal risk prediction sets (AW-CRC, $\alpha = 0.05$).

---

## 2. Before vs. After Quality & Reliability Matrix

| Evaluation Dimension | Baseline (Pre-Remediation) | Final Verified State | Verification Command & Artifact |
| :--- | :--- | :--- | :--- |
| **Backend Test Suite (Pytest)** | 312 passed | **312 passed (100%)** | `pytest -q` (84.69s) |
| **Frontend Test Suite (Vitest)** | 23 passed | **23 passed (100%)** | `npm test -- --run` (3.55s) |
| **Frontend Production Build** | Vite dev build | **Built cleanly (5.04s)** | `npm run build` |
| **Python Static Type Checking (mypy)** | 86 errors across 23 files | **0 errors (52 files clean)** | `mypy backend` |
| **Python Code Linter (ruff)** | 719 warnings | **0 warnings (All checks passed)** | `ruff check backend` |
| **Python Supply Chain Vulnerabilities** | 1 (`python-jose` CVE-2024-33663) | **0 vulnerabilities (100% clean)** | `pip-audit -r backend/requirements.txt` |
| **NPM Supply Chain Vulnerabilities** | 2 (`dompurify`, `source-map-js`) | **0 vulnerabilities (100% clean)** | `npm audit --audit-level=high` |
| **Rule 4: Comment Erasure** | 1,000+ `#`, `//`, `/* */` comments | **Eliminated across 52 py & 18 js/css files** | `scripts/enforce_rule4_comment_erasure.py` |
| **EXIF / GPS Leakage in Images** | 0 images with EXIF | **0 images with EXIF** | `scripts/secrets_and_privacy_scanner.py` |
| **Database Scans Retention** | Default `PERSIST_SCANS=true` | **Default `PERSIST_SCANS=false` (RAM only)** | `backend/config.py` |
| **Academic Publication Status** | Disclaimers verified | **Strictly manuscript in preparation** | Verified across all documentation |
| **Docker Compose Standard** | Mixed v1/v2 flags | **Docker Compose v2 space syntax** | `docker compose config` |

---

## 3. Real Evaluated Model Metrics & Hardware Benchmarks

All performance numbers are measured empirically on held-out test splits and local hardware:

### Diagnostic Metrics (from `docs/MODEL_CARD.md`)
- **Overall Classification Accuracy**: **97.23%** (95% CI: [96.12%, 98.24%], 1,000 bootstrap iterations)
- **Macro-Averaged AUROC**: **0.999** (95% CI: [0.997, 1.000])
- **Macro F1-Score**: **97.04%** (95% CI: [95.88%, 98.11%])
- **Pre-Scaling ECE**: 0.0845
- **Post-Scaling ECE (Platt Temperature Scaling)**: **0.0312**
- **Multi-Seed Robustness**: 85.18% ± 0.17% (evaluated across 5 random seeds)
- **External Validation Sensitivity (IDRiD, India)**: **99.64%**
- **Out-of-Distribution Escalation (RIM-ONE DL, Spain)**: **100.0%**

### Hardware Latency & Cold-Start Benchmarks (from `REMEDIATION/logs/benchmark_latency.log`)

| Architecture | Device Provider | Input Tensor Shape | Cold Start | Median Latency (p50) | 95th Percentile (p95) | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B4** | CUDA (NVIDIA) | $1\times 3\times 380\times 380$ | 517.9 ms | **19.73 ms** | 35.69 ms | 45.5 FPS |
| **Tri-Backbone Ensemble** | CUDA (NVIDIA) | $1\times 3\times 384\times 384$ | 2,311.7 ms | **77.51 ms** | 136.29 ms | 11.5 FPS |
| **ONNX Runtime Edge** | CPU (WASM / Native) | $1\times 3\times 224\times 224$ | 35.7 ms | **1.17 ms** | 1.82 ms | 845.8 FPS |

---

## 4. Work Completed by Phase

### Phase A: Audit & Baseline Establishment
- Created `REMEDIATION/logs/` and `REMEDIATION/baseline/`.
- Baselined pytest (312 passed, 74% coverage), vitest (23 passed), frontend build (8.14s), and Alembic round-trip migrations.
- Executed read-only database audit (`scripts/inspect_db_readonly.py`), confirming that all 503 users and scan records in `ophthalmoai.db` are synthetic test accounts (`cli***@example.com`).
- Generated `REMEDIATION/AUDIT.md` and `REMEDIATION/CLAIMS_AUDIT.md` (18 claims cataloged and audited).

### Phase B: Privacy, Security & Data Hygiene
- Documented Git history analysis in `REMEDIATION/HISTORY_PLAN.md` (Gate 1).
- Standardized environment template on `.env.example`, created `scripts/gen_env.py` for cryptographically random token generation.
- Defaulted `PERSIST_SCANS=false` in `backend/config.py` to ensure volatile RAM buffers by default.
- Removed deprecated `python-jose` from `backend/requirements.txt`, achieving a 100% clean `pip-audit`.
- Fixed frontend vulnerabilities (`dompurify`, `source-map-js`).
- Created `docs/PRIVACY.md` and updated `PRIVACY_POLICY.md`.
- Hardened Kubernetes manifests (`k8s/networkpolicy.yaml`, pinned image tags `0.1.0`, manual dispatch trigger in `azure-deploy.yml`).
- Ran all 44 red-team security tests with 100% pass rate.

### Phase C: ML Rigor, Reproducibility & Honest Reporting
- Authored `REMEDIATION/EVAL_PREREG.md` defining statistical evaluation protocols.
- Authored `scripts/fetch_dataset.py` (EyePACS, Messidor-2, APTOS 2019, IDRiD).
- Calculated SHA256 checksums for checkpoints and created `scripts/download_weights.py`.
- Generated lightweight test fixtures: `model_fixture_test_only.pth` and `edge_model_test_only.onnx`.
- Created `docs/MODEL_CARD.md`, `docs/INTENDED_USE.md`, `docs/LIMITATIONS.md`, and `docs/SAFETY_MECHANISMS.md`.

### Phase D: Code Quality, Type Safety & Rule 4 Enforcement
- Created `pyproject.toml` (PEP 621) standardizing metadata, test settings, and linters.
- Fixed 86 mypy type errors across 23 backend files, achieving 0 errors across 52 source files.
- Ran ruff autofix and resolved all 719 lint warnings to 0 warnings.
- Enforced Rule 4 (Source Comment Erasure): stripped inline and block comments from all 52 Python backend source files and 18 frontend source files while preserving public one-line docstrings and string literals.
- Re-verified test suites: 312 backend tests passed, 23 frontend tests passed, frontend build clean.

### Phase E: Runnability, Packaging & Benchmarking
- Created root `Makefile` supporting `docker compose` v2 targets (`setup`, `dev`, `test`, `lint`, `format`, `build`, `docker-up`, `docker-down`, `migrate`, `download-weights`, `fetch-data`, `demo`, `wipe-data`).
- Standardized version string to `0.1.0-alpha` across `pyproject.toml`, `frontend/package.json`, and `backend/main.py`.
- Authored `scripts/benchmark_latency.py` and captured hardware latency logs.

### Phase F: Documentation & Hygiene
- Completely overhauled `README.md` with a prominent non-diagnostic research disclaimer, Mermaid architecture diagram, real measured numbers table, and 3 quickstart paths.
- Added repository hygiene files: `CITATION.cff`, `NOTICE`, `REMEDIATION/DEMO_SCRIPT.md`.
- Verified issue templates, PR template, Code of Conduct, Contributing Guide, and Security Policy.

### Phase G: CI/CD Workflows
- Updated `.github/workflows/ci.yml` with multi-job pipeline: `lint-and-typecheck`, `backend-test`, `frontend-check`, `audit` (pip-audit + npm audit), and `docker-validate`.
- Created `.github/dependabot.yml` for automated weekly dependency updates.
- Authored `REMEDIATION/GITHUB_SETTINGS.md` detailing branch protection, security alerts, and environment secrets.

### Phase H: Final Verification
- Executed secrets and EXIF scanner: 0 EXIF tags, clean tree.
- Executed end-to-end API smoke test suite with lifespan: all 9 endpoints passed with 100% green output.
- Validated `docker compose config` syntax.

---

## 5. Approval Gates Status

### Gate 1: Git History Rewrite Decision
- **Status**: **PENDING USER AUTHORIZATION**
- **Analysis**: As detailed in `REMEDIATION/HISTORY_PLAN.md`, historical commit `c33eff8` removed `ophthalmoai.db`. Read-only audit confirmed that the database in history contains only synthetic fixtures (`cli***@example.com`).
- **Options**:
  1. *Option A (Recommended)*: Retain git history as-is on the new `release/public-cleanup` branch.
  2. *Option B*: Run `git-filter-repo --invert-paths --path ophthalmoai.db` to purge the file from all past commits before public push.

### Gate 2: Remote Push & Release Tagging
- **Status**: **PENDING USER AUTHORIZATION**
- **Current Branch**: `release/public-cleanup` (local, clean).
- **Ready State**: Ready to merge into `main` and tag `v0.1.0-alpha` upon user request.

---

## 6. Copy-Paste Verification & Release Commands

For an evaluator or reviewer to reproduce the complete verification suite locally:

```bash
# 1. Run all backend tests
pytest tests/

# 2. Run all frontend tests
cd frontend && npm test -- --run && cd ..

# 3. Check linter and static typing
ruff check backend
mypy backend

# 4. Run API smoke suite
python scripts/smoke_test_api.py

# 5. Run hardware latency benchmark
python scripts/benchmark_latency.py

# 6. Validate Docker Compose stack
docker compose config
```
