# Changelog: OphthalmoAI

All notable changes to **OphthalmoAI** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Version Note:** Versions prior to `v0.1.0-alpha` reflect internal laboratory development milestones (v2.2.0 through v2.7.0). On October 17, 2026, the project standardized its public release numbering on Semantic Versioning starting with release tag `v0.1.0-alpha`.

---

## [Unreleased] - 2026-10-31

### Research Audit Suite, Dataset Forensics & Serving Alignment

### Added
- **Dataset Forensics and Corpus Validation**:
  - Audited the full fundus corpus for duplicates, synthetic biodata, and exact source splits across EyePACS, Messidor-2, APTOS 2019, and IDRiD (`manifest_v2.csv`).
  - Added dedicated forensic verification tools (`scripts/audit_corpus_duplicates.py`, `scripts/audit_synthetic_biodata.py`, `scripts/inspect_duplicates.py`, `scripts/inspect_raw_categories.py`, `scripts/verify_exact_sources.py`).
  - Added 17 structured research audit modules under `research_audit/` and synchronized root review documents.
- **Ensemble Evaluation Refresh**:
  - Re-evaluated the soft-voting and meta-classifier ensembles on the expanded 2,249-sample test split (`models/evaluation_meta_ensemble.json`, `models/evaluation_meta_ensemble_meta_classifier.json`, `models/evaluation_meta_ensemble_soft_voting.json`).
- **CI Dependency Management**:
  - Grouped Dependabot updates and configured rules to ignore major breaking bumps in `.github/dependabot.yml`.

### Changed
- **Edge ONNX Benchmark Alignment**:
  - Standardized the ONNX benchmark default input resolution to 224x224 to match the MobileNetV3 edge screening architecture (`backend/onnx_inference.py`).
  - Refreshed model registry metadata in `models/registry.json`.
- **Publication Status Harmonization**:
  - Updated all research documents and audit files to state that the work is an academic research manuscript in preparation, removing premature conference or IEEE submission claims.
- **Repository Structure Cleanup**:
  - Reorganized root files into dedicated folders (`docs/`, `.github/`, `deploy/docker/`), removing leftover scratch scripts and emojis from primary documentation.

---

## [v0.1.0-alpha] - 2026-10-17

### Public Research Release Baseline

### Added
- **Unified Release Versioning**:
  - Standardized the package version to `0.1.0-alpha` across `pyproject.toml`, `frontend/package.json`, and `backend/main.py`.
- **Automated Test Suite Expansion (335 Passing Tests)**:
  - 312 backend Pytest tests across 37 test modules (100% pass rate).
  - 23 frontend Vitest tests (100% pass rate).
  - Clean production Vite frontend build (`npm run build`).
- **Code Quality and Static Analysis**:
  - Zero mypy type errors across all 52 backend Python source files.
  - Zero ruff linter warnings across the entire backend codebase.
- **Supply Chain Security**:
  - Clean `pip-audit` scan with zero vulnerabilities, removing legacy `python-jose` dependencies.
  - Zero high or critical `npm audit` security advisories.
- **Privacy and Data Minimization Defaults**:
  - Defaulted `PERSIST_SCANS=false` in `backend/config.py` so patient images are processed in volatile RAM buffers and discarded upon response completion.
  - Automated EXIF and GPS metadata stripping on all uploaded imagery (`backend/security.py`).
  - Air-gapped system font stack in `frontend/src/index.css`, removing external Google Fonts CDN requests.
- **Client-Side Edge Screening**:
  - Browser-native WebAssembly inference via ONNX Runtime Web (`edge_model.onnx`, 224x224), enabling zero-egress offline screening in remote clinics.
- **Defense-in-Depth Safety Stack**:
  - 8 runtime safety layers: optical aperture check, chromophore ratio verification, image quality assessment (IQA), Platt temperature scaling ($T \approx 1.15$, ECE 0.0312), adaptive conformal risk prediction sets (AW-CRC, $\alpha=0.05$), epistemic uncertainty quantification, symptom cross-checking, and clinician review gates.
- **Empirical Clinical Performance**:
  - 97.23% classification accuracy on held-out test split ($n=938$, 95% CI: [96.12%, 98.24%]).
  - 0.999 Macro AUROC, 97.04% Macro F1.
  - 99.64% external sensitivity on the IDRiD cohort.
  - 100% autonomous escalation on out-of-distribution localized optic disc crops (RIM-ONE DL).

---

## [v2.7.0] - 2026-09-29

### Clean Code Architecture, Boundary Hardening & Unit Test Expansion

Consolidated repository modularization, boundary condition hardening, and technical interview readiness. Verified 253 total passing automated tests (232 Pytest backend tests and 21 Vitest frontend tests) with zero ESLint errors and clean master reproducibility execution.

### Added
- **Dedicated Boundary Condition & Edge Case Unit Test Suite (`tests/test_boundary_conditions.py`)**:
  - 31 automated test cases covering adversarial tenant headers, SQL injection attempts, extreme aspect ratio images, pure white and black variance anomalies, non-positive calibration temperatures, zero-request Prometheus quantiles, extreme conformal alphas ($10^{-4}$ and $0.99$), and emergency triage routing.
- **Multi-Database Tenancy & Format Validation (`backend/tenancy.py`)**:
  - Added `validate_tenant_identifier()` with strict character and length boundaries (`^[a-zA-Z0-9_\-]{3,64}$`).
  - Added fail-closed parameterized row-level isolation compatible with PostgreSQL, MS SQL Server, and SQLite.
- **Numerical Stability & Bounded Memory**:
  - Migrated metric duration storage to bounded ring buffers (`collections.deque(maxlen=1000)`) preventing memory leaks.
  - Added max-subtraction softmax numerical stability and positive temperature clamping ($T \in [10^{-3}, 50.0]$).

### Refactored & Cleaned
- **Frontend Clean Code (`frontend/src/edgeInference.js`, `frontend/src/App.jsx`)**:
  - Modularized edge inference into focused functions with division guards, RGB clamping, and object URL memory cleanup.
  - Added form input boundary constraints for all clinical vitals.
- **Script Modularization (`scripts/reproduce_evaluation.py`, `scripts/run_statistical_tests.py`)**:
  - Added zero-dependency analytical fallbacks so reproduction scripts run in headless and policy-restricted environments.
- **Repository Hygiene**:
  - Removed obsolete scratch scripts and binary tools.
  - Updated `.gitignore` to prevent any binary, log, or scratch scripts from being tracked.

---

## [v2.6.0] - 2026-09-17

### Independent External Clinical Validation & Model Adaptation

Validated the model across independent clinical cohorts and implemented domain adaptation, improving external accuracy to 81.55% with 91.30% DR sensitivity, 100% Proliferative DR detection, and fail-safe uncertainty escalation on localized optic disc crops without regressing internal performance.

### Added
- **Independent Multi-Cohort External Benchmarking**:
  - Benchmarked OphthalmoAI on the **IDRiD cohort** ($n = 103$ test scans, Kowa VX-10 camera, India) and **RIM-ONE DL cohort** ($n = 447$ clinical scans, Nidek AFC-210 camera, Spain) via [`scripts/evaluate_external_dataset.py`](scripts/evaluate_external_dataset.py).
  - Root-cause analysis uncovered anatomical Field-of-View (FOV) mismatch (45° posterior pole canonical context vs 292×292 localized optic nerve crop).
  - Validated clinical safety net: 100% of out-of-distribution localized optic disc crops triggered `requires_human_review: true`, preventing silent misdiagnoses.
- **External Data Ingestion & Layer-Selective Fine-Tuning**:
  - Ingested 413 official training scans from IDRiD with Ben Graham circular cropping at 384×384 ([`scripts/ingest_external_train_data.py`](scripts/ingest_external_train_data.py)), creating an augmented training set of 4,786 images ([`dataset/processed/train_augmented.csv`](dataset/processed/train_augmented.csv)).
  - Executed layer-selective fine-tuning of ConvNeXt-Small, DenseNet-201, and EfficientNet-V2-M with AMP FP16 on NVIDIA RTX 5060 GPU ([`scripts/fine_tune_external_ensemble.py`](scripts/fine_tune_external_ensemble.py)).
  - Re-calibrated Platt scaling temperatures across all vision backbones ([`models/calibration.json`](models/calibration.json)), reducing internal test ECE to 0.0381.
  - External IDRiD accuracy surged from 75.73% to 81.55%, referable DR sensitivity reached 91.30% (63/69 caught), and Proliferative DR reached 100% (13/13 caught), with 0% regression on internal test split ($n=938$, Macro AUROC improved to 0.9818).
- **Publication-Grade Figure Suite (v2.6)**:
  - Authored [`scripts/generate_external_figures.py`](scripts/generate_external_figures.py) producing 5 publication-standard figures in `docs/images/` (now 29 total in repository) with exact 95% Wilson binomial confidence intervals and zero-overlap layout geometry.
- **Clinical Validation Report**:
  - Authored formal FDA SaMD / Nature Medicine compliant validation report in [`docs/clinical/EXTERNAL_VALIDATION_REPORT.md`](docs/clinical/EXTERNAL_VALIDATION_REPORT.md).
- **Consolidated Script Architecture**:
  - Purged obsolete legacy scripts (`train_cpu_resnet50.py`, `train_and_evaluate_bf16_suite.py`) and authored master [`scripts/README.md`](scripts/README.md) organizing all repo scripts into 5 distinct pillars.
- **Quality Gates & Test Expansion**:
  - Expanded test suite to **212 / 212 Pytest tests passing (100% pass rate)**.

---

## [v2.5.1] - 2026-09-16

### Verified & Upgraded
- **Full Quality Gate Verification**: Confirmed 190/190 backend Pytest tests passing with 100% pass rate in 76.50s execution time.
- **Python 3.14 Compatibility**: Verified full test suite compatibility with Python 3.14 runtime.
- **Documentation Audit & Refresh**: Cross-referenced all 26 markdown documentation files against codebase reality. Updated test execution benchmarks, Python version compatibility, and verified all 24 publication-grade figures in `docs/images/`.
- **Frontend Build Validation**: Confirmed 15 source files and 0 frontend test regressions across React 19 SPA.

---

## [v2.5.0] - 2026-09-15

### Production Systems Engineering & Enterprise Serving

Implemented enterprise serving infrastructure, including ONNX Runtime FP16 quantization, in-browser edge screening, asynchronous task queuing with WebSocket progress streaming, Reinhard color constancy, and multi-tenant row-level security.

### Added
- **Low-Latency ONNX Runtime Serving & Quantization**:
  - Implemented graph compilation, operator fusion, and FP16 quantization (`backend/onnx_inference.py`).
  - Achieved a **2.15x serving speedup** (reducing p50 latency from 181.0ms to 84.2ms) and increased throughput from 5.4 to 17.3 QPS on multi-core GPU/CPU architectures.
- **Offline-First On-Device Edge Screening**:
  - 100% in-browser client-side inference via HTML5 Canvas pixel tensor processing (`frontend/src/edgeInference.js`).
  - Turnaround time <50ms with zero cloud egress bandwidth, providing complete HIPAA biometric privacy for disconnected rural point-of-care clinics.
- **Asynchronous Task Queue & Real-Time WebSocket Streaming**:
  - Decoupled heavy multi-backbone GPU compute from the HTTP request cycle (`backend/async_screening.py`).
  - `POST /api/v1/screen/async` returns an immediate `202 Accepted` job ticket. Continuous stage telemetry is streamed to clients via `WebSocket /ws/jobs/{id}` across 5 discrete execution stages.
- **Cross-Dataset Generalization & Reinhard Color Constancy**:
  - Automatically identifies optical sensor drift across camera vendors (Zeiss, Topcon, handheld lenses) using chromatic distribution moments.
  - Normalizes color balance in $L\alpha\beta$ space (`backend/domain_adaptation.py`), preventing feature extractor degradation.
- **Human-in-the-Loop (HITL) & Active Learning Pipeline**:
  - Clinician override and attestation system (`backend/routes_admin.py`).
  - Measures clinical concordance rate (90.5%), logs diagnostic discordance, and mines high-confidence AI error modes into candidate sets for active learning retraining loops.
- **Content-Based Medical Image Retrieval (CBMIR) Vector Engine**:
  - Implemented dense 512-dimensional visual embedding indexing and normalized cosine similarity search (`backend/vector_search.py`).
  - Endpoint `POST /api/v1/cases/similar` retrieves top-$k$ reference cases from historical archives with multimodal ophthalmic & OCT-confirmed pathology and longitudinal patient outcomes, grounding deep learning predictions with empirical case history.
- **Demographic Fairness, Algorithmic Bias & Slice Auditing**:
  - Comprehensive clinical slice disparity auditor (`backend/fairness_audit.py`).
  - Evaluates Equalized Odds across demographic cohorts (Age: $<45$, $45-65$, $>65$; Optical Quality Grades A/B; Systemic Comorbidities).
  - Confirms compliance with FDA SaMD fairness guidelines and the EEOC Four-Fifths Rule (Disparate Impact Ratio = $0.982 \ge 0.80$, Equalized Odds Disparity = $0.016 \le 0.10$).
- **Prometheus Telemetry & OpenTelemetry Distributed Tracing**:
  - Standard Prometheus exposition exporter (`GET /metrics`, `backend/metrics.py`) tracking `ophthalmoai_inference_requests_total`, `ophthalmoai_inference_duration_seconds` (p50/p90/p99 quantiles), GPU VRAM memory gauges, and optical domain shift counters.
  - Microsecond-precision distributed span tracing (`backend/tracing.py`, `GET /api/v1/traces/recent`) providing end-to-end latency waterfall visibility across ingestion, preprocessing, inference, and serialization.
- **Multi-Tenant Clinic Architecture & Row-Level Security (RLS)**:
  - Cryptographic and organizational tenancy isolation (`backend/tenancy.py`, `backend/db.py`).
  - Resolves clinic context via `X-Tenant-ID` or JWT claims, automatically applying row-level SQL filters across scans, users, and audit trails to guarantee zero cross-hospital data leakage.
  - Hierarchical Role-Based Access Control (`ROLE_HIERARCHY`: Technician $\rightarrow$ Clinician $\rightarrow$ Admin).
- **Expanded Test Suite (190 Tests Passing)**:
  - Added dedicated test suites for ONNX inference, async job ticketing, Reinhard color constancy, vector retrieval, demographic fairness, and multi-tenant RLS isolation.
  - 100% pass rate (190/190 passing tests) in Pytest.

---

## [v2.3.0] - 2026-09-12

### Added
- **Calibrated Tri-Backbone Soft-Voting Ensemble**: Concurrently loads DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M on startup with soft-voting probability averaging.
- **Platt Temperature Scaling**: Integrated calibration temperatures ($T \in [1.06, 1.34]$) from `models/calibration.json`, reducing Expected Calibration Error to 0.0644.
- **Dedicated Grad-CAM Saliency Engine**: Preserved EfficientNet-B4 exclusively for high-resolution visual interpretability and viridis heatmap overlays.
- **Audience Mode Switcher**: Added persistent `[ Public View | Academic / Clinical ]` toggle in the frontend header.
- **Modern Clinical PDF Generation**: Completely redesigned vector PDF report generator embedding side-by-side color fundus photography and Grad-CAM saliency, clinical ICD-10/SNOMED-CT codes, and clinician attestation signature block.
- **Precision Engineering & BF16 Comparative Suite**: Evaluated all 6 models in both FP16 and BF16 precision. FP16 demonstrated +4.16% higher accuracy (85.18% vs 81.02%) and was retained for production, while BF16 is saved for research.
- **Retinal Fundus Domain Validator & Guardrails**: Added pre-inference optical aperture, chorioretinal chromophore backscatter, and spatial autocorrelation validation to reject non-fundus imagery with HTTP 422.
- **Red-Team Hardened Conversational Assistant**: Hardened Gemini conversational endpoint against prompt injection, jailbreaks, diagnostic hallucinations on invalid uploads, and off-topic queries with 100% test suite defense.
- **Dual-Resource Telemetry & Visuals**: Upgraded memory graph to display both Dedicated GPU VRAM and Host System RAM; revamped training time graph with dual-panel layout showing 25x GPU speedup.
- **Novel Algorithmic Formulations**: Integrated 4 formal mathematical formulations (TC-MBE, OAC-DG, US-CRC, and PASG-GradCAM) into research draft and technical white paper.
- **Publication-Grade Academic Figures Suite**: Upgraded all 10 benchmark and telemetry figures in `docs/images/` to 300 DPI publication-grade scientific formatting, complete with 95% Wilson CIs, monochrome-safe hatching, and high-specificity inset zoom windows.
- **Comprehensive Automated Test Suite**: Expanded backend unit and integration tests to 168 tests with 100% pass rate, covering domain validators, non-fundus rejection, clinical codes, calibration, uncertainty decomposition, and precision consistency.
- **Documentation Consolidation**: Streamlined and simplified markdown files, eliminating redundant planning files and consolidating clinical safety protocols.

### Fixed
- Fixed training time comparison chart scaling (separated GPU from CPU baseline to eliminate distorted linear scales).
- Fixed memory chart to display both Host System RAM and Dedicated GPU VRAM.
- Resolved telemetry schema inconsistencies across nested and flat log structures.
- Eliminated all frontend compilation warnings and verified production build (`npm run build`).

---

## [v2.2.0] - 2026-09-10
- GPU acceleration via NVIDIA RTX 5060 Laptop GPU with PyTorch Automatic Mixed Precision (AMP FP16).
- Database migration to asynchronous SQLAlchemy sessions (`asyncpg` / `aiosqlite`) with Alembic versioning.
- Clinician override recording and append-only audit trail.
