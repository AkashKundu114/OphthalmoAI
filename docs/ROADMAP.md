# OphthalmoAI Product Roadmap

This roadmap outlines completed milestones, the current public research release (`v0.1.0-alpha`), and future architectural directions for **OphthalmoAI**, created and maintained by **Akash Kundu**.

---

## Completed Milestones

### Historical Internal Iterations (v2.0 - v2.7)
- [x] **v2.0 (Core Vision Ensemble)**: Multi-backbone training (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M, EfficientNet-B4), soft-voting ensemble with Platt calibration, Grad-CAM saliency, and optical domain guardrails.
- [x] **v2.3 (Clinical Reporting & Telemetry)**: Dual audience view toggle, vector clinical PDF generator with clinician attestation, FP16 vs BF16 precision evaluation, and publication figure generation.
- [x] **v2.5 (Enterprise Serving Infrastructure)**: ONNX Runtime FP16 serving, in-browser HTML5 Canvas edge execution, async WebSocket screening queues, Reinhard color constancy, and CBMIR vector search.
- [x] **v2.6 (External Multi-Cohort Validation)**: External clinical evaluation on IDRiD ($n = 103$) and RIM-ONE DL ($n = 447$), layer-selective fine-tuning with 91.30% DR sensitivity, and fail-safe out-of-distribution escalation.
- [x] **v2.7 (Boundary Hardening & Code Architecture)**: Dedicated 31-case boundary condition suite, multi-database tenancy identifiers, and bounded memory buffers.

### Current Public Release: v0.1.0-alpha (October 2026)
- [x] **Semantic Versioning Consolidation**: Unified repository release tagged `v0.1.0-alpha`.
- [x] **Comprehensive Test Verification**: 335 total passing automated tests (312 backend Pytest tests + 23 frontend Vitest tests, 100% pass rate).
- [x] **Code Quality & Typing**: 0 mypy static type errors across 52 Python files; 0 ruff linter warnings.
- [x] **Supply Chain Hardening**: 0 vulnerabilities via `pip-audit` and `npm audit`.
- [x] **Zero-Retention Medical Privacy**: Default `PERSIST_SCANS=false` RAM processing, EXIF/GPS scrubbing, and air-gapped system fonts.
- [x] **In-Browser Edge Screening**: Quantized MobileNetV3 ONNX Runtime Web model (224x224) running fully offline.
- [x] **Forensic Research Audit Suite**: 17 audit modules across dataset provenance, statistical validity, conformal coverage, and review ledger reconciliation.

---

## Upcoming Milestones

### v0.2.0: Enterprise Systems & Interface Hardening
- [ ] **Database-Native PostgreSQL RLS**: Transition application-level tenancy filters to database-level `CREATE POLICY` rules for multi-tenant deployments.
- [ ] **Frontend TypeScript Migration**: Migrate React 19 JSX modules to strict TypeScript for enhanced compile-time interface safety.
- [ ] **Test-Time Augmentation (TTA) Flag**: Add an optional `?use_tta=true` query parameter for multi-crop inference during offline research evaluations.
- [ ] **Ordinal Conformal Risk Control**: Severity-aware non-conformity guarantees to eliminate skip-grade staging errors in diabetic retinopathy.

### v0.3.0: Mobile Point-of-Care & Foundation Models
- [ ] **RETFound / FLAIR Knowledge Distillation**: Distill representations from large ophthalmic foundation models into the lightweight edge ensemble.
- [ ] **Direct WebRTC Mobile Video Capture**: Smartphone-mounted direct ophthalmoscope video streaming with automated best-frame selection.
- [ ] **Adnexal & Eyelid Specialist Models**: Multi-task screening models for anterior segment and eyelid lesions based on validated clinical datasets.
- [ ] **Longitudinal Progression Tracking**: Pixel-aligned fundus registration and change detection heatmaps across sequential patient visits.

### v1.0.0: Hospital Federation & Standards
- [ ] **DICOM / PACS Ingestion Adapter**: Native ingestion from hospital picture archiving and communication systems (PACS).
- [ ] **Federated Learning Network**: Multi-institutional privacy-preserving collaborative model updates with differential privacy.
