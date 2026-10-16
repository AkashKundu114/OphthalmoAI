# OphthalmoAI: Deferred Engineering & Research Worklist

This document catalogs non-blocking engineering enhancements and research capabilities that were evaluated during the public-release cleanup and deliberately deferred to subsequent milestones (`v0.2.0+`), along with technical justifications and prerequisite conditions.

---

## 1. Native PostgreSQL Row-Level Security (RLS)

- **Current Implementation**: Multi-tenant clinic data isolation is implemented strictly at the application/ORM layer in `backend/tenancy.py` via parameterized SQLAlchemy query filters and JWT tenant claim validation.
- **Proposed Enhancement**: Enforce database-native `CREATE POLICY` Row-Level Security (RLS) rules directly inside PostgreSQL.
- **Rationale for Deferral**: The application supports zero-config local SQLite deployments for researchers and single-clinic evaluations. PostgreSQL-specific RLS policies would break SQLite compatibility without a dedicated dialect-switching migration layer.
- **Target Milestone**: `v0.2.0` (when enterprise multi-tenant PostgreSQL clustering is enabled).

---

## 2. Frontend TypeScript Migration

- **Current Implementation**: React 19 codebase authored in JavaScript/JSX with modern modular architecture, clean ESLint configurations, and 23 passing Vitest unit tests.
- **Proposed Enhancement**: Full TypeScript rewrite of all 18 UI components and edge inference modules.
- **Rationale for Deferral**: The JSX frontend build (`vite build`) and test suites pass 100% cleanly. An immediate TypeScript refactor would introduce high regression risk to UI state management and WebAssembly ONNX loader bindings without adding functional clinical capabilities.
- **Target Milestone**: `v0.2.0`.

---

## 3. Dedicated Adnexal / Eyelid Specialist Models

- **Current Implementation**: The active production model is an ensemble of `DenseNet-201`, `ConvNeXt-Small`, and `EfficientNet-V2-M` alongside a standalone `EfficientNet-B4` backbone for 5-class color fundus screening (`Normal`, `Diabetic Retinopathy`, `Glaucoma`, `Cataract`, `AMD`). An anatomical router was drafted in conceptual design scripts.
- **Proposed Enhancement**: Train separate deep learning vision specialists for adnexal/eyelid conditions and anterior slit-lamp photographs.
- **Rationale for Deferral**: No verified, high-quality public training datasets with patient-level splits exist in the repository for adnexal conditions. Inventing or reporting unverified models without verified checkpoints violates Rule 5 (no invented facts).
- **Target Milestone**: `v0.3.0` (subject to clinical dataset partner acquisition).

---

## 4. Test-Time Augmentation (TTA) in Live Inference Endpoints

- **Current Implementation**: Standard deterministic preprocessing ($380\times380$ and $384\times384$ scaling with ImageNet normalization).
- **Proposed Enhancement**: 5-crop / multi-scale Test-Time Augmentation during live `/predict` requests.
- **Rationale for Deferral**: TTA increases inference latency by $4\times$ to $8\times$, degrading median response time from 19.7ms to over 150ms on GPU, and up to 2 seconds on CPU. TTA is retained in offline research evaluation scripts (`scripts/evaluate_models.py`) but excluded from latency-sensitive point-of-care screening.
- **Target Milestone**: `v0.2.0` (behind an optional query parameter `?use_tta=true`).

---

## 5. Live Multi-Center Federated Learning Aggregation

- **Current Implementation**: Federated learning simulation logic in `backend/federated.py`.
- **Proposed Enhancement**: Secure aggregation with differential privacy over live gRPC/TLS sockets across hospital sites.
- **Rationale for Deferral**: Production federated learning requires cross-institutional governance, IRB protocol approval, and authenticated multi-site infrastructure.
- **Target Milestone**: Future multi-center academic trial.
