# OphthalmoAI: Recommended Issues to File on GitHub

This file formats identified enhancements and deferred tasks into ready-to-file GitHub issues with titles, markdown descriptions, labels, and priority tags.

---

## Issue 1: Implement PostgreSQL Native Row-Level Security (RLS)

- **Title**: `feat(db): Implement PostgreSQL Row-Level Security (RLS) policies`
- **Labels**: `enhancement`, `database`, `security`
- **Priority**: Medium
- **Body**:
```markdown
### Summary
Currently, multi-tenant clinic data isolation is implemented at the application/ORM layer in `backend/tenancy.py`. To establish defense-in-depth against direct database SQL injection or compromised DB connection pools, we should enforce PostgreSQL-native `CREATE POLICY` Row-Level Security rules.

### Requirements
1. Define PostgreSQL RLS migration in Alembic conditioned on `engine.dialect.name == "postgresql"`.
2. Configure tenant context session variables (`SET LOCAL app.current_tenant_id = :tenant_id`) on each session checkout.
3. Verify backward compatibility with development SQLite databases.
```

---

## Issue 2: Frontend Progressive TypeScript Migration

- **Title**: `refactor(frontend): Migrate React JSX components and edge inference to TypeScript`
- **Labels**: `refactor`, `frontend`, `typescript`
- **Priority**: Low
- **Body**:
```markdown
### Summary
The React 19 frontend is currently written in modern JavaScript/JSX with full ESLint and Vitest test coverage (23 passing tests). Migrating to TypeScript will improve prop typing and prevent UI schema mismatches.

### Scope
1. Introduce `tsconfig.json` with strict mode.
2. Type API response interfaces matching `backend/schemas/` models.
3. Migrate components starting with utility and edge inference modules (`edgeInference.js`, `edgeHeuristic.js`).
```

---

## Issue 3: Curate and Train Dedicated Adnexal / Eyelid Vision Classifier

- **Title**: `feat(ml): Curate dataset and train adnexal/eyelid screening specialist`
- **Labels**: `ml`, `clinical-research`, `dataset`
- **Priority**: Medium
- **Body**:
```markdown
### Summary
An early anatomical router architecture proposed separate specialists for posterior fundus, anterior segment, and external adnexal/eyelid conditions. While fundus screening is thoroughly validated, adnexal models require formal dataset acquisition with patient-level splits before publication.

### Tasks
1. Identify and pre-register public or partner clinical adnexal datasets (e.g. blepharitis, chalazion, ectropion).
2. Establish patient-level train/val/test splits to eliminate leakage.
3. Benchmark against baseline EfficientNet-B4 architectures.
```

---

## Issue 4: Add Optional Test-Time Augmentation (TTA) Flag to `/predict`

- **Title**: `perf(api): Add optional ?use_tta=true query parameter to /predict`
- **Labels**: `performance`, `inference`, `api`
- **Priority**: Low
- **Body**:
```markdown
### Summary
Test-Time Augmentation (TTA) provides slight calibration improvements on borderline images but introduces a ~4x latency overhead.

### Proposal
Expose an opt-in boolean query parameter `?use_tta=false` on `POST /predict`. When enabled, run multi-scale evaluation; when false, use the fast deterministic path (19.7ms median latency).
```

---

## Issue 5: Playwright End-to-End Browser Integration Test Suite

- **Title**: `ci(testing): Add Playwright browser tests for offline edge screening and UI flow`
- **Labels**: `testing`, `ci`, `e2e`
- **Priority**: Medium
- **Body**:
```markdown
### Summary
Add headless browser integration tests verifying:
1. Drag-and-drop retinal fundus upload.
2. WebAssembly ONNX Runtime edge inference execution in offline mode.
3. Interactive Grad-CAM opacity slider and biomarker display.
4. Error banner rendering on non-fundus image uploads.
```
