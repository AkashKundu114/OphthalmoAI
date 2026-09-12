# OphthalmoAI — Technical Debt & Issues Log

This log tracks architectural items, resolved engineering blockers, and ongoing operational maintenance for OphthalmoAI.

---

## 1. Resolved Items

| Issue ID | Severity | Component | Resolution Summary |
|---|---|---|---|
| **R1** | Critical | Inference Pipeline | Migrated to Calibrated Tri-Backbone Soft-Voting Ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M) with Platt temperature scaling. |
| **R2** | Critical | Interpretability | Integrated dedicated EfficientNet-B4 backbone for high-resolution, pixel-aligned Grad-CAM saliency heatmaps. |
| **R3** | High | Frontend / Reporting | Built modern clinical PDF export with side-by-side fundus and Grad-CAM embeddings, ICD-10/SNOMED-CT codes, and clinician attestation. |
| **R4** | High | Frontend / UX | Implemented dual Audience Mode switcher (`[ Public View | Academic / Clinical ]`) with persistent `localStorage` preference. |
| **R5** | Medium | Security | Hardened CORS configuration, magic-byte image validation, slowapi rate-limiting, and JWT authentication. |
| **R6** | Medium | Asynchronous DB | Migrated auth paths to `AsyncSession` with `asyncpg` / `aiosqlite` and Alembic schema migrations. |
| **R7** | Low | Telemetry & Visuals | Created unified benchmark visualizer generating 10 high-resolution charts in `docs/images/` including dual RAM+VRAM and training speedups. |
| **R8** | High | Precision Engineering | Evaluated full BF16 suite across all backbones vs FP16 (+4.16% accuracy gain for FP16); saved weights and calibrated with Platt scaling. |
| **R9** | Critical | Clinical Safety / Guardrails | Built multi-spectral retinal fundus domain validator and hardened Gemini conversational API against prompt injection and jailbreaks. |
| **R10** | High | Visuals & Test Coverage | Upgraded all 10 benchmark figures to publication-grade academic standards (300 DPI, IEEE/Nature style) and expanded automated test suite to 169 tests with 100% pass rate. |

---

## 2. Active Roadmap Items

| Issue ID | Priority | Description | Target Milestone |
|---|---|---|---|
| **A1** | Medium | Worker Concurrency: Add multi-worker Uvicorn configuration or background task queue (Celery/Redis) for high-load production scaling. | v2.3 |
| **A2** | Low | Mobile Camera Integration: Direct WebRTC camera stream capture for real-time mobile fundoscope attachments. | v2.4 |
