# OphthalmoAI — Technical Debt & Issues Log

This log tracks architectural items, resolved engineering blockers, and ongoing operational maintenance for OphthalmoAI, maintained by **Akash Kundu**.

---

## 1. Resolved Items

| Issue ID | Severity | Component | Resolution Summary | Status |
|---|---|---|---|:---:|
| **R1** | Critical | Inference Pipeline | Migrated to Calibrated Tri-Backbone Soft-Voting Ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M) with Platt temperature scaling ($T \in [1.06, 1.34]$). | **Resolved** |
| **R2** | Critical | Interpretability | Integrated dedicated EfficientNet-B4 backbone for high-resolution, pixel-aligned Grad-CAM saliency heatmaps. | **Resolved** |
| **R3** | High | Frontend / Reporting | Built modern clinical PDF export with side-by-side fundus and Grad-CAM embeddings, ICD-10/SNOMED-CT codes, and clinician attestation. | **Resolved** |
| **R4** | High | Frontend / UX | Implemented dual Audience Mode switcher (`[ Public View | Academic / Clinical ]`) with persistent `localStorage` preference. | **Resolved** |
| **R5** | Medium | Security | Hardened CORS configuration, magic-byte image validation, slowapi rate-limiting, and JWT authentication. | **Resolved** |
| **R6** | Medium | Asynchronous DB | Migrated auth paths to `AsyncSession` with `asyncpg` / `aiosqlite` and Alembic schema migrations. | **Resolved** |
| **R7** | Low | Telemetry & Visuals | Created unified benchmark visualizer generating 24 high-resolution charts in `docs/images/` including dual RAM+VRAM and training speedups. | **Resolved** |
| **R8** | High | Precision Engineering | Evaluated full BF16 suite across all backbones vs FP16 (+4.16% accuracy gain for FP16); saved weights and calibrated with Platt scaling. | **Resolved** |
| **R9** | Critical | Clinical Safety / Guardrails | Built multi-spectral retinal fundus domain validator and hardened Gemini conversational API against prompt injection and jailbreaks. | **Resolved** |
| **R10** | High | Visuals & Test Coverage | Upgraded all benchmark figures to publication-grade academic standards (300 DPI, IEEE/Nature style) and expanded automated test suite to **190 tests with 100% pass rate**. | **Resolved** |
| **R11** | Critical | Latency & Optimization | Implemented ONNX Runtime serving with FP16 quantization, cutting p50 latency from 181.0ms to 84.2ms (2.15x speedup) and scaling to 17.3 QPS. | **Resolved** |
| **R12** | High | Concurrency & Async | Implemented asynchronous task queue (`backend/async_screening.py`) with `202 Accepted` tickets and real-time WebSocket progress updates. | **Resolved** |
| **R13** | Critical | Telemedicine Privacy | Built 100% client-side in-browser edge screening (`frontend/src/edgeInference.js`) with <50ms turnaround and zero cloud egress. | **Resolved** |
| **R14** | High | Domain Adaptation | Implemented Reinhard color constancy in $L\alpha\beta$ space (`backend/domain_adaptation.py`) to eliminate camera sensor color drift. | **Resolved** |
| **R15** | High | Observability | Deployed Prometheus telemetry metrics exporter and OpenTelemetry distributed span tracing (`backend/metrics.py`, `backend/tracing.py`). | **Resolved** |
| **R16** | High | Tenancy Isolation | Implemented multi-tenant clinic architecture with Row-Level Security (RLS) and cryptographic tenant separation (`backend/tenancy.py`). | **Resolved** |
| **R17** | Medium | Empirical Retrieval | Integrated Content-Based Medical Image Retrieval (CBMIR) with 512-dim visual embeddings (`backend/vector_search.py`). | **Resolved** |
| **R18** | High | Algorithmic Fairness | Conducted demographic slice disparity audit across age, comorbidity, and optical grade cohorts complying with EEOC 4/5ths rule. | **Resolved** |

---

## 2. Active Roadmap Items

| Issue ID | Priority | Description | Target Milestone | Status |
|---|---|---|---|:---:|
| **A1** | Medium | Mobile Camera Integration: Direct WebRTC camera stream capture for real-time mobile fundoscope attachments. | v2.6 | Active |
| **A2** | Low | Multi-modal Report Generator: Fine-tuned local vision-language report generation (Qwen2.5-VL) synthesizing fundus scans directly into ICD-10 narrative summaries. | v2.6 | Planned |
| **A3** | Medium | Federated Learning: Multi-hospital federated training nodes with differential privacy preserving patient scans across medical centers. | v3.0 | Planned |
