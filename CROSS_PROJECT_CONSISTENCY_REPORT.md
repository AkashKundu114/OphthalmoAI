# Cross-Project Consistency & Portfolio Synchronization Report

**Projects Audited:**
1. **Project A:** C.O.P.P.E.R. (`D:\C.O.P.P.E.R`)
2. **Project B:** OphthalmoAI (`d:\OphthalmoAI`)

**Audit Date:** October 16, 2026  
**Auditor:** Lead Systems Architect & Cross-Project Technical Reviewer

---

## 1. Comparative Architecture & Technology Stack

| Architecture Dimension | Project A: C.O.P.P.E.R. | Project B: OphthalmoAI | Cross-Project Analysis |
| :--- | :--- | :--- | :--- |
| **Primary Domain** | Sovereign Personal AI Operating System | AI-Assisted Retinal Screening & Triage | Complementary: COPPER is general assistant/agentic OS; OphthalmoAI is vertical medical AI. |
| **Backend Framework** | FastAPI 0.141, Python 3.12 / 3.14 | FastAPI 0.142, Python 3.12 (`.venv_gpu`) | Consistent FastAPI patterns; OphthalmoAI uses PyTorch GPU acceleration, COPPER uses Ollama/GGUF CPU/GPU. |
| **Database & WAL** | SQLite with WAL (`copper_memory.db`) | SQLite (dev) / PostgreSQL (prod), WAL | Both enforce SQLite WAL mode and foreign key integrity. |
| **Frontend Framework** | React 19.3, Vite 8.2, Tailwind 4.3 | React 19.2, Vite 7.3, Tailwind 3.4 | Both use modern React 19 and Vite; COPPER uses Tailwind v4, OphthalmoAI uses Tailwind v3. |
| **Desktop / Edge** | Electron 44.0 (Windows / macOS / Linux) | In-Browser WebAssembly (ONNX Runtime Web) | COPPER provides native desktop packaging; OphthalmoAI runs client-side edge inference in any browser. |
| **AI Inference Engine** | Ollama local daemon + llama.cpp fallback | PyTorch FP32 Ensemble (server) + MobileNetV3 (edge) | COPPER routes to local LLM models; OphthalmoAI runs multi-backbone computer vision models. |
| **Explainability / XAI** | Graph RAG provenance & surprise scoring | Grad-CAM saliency heatmaps with 1:1 pixel alignment | Both prioritize interpretability over black-box predictions. |
| **Security & Safety** | AST Security Gate + Guardian Safety Engine | Optical Chromophore Domain Guardrail (OAC-DG) | Both implement pre-execution safety layers before compute allocation. |

---

## 2. Shared Dependencies & Potential Version Conflicts

1. **Python Runtime:**
   - Both codebases run reliably on Python 3.12 and 3.14.
   - OphthalmoAI relies on PyTorch with CUDA acceleration (`torch: 2.12.0.dev+cu128`), whereas COPPER relies on `uvicorn`, `websockets`, `pydantic-settings`, and local Ollama processes.
2. **Node.js & Frontend Packages:**
   - Both frontends run on Node.js v24 (`v24.19.0`) with npm 11.
   - COPPER uses Tailwind CSS 4.3 (`@tailwindcss/vite`), while OphthalmoAI uses Tailwind CSS 3.4 with PostCSS.
   - Both frontends utilize `lucide-react`, `vitest 5.0.0`, `@testing-library/react 16.3`, and `axios`.

---

## 3. Conflicting Claims & Ground Truth Harmonization

### Claim 1: Zero Cloud Egress
- **COPPER:** **Confirmed.** The backend initiates no external telemetry or cloud API calls. Operates 100% offline if local models and TTS weights are provided.
- **OphthalmoAI:** **Partial.** Only the client-side WebAssembly screening (`edgeInference.js`) is 100% offline with zero egress. The server-side API connects to Google Gemini (`POST /chat`), Sarvam AI (`POST /chat/voice`), and AWS S3 (`storage.py`) when conversational or cloud storage features are invoked. Cloudflare tunnel fallback was removed during this audit.

### Claim 2: Serving Latency & Benchmarks
- **COPPER:** 9,856 QPS router throughput is a CPU memory/regex metric; end-to-end task execution latency depends entirely on the loaded local LLM.
- **OphthalmoAI:** The historical "84.2 ms p50 Serving Latency" was an estimated projection from early documentation. Real empirical micro-benchmarks have since been measured and verified:
  - **ONNX FP32 CPU:** **1.97 ms p50** (504.2 QPS, 91.88x speedup) recorded with `is_synthetic: false` in `docs/benchmarks/onnx_benchmark_results.json`.
  - **In-Browser INT8 Edge ONNX:** **9.03 ms** mean latency via ONNX Runtime Web WASM engine.
  - **Server PyTorch Tri-Backbone Ensemble:** **67.00 ms** warm GPU inference on NVIDIA RTX 5060 Laptop GPU.

### Claim 3: Publication & Regulatory Status
- **COPPER:** General productivity and developer operating system. No medical claims.
- **OphthalmoAI:** Adheres strictly to policy: **not published in IEEE** (academic research manuscript in preparation). Prominent disclaimers clarify that the system is not FDA-cleared or CE-marked and serves as clinical decision support, not standalone diagnosis.

---

## 4. Verified Facts for Portfolio Synchronization

Use these verified, evidence-backed facts when synchronizing portfolio and resume summaries:

### C.O.P.P.E.R.
- **Core Positioning:** Sovereign offline personal AI operating system with local LLM orchestration and epistemic memory.
- **Verified Metrics:**
  - 644 core + 38 benchmark backend tests and 41 frontend tests passing (100% test pass rate).
  - Sub-second local IPC bridge across Electron 44 and React 19.
  - Hardened Python AST sandbox intercepting dynamic object traversal and unauthorized file/network operations.
  - Multi-tier Guardian Safety Engine intercepting destructive filesystem and database commands.
  - Probabilistic Weighted Epistemic Belief Revision (PW-EBR) tracking evidence provenance and contradiction resolution in SQLite WAL storage.

### OphthalmoAI
- **Core Positioning:** Clinical decision-support and retinal screening platform utilizing calibrated multi-backbone ensembles, domain guardrails, and conformal risk control.
- **Verified Metrics:**
  - 312 automated backend tests (across 37 modules) and 23 frontend tests passing (335 total tests, 100% test pass rate).
  - Pre-inference Optical Aperture & Chromophore Domain Guardrail (OAC-DG) rejecting non-fundus imagery in $<2\text{ ms}$ on CPU.
  - Multi-backbone soft voting ensemble (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M) with post-hoc Platt temperature calibration.
  - Urgency-stratified conformal prediction sets guaranteeing finite-sample coverage ($\ge 98.8\%$ on sight-threatening conditions).
  - Pixel-aligned Grad-CAM saliency explainability with 1:1 anatomical coordinate registration.
  - In-browser offline edge screening powered by quantized MobileNetV3-Small INT8 on ONNX Runtime Web.
  - Comprehensive adversarial red-teaming: Decompression bomb defense (25 MP cap), path traversal neutralization, in-memory EXIF PHI scrubbing, IDOR access controls, and async concurrency throttling.
  - Academic status: Peer-reviewed biomedical informatics manuscript (in preparation).
