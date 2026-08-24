# Setup Checklist
## OphthalmoAI

---

## 1. Quick Checklist

Use this list to verify a complete local setup from scratch.

### Environment

- [ ] Python 3.10+ installed
- [ ] Node.js 18+ and npm installed
- [ ] Git installed
- [ ] CUDA 12.x driver installed (if using GPU)
- [ ] (Optional) Ollama installed from https://ollama.ai

### Repository

- [ ] `git clone https://github.com/AkashKundu114/Eye-Disease-AI-Diagnosis.git`
- [ ] `cd Eye-Disease-AI-Diagnosis`
- [ ] `python -m venv venv && source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
- [ ] `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- [ ] `python scripts/check_setup.py` → confirm GPU detected
- [ ] `pip install -r backend/requirements.txt`

### Dataset

- [ ] Dataset organised per the directory structure in README
- [ ] `python scripts/verify_dataset.py` → 0 corrupt images
- [ ] `python scripts/explore_data.py` → distributions look reasonable

### Model Training

- [ ] `python scripts/train_router.py` → `models/router.pth` saved
- [ ] `python scripts/train_anterior.py` → `models/specialist_anterior.pth` saved
- [ ] `python scripts/train_surface.py` → `models/specialist_surface.pth` saved
- [ ] `python scripts/train_eyelid.py` → `models/specialist_eyelid.pth` saved (optional — not used in inference)

### Database (added this session)

- [ ] `alembic upgrade head` → creates/updates `users`, `scan_results`,
      `clinician_overrides`, `audit_logs`, `model_versions` tables
      (previously this relied solely on `create_tables()` at app startup, which
      creates tables but cannot apply schema changes to an existing database)

### Backend

- [ ] `.env` file created in project root
- [ ] `ANTHROPIC_API_KEY` or `OLLAMA_URL` set in `.env`
- [ ] `python backend/main.py` starts without errors
- [ ] `curl http://localhost:8000/health` → `{"ok": true, ...}`
- [ ] `http://localhost:8000/docs` loads Swagger UI

### Frontend

- [ ] `cd frontend && npm install`
- [ ] `npm run dev` starts without errors
- [ ] `http://localhost:5173` loads the app
- [ ] Upload a test image → result returned
- [ ] Chat sends and receives a response

### Docker (optional)

- [ ] Docker Desktop installed and running
- [ ] `.env` populated with API keys
- [ ] `docker compose up --build`
- [ ] `http://localhost:8080` loads the app
- [ ] `http://localhost:8000/health` returns `{"ok": true}`

---

## 2. Build Notes

### Phase 1 — Infrastructure Setup (Day 1)

| Task | Owner | Status |
|------|-------|--------|
| Repository structure defined | Dev | Done |
| Python virtual environment + PyTorch install | Dev | Done |
| FastAPI skeleton with lifespan + CORS | Dev | Done |
| Vite + React + Tailwind setup | Dev | Done |
| Docker Compose (backend + frontend) | Dev | Done |
| `.gitignore` / `.dockerignore` | Dev | Done |

### Phase 2 — Model Architecture & Training (Days 2–5)

| Task | Owner | Status |
|------|-------|--------|
| Dataset directory structure defined | Dev | Done |
| `verify_dataset.py` + `explore_data.py` | Dev | Done |
| `train_router.py` (MobileNetV3, 3 classes) | Dev | Done |
| `train_anterior.py` (EfficientNet-B4, 2 classes) | Dev | Done |
| `train_surface.py` (EfficientNet-B4, 4 classes) | Dev | Done |
| `train_eyelid.py` (single-class helper) | Dev | Done |
| WeightedRandomSampler class balancing | Dev | Done |
| AMP mixed precision + gradient accumulation | Dev | Done |

### Phase 3 — Backend Inference API (Days 3–6)

| Task | Owner | Status |
|------|-------|--------|
| Model loading via lifespan context manager | Dev | Done |
| `/predict` endpoint — full inference pipeline | Dev | Done |
| Grad-CAM integration (pytorch-grad-cam) | Dev | Done |
| Heatmap → base64 JPEG encoding | Dev | Done |
| Symptom cross-check logic | Dev | Done |
| `MEDICAL_INFO` dictionary (7 conditions) | Dev | Done |
| `/chat` endpoint — Gemini + Ollama support | Dev | Done |
| Ophthalmology system prompt | Dev | Done |
| `/health` and `/ready` probes | Dev | Done |
| GPU cache cleanup per request | Dev | Done |
| Async DB path for auth (`db_async.py`) | Dev | Done (this session) |
| Admin endpoints (`routes_admin.py`) | Dev | Done (this session) |
| Alembic migrations | Dev | Done (this session) |

### Phase 4 — Frontend Core (Days 4–8)

| Task | Owner | Status |
|------|-------|--------|
| Design system (tokens, fonts, Tailwind config) | Dev | Done |
| Navigation (5-tab SPA, mobile + desktop) | Dev | Done |
| Home page (hero, quick-access grid, features) | Dev | Done |
| DiagnosticPage — file upload + crop modal | Dev | Done |
| Symptom form (8 dropdowns) | Dev | Done |
| API call to `/predict` | Dev | Done |
| Result card (diagnosis, confidence, severity) | Dev | Done |
| Clinical alerts strip | Dev | Done |
| Heatmap toggle | Dev | Done |
| Tabbed detail panel (4 tabs) | Dev | Done |
| Probability bars (`<ProbabilityBar>`) | Dev | Done |
| TTS (Web Speech API) | Dev | Done |
| `ChatBox.jsx` — floating chat widget | Dev | Done |
| Quick question chips | Dev | Done |
| How It Works page | Dev | Done |
| Conditions page + modal | Dev | Done |
| Medical News page + category filter | Dev | Done |

### Phase 5 — PDF Report (Days 7–9)

| Task | Owner | Status |
|------|-------|--------|
| 4-page jsPDF layout | Dev | Done |
| Patient scan + heatmap images in PDF | Dev | Done |
| `autoTable` for symptoms, treatment, precautions | Dev | Done |
| Differential diagnosis bar chart | Dev | Done |
| Emergency signs section (Page 4) | Dev | Done |
| "Find an Ophthalmologist" links (Page 4) | Dev | Done |
| Clinical disclaimer footer (all pages) | Dev | Done |

### Phase 6 — Infrastructure & Production (Days 9–11)

| Task | Owner | Status |
|------|-------|--------|
| Backend Dockerfile (CPU torch wheel) | Dev | Done |
| Frontend Dockerfile (multi-stage build) | Dev | Done |
| Nginx config (proxy, SPA, cache, healthz) | Dev | Done |
| Docker Compose (healthchecks, depends_on) | Dev | Done |
| Kubernetes manifests (namespace, configmap, deployments, services, ingress) | Dev | Done |
| Kubernetes liveness + readiness probes | Dev | Done |
| Official NVIDIA NGC PyTorch Container (`Dockerfile.gpu`) | Dev | Done (this session) |
| Hardware Telemetry & Benchmarking JSON extraction | Dev | Done (this session) |

### Phase 7 — Documentation (Day 11–12)

| Task | Owner | Status |
|------|-------|--------|
| README.md | Dev | Done |
| PRODUCTION.md | Dev | Done |
| `docs/PRD.md` | Dev | Done |
| `docs/TRD.md` | Dev | Done |
| `docs/APP_FLOW.md` | Dev | Done |
| `docs/UI_UX_BRIEF.md` | Dev | Done |
| `docs/BACKEND_SCHEMA.md` | Dev | Done |
| `docs/IMPLEMENTATION_PLAN.md` | Dev | Done |

---

## 3. Known Issues & Recommended Fixes

These items are documented in the TRD but summarised here as actionable tasks.

### Issue 1 — Extra symptom fields not sent to API

**Status: Resolved.** `frontend/src/App.jsx` — `handleAnalyze()` now sends all 8 symptom
fields, and `backend/main.py`'s `/predict` endpoint and `_build_symptom_alerts()` accept
and use all of them.

### Issue 2 — Unused `App.css` file

**Status: Resolved.** The unused Vite-template `App.css` was removed; nothing imported it.

### Issue 3 — No Vite proxy for development

**Status: Resolved.** `frontend/vite.config.js` includes the `/api` proxy.

### Issue 4 — No input image validation on backend

**Status: Resolved.** `backend/security.py`'s `validate_magic_bytes()` and
`validate_image_dimensions()` are called from `/predict` before any PIL/PyTorch processing.

### Issue 5 — Single Uvicorn worker limits concurrency

**Status: Open.** Still a known limitation — safe for GPU (avoids model contention) but
blocks concurrent requests on CPU. A Celery/Redis queue is the documented medium-term fix
(see `ROADMAP.md` §3.1); not implemented.

---

## 4. Future Roadmap (v2.0+)

### High Priority

| Feature | Notes |
|---------|-------|
| Expand symptom API | Done — all 8 symptom fields now sent to backend |
| User accounts / history | Done — JWT auth + `ScanResult`/`AuditLog` tables exist |
| Retinal fundus support | Not started — see `ROADMAP.md` §2.2 |
| Mobile app | Not started |

### Medium Priority

| Feature | Notes |
|---------|-------|
| Multi-language | Not started |
| Confidence calibration | Done — `backend/calibration.py`, temperature scaling |
| Model versioning | Done — `backend/model_registry.py` + admin activate endpoint |
| Offline mode | Not started |

### Low Priority / Research

| Feature | Notes |
|---------|-------|
| Federated learning | Not started |
| OCT / Fundus specialist | Not started |
| DICOM support | Not started |
| EHR integration | Not started |
| Batch inference API | Not started |
