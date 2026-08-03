# Backend Schema & API Reference
## OphthalmoAI

**Base URL (local dev):** `http://localhost:8000`  
**Base URL (Docker prod):** `http://localhost:8000` (or via Nginx `/api/` proxy)

---

## 1. Overview

The backend is a **FastAPI** app. Models are loaded once at startup into module globals. Image uploads are processed in memory; files are not persisted to disk unless `SCAN_STORAGE_BUCKET` is configured. `/predict`, `/chat`, and the `/auth/*` endpoints are rate-limited via `slowapi`.

**Added this session:** three admin/clinician endpoints (`backend/routes_admin.py`), and `get_current_user`/`require_role`/`authenticate_user` (in `backend/auth.py`) now use a real async SQLAlchemy session (`backend/db_async.py`) instead of blocking the event loop on every request.

---

## 2. Startup & Model Loading

Models are loaded via FastAPI's `lifespan` async context manager on application startup.

### Loading sequence

```
1. Build router architecture (MobileNetV3-Large, 3 output classes)
2. Load router.pth → router.to(device).eval()
3. For each group in HIERARCHY:
   a. If single class (Adnexal/Eyelid): register as "direct" pass-through
   b. Otherwise: build EfficientNet-B4 with N output classes
      → load specialist_*.pth → model.to(device).eval()
4. Yield (app is now ready to serve requests)
5. On shutdown: gc.collect() + torch.cuda.empty_cache()
```

### HIERARCHY constant

```python
HIERARCHY = {
    0: {
        'name': 'Adnexal Oculoplastic',
        'model_file': 'specialist_eyelid.pth',
        'classes': ['Eyelid']          # Single class → direct pass-through
    },
    1: {
        'name': 'Anterior Segment Pathology',
        'model_file': 'specialist_anterior.pth',
        'classes': ['Cataract', 'Uveitis']
    },
    2: {
        'name': 'Ocular Surface Disorders',
        'model_file': 'specialist_surface.pth',
        'classes': ['Conjunctivitis', 'Jaundice', 'Normal', 'Pterygium']
    }
}
```

---

## 3. Image Pre-processing

Applied identically to all inference inputs:

```python
transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Output tensor shape: `[1, 3, 380, 380]` (batch of 1).

---

## 4. Endpoints

### `GET /`

System status overview.

### `GET /health`

Simple liveness probe.

### `GET /ready`

Readiness probe. Returns 503 if the router model is not loaded.

### `GET /conditions`

Returns clinical metadata for all 7 detectable conditions, sourced from `backend/medical_data.py`.

### `POST /predict`

Main inference endpoint. Rate limited (default `10/minute` per IP).

**Request:** `multipart/form-data` — `file`, `pain`, `vision`, `itch` (required); `halos`, `discharge`, `light_sens`, `floaters`, `duration` (optional, all 8 symptom fields feed the cross-check engine).

**Response 200:** diagnosis, confidence, heatmap, probabilities, `hybrid_warnings`/`hybrid_warnings_structured`, calibration fields, uncertainty fields, ICD-10/SNOMED/urgency fields, IQA fields. Full shape unchanged from prior versions of this doc — see `backend/main.py: predict()`.

### `POST /chat`

AI Doctor chatbot proxy. Rate limited (default `30/minute` per IP). Routes to Google Gemini (`GEMINI_API_KEY` set, model defaults to `gemini-2.0-flash` — the Google AI Studio free-tier Flash model) or Ollama (`OLLAMA_URL` set) as a local fallback.

### `POST /auth/register`, `POST /auth/token`, `POST /auth/logout`, `GET /auth/me`

Standard JWT auth flow. `/auth/register` and `/auth/token` use an `AsyncSession` (added this session — see `backend/db_async.py`).

### `POST /scans/{scan_id}/override` — **added this session**

Records a clinician's second opinion on a scan. Role: `clinician` or `admin`.

**Request body:**
```json
{
  "verdict": "agree" | "disagree" | "inconclusive" | "insufficient_image_quality",
  "corrected_diagnosis": "string, required if verdict is 'disagree'",
  "corrected_icd10": "string, optional",
  "notes": "string, optional"
}
```

**Responses:** `201` on success; `404` unknown scan; `409` a scan can have at most one override (append-only, enforced at the DB level and checked here for a clean error); `422` invalid verdict or missing `corrected_diagnosis`; `401`/`403` auth failures.

### `GET /admin/audit-logs` — **added this session**

Paginated audit log query. Role: `admin` only. Query params: `action`, `user_id`, `success`, `limit` (default 50, max 500), `offset`.

### `POST /admin/model-registry/activate` — **added this session**

Promotes a `ModelVersion` to active for its group, deactivating any other active version in the same group. Role: `admin` only.

**Request body:** `{"version_id": "..."}`

**Response** includes an explicit `warning` field stating this updates the DB record only — it does not hot-swap weights in the running process's memory; a restart is required for the change to affect live inference (see `docs/clinical/CLINICAL_SAFETY.md` §6).

---

## 5. Symptom Cross-Check Rules

The `analyze_symptoms()` / `analyze_symptoms_structured()` functions implement 10 rules across all 8 symptom fields — see `backend/main.py: _build_symptom_alerts()` for the authoritative logic. Each rule's severity (`info` | `warning` | `urgent`) maps to an emoji prefix (`✅` | `⚠️` | `🚨`) in the legacy `hybrid_warnings` string list, and to a `{"severity", "message"}` object in `hybrid_warnings_structured`.

---

## 6. Medical Data Schema

`backend/medical_data.py: MEDICAL_INFO` — one entry per of the 7 conditions (`Cataract`, `Conjunctivitis`, `Eyelid`, `Jaundice`, `Uveitis`, `Normal`, `Pterygium`), each with `name`, `group`, `color`, `analysis`, `description`, `symptoms`, `treatment`, `precautions`, `severity`, `advice`. Also exposed via `GET /conditions`.

---

## 7. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|--------------|
| `GEMINI_API_KEY` | `""` | Takes priority over Ollama |
| `GEMINI_MODEL` | `"gemini-2.0-flash"` | Free-tier Flash model, used by `/chat` |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `""` / `"llama3.2:3b"` | Local LLM fallback |
| `DATABASE_URL` | `sqlite:///./ophthalmoai.db` | Sync DB URL |
| `ASYNC_DATABASE_URL` | derived from `DATABASE_URL` | **Added this session** — see `backend/db_async.py` |
| `JWT_SECRET_KEY` | placeholder (must be changed) | |
| `PREDICT_RATE_LIMIT` / `CHAT_RATE_LIMIT` / `AUTH_RATE_LIMIT` | `10/minute` / `30/minute` / `20/minute` | |
| `MAX_FILE_SIZE_BYTES` | `20971520` | |
| `CORS_ORIGINS` / `CORS_ALLOW_CREDENTIALS` | `"*"` / `"false"` | |

Full list in `env.example`.
