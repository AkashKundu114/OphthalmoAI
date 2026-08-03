# Technical Notes
## OphthalmoAI — v2.2

*(Condensed from the original v2.1 TRD; updated to reflect this session's changes. See `git log` / `WIRING.md` for the detailed change history if this repo is under version control.)*

---

## 1. System Overview

- **Backend:** Python / FastAPI serving a PyTorch inference pipeline via REST API, with JWT auth, SQLAlchemy persistence (sync + async), Alembic migrations, and structured logging.
- **Frontend:** React SPA served by Nginx (production) or Vite dev server (development).

---

## 2. Technology Stack — changes this session

| Component | Change |
|---|---|
| DB access (auth) | `backend/auth.py` now uses `AsyncSession` (`backend/db_async.py`) instead of blocking sync calls for `get_current_user`, `require_role`, `authenticate_user` |
| Schema management | Alembic added (`alembic/`, `alembic.ini`) — `alembic upgrade head` is now the documented path for schema changes; `create_tables()` remains for first-run convenience only |
| New endpoints | `backend/routes_admin.py`: `POST /scans/{id}/override`, `GET /admin/audit-logs`, `POST /admin/model-registry/activate` |
| New dependencies | `aiosqlite==0.20.0`, `asyncpg==0.30.0` added to `backend/requirements.txt` |
| CI | `.github/workflows/ci.yml` gained an Alembic upgrade/downgrade/upgrade validation step |

Everything else in the original stack table (FastAPI, PyTorch, MobileNetV3/EfficientNet-B4, pytorch-grad-cam, slowapi, structlog, React 19, Vite 7, Tailwind 3, jsPDF, react-easy-crop, DOMPurify) is unchanged.

---

## 3. Three Bugs Found and Fixed This Session

Found by actually running the app end-to-end (import + live `TestClient` requests), not by static review. All three were present in the codebase before this session's changes and are unrelated to the async/endpoint work itself:

1. **`backend/main.py` imported `JWT_SECRET_KEY` from `backend/auth.py`, which only defined `SECRET_KEY`.** `ImportError` on any clean checkout. Fixed with an alias.
2. **`backend/logging_config.py` paired `structlog.stdlib.add_logger_name` with `PrintLoggerFactory()`.** `PrintLogger` has no `.name` attribute, so this crashed on the very first log call — i.e., at startup. Fixed by dropping that processor and binding the logger name explicitly instead.
3. **`backend/security.py`'s `SecurityHeadersMiddleware` called `response.headers.pop(h, None)`.** Starlette's `MutableHeaders` doesn't implement `.pop()`. Since this middleware wraps every response, the app would have 500'd on every single request in production. Fixed with `del response.headers[h]` guarded by a membership check.

---

## 4. Data Models (DB) — unchanged from v2.1

`users`, `scan_results`, `clinician_overrides`, `audit_logs`, `model_versions` — see `backend/db.py` for the authoritative SQLAlchemy models, and `alembic/versions/0001_initial_schema.py` (added this session) for the migration that creates them.

---

## 5. Model Specifications — unchanged from v2.1

Router: MobileNetV3-Large, 3 classes, 224×224 input. Specialists: EfficientNet-B4, 380×380 input (2 classes anterior, 4 classes surface). Adnexal/Eyelid: direct pass-through, no specialist model at inference. See `docs/technical/BACKEND_SCHEMA.md` for full detail.

---

## 6. Security — additions this session

| Requirement | Implementation |
|-------------|---------------|
| Async auth path | `backend/db_async.py`, real `AsyncSession` + `select()` in `auth.py` |
| Schema migrations | Alembic, verified `upgrade head` → `downgrade base` → `upgrade head` round-trip |
| Admin-only endpoints | `GET /admin/audit-logs`, `POST /admin/model-registry/activate` gated via `require_role("admin")` |
| Clinician-only endpoint | `POST /scans/{id}/override` gated via `require_role("clinician", "admin")`, with append-only (409 on duplicate) enforcement |

Everything else in the original Security table (JWT/JTI blacklist, brute-force lockout, CORS wildcard guard, magic-byte file validation, security headers, prompt-injection filtering, SSRF guard, IP anonymisation, audit trail, rate limiting, container security) is unchanged — see `docs/technical/SECURITY_AUDIT.md`.

---

## 7. Known Limitations & Technical Debt — updated

| Item | Status |
|------|--------|
| Single Uvicorn worker limits CPU concurrency | Open |
| In-memory TokenBlacklist / LoginAttemptTracker not shared across workers | Open |
| `specialist_eyelid.pth` trained but unused at inference | Open |
| `backend/model_registry.py: set_active()` still sync-only | Open — `routes_admin.py`'s activate endpoint re-implements the two statements against `AsyncSession` directly rather than calling it; consolidating is a follow-up |
| `backend/audit.py: log_event()` still sync-only | Partially addressed — `register()`/`login()` now use a dedicated async-safe audit write (`_log_audit_async()` in `main.py`) so they don't silently stop writing to `audit_logs`; other call sites (`predict`, `chat`) are unaffected since they still use the sync `Session` |
| No secrets manager integration | Open |
| No Prometheus/Grafana/Sentry monitoring | Open |
