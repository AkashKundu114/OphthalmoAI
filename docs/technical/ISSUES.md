# OphthalmoAI — Issues & Technical Debt Tracker

*(Condensed from the original tracker; statuses updated to reflect fixes already applied in the codebase, including three bugs found and fixed this session.)*

> See [`ROADMAP.md`](../../ROADMAP.md) for how these map onto release milestones.

## Summary

| Severity | Resolved | Open |
|---|---|---|
| 🔴 Critical | 5 | 0 |
| 🟠 High | 4 | 1 |
| 🟡 Medium | 7 | 2 |
| 🟢 Low | 5 | 0 |

---

## 🔴 Critical — Security & Correctness (all resolved)

- **C1. CORS wildcard + credentials.** Resolved — startup guard in `main.py` rejects the combination.
- **C2. No request validation on `/predict`.** Resolved — `security.py`'s `validate_magic_bytes()`/`validate_image_dimensions()`.
- **C3. No rate limiting.** Resolved — `slowapi` on `/predict`, `/chat`, `/auth/*`.
- **C4. No authentication on inference endpoints.** Resolved — JWT + RBAC (`patient`/`clinician`/`admin`).
- **C5. LICENSE inconsistency (MIT vs Apache).** Resolved — all references now say Apache 2.0.

## 🟠 High — Functional Bugs & Reliability

- **H1. Only 3 of 8 symptoms reached `/predict`.** Resolved — all 8 fields sent and used.
- **H2. Hardcoded LLM model snapshot.** Resolved — `GEMINI_MODEL` env var, defaults to `gemini-2.0-flash` (Google's free-tier Flash model).
- **H3. Single synchronous Uvicorn worker.** **Open.** CPU inference can serialize under load; documented mitigation (multiple workers or a Celery/Redis queue) not implemented.
- **H4. Backend dependencies unpinned.** Resolved — `backend/requirements.txt` pins exact versions.
- **H5. Broken `<meta description>` tag.** Resolved.

## 🟡 Medium — Code Quality & Consistency

- **M1. Dead `App.css` file.** Resolved (removed).
- **M2. No Vite dev proxy.** Resolved.
- **M3. Condition data duplicated frontend/backend.** Resolved — `GET /conditions` is the single source of truth.
- **M4. Personal tunnel hostname in `vite.config.js`.** Resolved — env-var-driven `allowedHosts`.
- **M5. Unused Inter font.** Resolved.
- **M6. `/predict`/`/chat` returned HTTP 200 on errors.** Resolved — proper status codes (413/415/422/503/500).
- **M7. `specialist_eyelid.pth` trained but unused at inference.** **Open** — documented as intentional (router confidence used directly for the single-class Adnexal group), not wired in as a second-opinion step.
- **M8. `ChatBox` only rendered `**bold**` markdown.** Resolved — `renderMarkdown()` handles bullet lists, code, italics, all DOMPurify-sanitised.
- **M9. Frontend version stuck at `0.0.0`.** **Open**, low priority.

## 🟢 Low — Polish (all resolved)

- **L1. Default Vite favicon.** Resolved (custom eye-mark SVG).
- **L2. No `.env.example`.** Resolved.
- **L3. Hardcoded copyright year.** Resolved (`{new Date().getFullYear()}`).
- **L4. Emoji-prefixed alert strings couple presentation to data.** Resolved — `hybrid_warnings_structured` carries `{severity, message}` separately from the legacy emoji-prefixed `hybrid_warnings` list.
- **L5. Modal focus trap not implemented.** Still not implemented; acknowledged as a known accessibility gap.

---

## Added this session (not in the original tracker)

- **New bug — `main.py` imported `JWT_SECRET_KEY` from `auth.py`, which only defined `SECRET_KEY`.** Found via a real import test; fixed with an alias.
- **New bug — `logging_config.py` crashed on first log call** due to `add_logger_name` + `PrintLoggerFactory` incompatibility. Fixed.
- **New bug (most severe) — `security.py`'s header-stripping middleware called the nonexistent `MutableHeaders.pop()`,** which would 500 every response in production. Fixed.
- **New gap — async DB path needed new dependencies** (`aiosqlite`, `asyncpg`) not previously in `requirements.txt`. Added.
- **New capability — three previously-documented-but-missing endpoints implemented** (`POST /scans/{id}/override`, `GET /admin/audit-logs`, `POST /admin/model-registry/activate`), each with tests in `tests/backend/test_routes_admin.py`.
- **New capability — Alembic migrations scaffolded and verified** (upgrade/downgrade round-trip against the real schema).
