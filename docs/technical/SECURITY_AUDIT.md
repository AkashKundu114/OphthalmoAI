# OphthalmoAI — Security Audit Report
**Classification:** Internal — Engineering
**Scope:** Full codebase (`backend/`, `frontend/`, `k8s/`, `.github/`)
**Remediation status:** All 20 original findings fixed; 3 additional bugs found and fixed this session (see §3).

*(Condensed from the original full audit narrative — this version preserves the finding table and remediation map, which is the operationally useful part; the original per-finding prose walkthroughs, CVSS breakdowns, and PoC snippets are omitted here for length but the substance — what was wrong and how it was fixed — is unchanged.)*

---

## 1. Original Findings (20 total, all fixed)

| ID | Severity | Finding | Fix |
|---|---|---|---|
| C1 | Critical | Hardcoded JWT default secret enabled token forgery | Startup guard rejects placeholder in production |
| C2 | Critical | No token revocation after logout | `TokenBlacklist` (JTI-based, in-memory) |
| C3 | Critical | File upload MIME spoofing | `validate_magic_bytes()` checks real file signatures |
| C4 | High | Internal stack traces leaked in 500 errors | `safe_error_detail()` — generic message + correlation ID in prod |
| C5 | High | Missing HTTP security headers | `SecurityHeadersMiddleware` (CSP, HSTS, X-Frame-Options, etc.) |
| C6 | High | CORS wildcard permitted in production | Startup guard rejects `CORS_ORIGINS=*` in production |
| C7 | High | Rate limiting silently disabled if `slowapi` missing | Fails hard in production instead of silently no-op'ing |
| C8 | High | XSS in chat message renderer | DOMPurify sanitisation + safe custom markdown renderer |
| C9 | High | No dependency/container scanning in CI | `security.yml` — Bandit, Semgrep, Trivy, Gitleaks, OWASP, pip-audit, npm audit |
| M1 | Medium | Prompt injection via unsanitised chat input | `sanitise_chat_message()` — 13 regex patterns blocked |
| M2 | Medium | SSRF via unvalidated `OLLAMA_URL` | `validate_ollama_url()` — private/reserved IP ranges blocked |
| M3 | Medium | Weak email validation | RFC-5321 regex + length limits in `validators.py` |
| M4 | Medium | No brute-force protection on login | `LoginAttemptTracker` — 5 failures → 15-min lockout |
| M5 | Low-Medium | No SRI on Google Fonts | Unused Inter font removed; CSP `style-src` allow-lists only the fonts actually used |
| M6 | Medium | Dev-tunnel hostname committed to source | Removed; env-var-driven `allowedHosts` |
| M7 | Medium | Chat endpoint used a manual DB session | Now uses `Depends(get_db)` like every other endpoint |
| M8 | Medium | Image processing DoS (decompression bomb) | `validate_image_dimensions()` sets `Image.MAX_IMAGE_PIXELS` |
| M9 | Medium | No SAST pipeline | Bandit + Semgrep in `security.yml` |
| L1 | Low | License inconsistency (MIT vs Apache) | All references updated to Apache 2.0 |
| L2 | Low | Unused Inter font loaded on every page | Removed |
| L3 | Low | Full IP addresses in logs (PII/GDPR) | `anonymise_ip()` masks last IPv4 octet / last 80 IPv6 bits |
| L4 | Low | No container vulnerability scanning | Trivy scans both images in `security.yml` |

---

## 2. New Findings — This Session

These were found by actually importing and running the app (not static review), and are unrelated to the async/endpoint work performed this session:

| Severity | Finding | Fix |
|---|---|---|
| Would break every deployment | `security.py`'s `SecurityHeadersMiddleware` called `response.headers.pop(h, None)` — Starlette's `MutableHeaders` has no `.pop()`. Since this middleware wraps every response, the app would 500 on every single request in production. | `if h in response.headers: del response.headers[h]` |
| Would break every startup | `logging_config.py` paired `structlog.stdlib.add_logger_name` with `PrintLoggerFactory()`; `PrintLogger` has no `.name`, crashing on the first log call. | Dropped the incompatible processor; bind the logger name explicitly instead |
| Would break import | `main.py` imported `JWT_SECRET_KEY` from `auth.py`, which only defined `SECRET_KEY`. | Added an alias in `auth.py` |

None of these three were introduced by this session's async/endpoint changes — all three are present verbatim in the pre-existing pasted source and would have surfaced on the very first real request/startup regardless of any other change.

---

## 3. Remediation Map (files touched, cumulative across the original audit + this session)

`backend/security.py`, `backend/validators.py`, `backend/auth.py`, `backend/main.py`, `backend/logging_config.py` (this session), `backend/db_async.py` (new, this session), `backend/routes_admin.py` (new, this session), `frontend/src/ChatBox.jsx`, `frontend/vite.config.js`, `.github/workflows/ci.yml`, `.github/workflows/security.yml`, `alembic/` (new, this session).

---

## 4. Residual Risk & Recommendations (carried over, still open)

| Item | Priority |
|---|---|
| Migrate `TokenBlacklist`/`LoginAttemptTracker` from in-memory to Redis for multi-instance deployments | High |
| `backend/model_registry.py: set_active()` is still sync-only; `routes_admin.py`'s activate endpoint duplicates its logic against `AsyncSession` rather than calling it | Medium |
| Implement HTTPS-only enforcement at the Ingress level (cert-manager + Let's Encrypt) | High |
| Conduct a clinical validation study before any patient-facing deployment | High |
| Enable GitHub branch protection requiring security scans to pass before merge | Medium |
| Complete `docs/clinical/CLINICAL_VALIDATION.md`'s sign-off table with real numbers | Medium |
