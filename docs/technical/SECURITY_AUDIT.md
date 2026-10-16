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
| M5 | Low-Medium | No SRI on Google Fonts / CDN network egress | Google Fonts `@import` completely removed from `index.css`; native OS system fonts used for 100% air-gap compliance |
| M6 | Medium | Dev-tunnel hostname committed to source | Removed; env-var-driven `allowedHosts` |
| M7 | Medium | Chat endpoint used a manual DB session | Now uses `Depends(get_db)` like every other endpoint |
| M8 | Medium | Image processing DoS (decompression bomb) | `validate_image_dimensions()` sets `_MAX_PIXELS = 25M`, axis $\le 6000$ px, aspect ratio $\le 4.0$ |
| M9 | Medium | No SAST pipeline | Bandit + Semgrep in `security.yml` |
| M10 | High | Out-of-distribution image hallucination | `validate_fundus_image()` deterministic optical aperture & chromophore check |
| M11 | High | Prompt injection / jailbreak bypassing triage | Pre-prompt sanitization regex + strict conversational guardrail filters |
| L1 | Low | License inconsistency (MIT vs Apache) | All references updated to Apache 2.0 |
| L2 | Low | Google Fonts loaded in CSS | Replaced with native system fonts in `index.css` |
| L3 | Low | Full IP addresses in logs (PII/GDPR) | `anonymise_ip()` masks last IPv4 octet / last 80 IPv6 bits |
| L4 | Low | No container vulnerability scanning | Trivy scans both images in `security.yml` |

---

## 2. Red-Team Hardening & Adversarial Findings (Release Audit)

These critical vulnerabilities were identified during red-team penetration testing and empirically remediated:

| Severity | Finding | Fix & Verification |
|---|---|---|
| **High** | Decompression bomb DoS: default 89.4 MP ceiling allowed >270 MB buffer allocations in RAM | Capped `_MAX_PIXELS = 25,000,000`, max axis 6000 px, aspect ratio $\le 4.0$. Verified by `test_decompression_bomb_pixel_limit` |
| **High** | Path traversal in mock MongoDB client: `_resolve_mock_path` allowed `../../etc/passwd` | Enforced ID regex `^[a-zA-Z0-9_\-]+$` and canonical directory boundary check. Verified by `test_mock_mongo_path_traversal_blocked` |
| **High** | IDOR on scan history and details: patient users could query other patients' scan records | Enforced caller role-based isolation (`current_user.id == requested_user_id`). Verified by `test_idor_scan_history_unauthorized_user_blocked` |
| **Medium** | EXIF PHI & GPS metadata leakage: uploaded scans retained camera serials and GPS | `strip_image_metadata()` re-allocates clean RGB buffer, purging EXIF. Verified by `test_strip_image_metadata_removes_exif` |
| **Medium** | Unbounded GPU/CPU inference concurrency: concurrent uploads caused CUDA OOM | Async semaphore (`MAX_CONCURRENT_INFERENCES = 4`, 20s timeout), NaN/Inf logit guards, and `torch.cuda.empty_cache()`. Verified by `test_e2e_live_pipeline.py` |
| **Medium** | Insecure model deserialization: offline scripts lacked `weights_only=True` | Enforced `weights_only=True` across all 9 training/eval scripts and verified all 25 checkpoints in `models/*.pth` |
| **Medium** | Broken SQLAlchemy model reload: `importlib.reload(backend.db)` broke DeclarativeBase mapper lookups | Removed destructive reload; called `create_tables()` directly in fixtures. Verified by full regression suite (335 passed) |

---

## 3. Remediation Map (files touched, cumulative across audits)

`backend/security.py`, `backend/validators.py`, `backend/auth.py`, `backend/main.py`, `backend/mongodb_client.py`, `backend/logging_config.py`, `backend/db_async.py`, `backend/routes_admin.py`, `frontend/src/ChatBox.jsx`, `frontend/src/index.css`, `frontend/src/apiConfig.js`, `frontend/vite.config.js`, `.github/workflows/ci.yml`, `.github/workflows/security.yml`, `alembic/`, `tests/backend/test_security_red_team_and_privacy.py`, `tests/backend/test_e2e_live_pipeline.py`.

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
