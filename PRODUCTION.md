# PRODUCTION.md - OphthalmoAI Production Deployment Guide

Three deployment paths: **Docker Compose** (single VM), **Azure Container Apps**
(`docs/technical/AZURE_DEPLOY.md`), or **Kubernetes/AKS** (`k8s/`).

---

## Prerequisites

- Docker ≥ 24, Docker Compose v2
- Trained model weights in `models/`
- Google Gemini API key (free-tier Flash model, default `gemini-2.0-flash`) **or**
  Ollama running and reachable
- `JWT_SECRET_KEY`: `python -c "import secrets; print(secrets.token_hex(32))"`

---

## Environment Variables

Copy `env.example` to `.env` and fill in real values. Never commit `.env`. Key ones:

```env
JWT_SECRET_KEY=<32-byte hex>
ENVIRONMENT=production # enables startup guards, HSTS, safe errors
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-2.0-flash # free-tier Flash — see backend/main.py comment
DATABASE_URL=postgresql://user:pass@host:5432/ophthalmoai?sslmode=require
CORS_ORIGINS=https://yourdomain.com # never "*" in production
```

**Added this session:** `ASYNC_DATABASE_URL` — normally left unset; `backend/db_async.py`
derives it automatically from `DATABASE_URL` (`sqlite:///` → `sqlite+aiosqlite:///`,
`postgresql://` → `postgresql+asyncpg://`). Only set it if the async engine needs to
point somewhere different from the sync one.

---

## Docker Compose

```bash
docker compose up --build -d
```

**Added this session - run once after the backend container is healthy:**

```bash
docker compose exec backend alembic upgrade head
```

The app's `create_tables()` call at startup still creates tables on a fresh database (a
no-op against existing ones), but it cannot apply schema *changes*. `alembic upgrade
head` is the real migration path now — see `alembic/` and `backend/main.py`'s comment
at the `create_tables()` call site.

### Startup guards (production)

The backend refuses to start if: `JWT_SECRET_KEY` is still the placeholder,
`CORS_ORIGINS` is `"*"`, `CORS_ORIGINS=*` is combined with credentials, or `OLLAMA_URL`
resolves to a private/reserved IP.

---

## Azure Container Apps

See `docs/technical/AZURE_DEPLOY.md` and `deploy.sh`.

---

## Kubernetes / AKS

```bash
docker build -t ophthalmoai-backend:latest -f backend/Dockerfile .
docker build -t ophthalmoai-frontend:latest -f frontend/Dockerfile --build-arg VITE_API_URL=/api .
kubectl create namespace ophthalmoai
kubectl create secret generic ophthalmoai-secrets --namespace ophthalmoai \
  --from-literal=GEMINI_API_KEY=AIza... \
  --from-literal=JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
kubectl apply -k k8s
kubectl -n ophthalmoai rollout status deployment/backend
```

**Added this session:** run the Alembic migration as a one-shot Job or `kubectl exec`
into a running backend pod (`alembic upgrade head`) before routing real traffic — not
automated as part of `kubectl apply -k k8s` in this revision.

`backend/Dockerfile` uses the CPU-only PyTorch wheel; GPU scheduling requires a separate
CUDA-based image (not included).

---

## Health Checks

| Endpoint | Type |
|----------|------|
| `GET /health` | Liveness — always 200 if process alive |
| `GET /ready` | Readiness — 503 if router model not loaded |

---

## Security Hardening Checklist

| Area | Setting |
|------|---------|
| `ENVIRONMENT` | `production` |
| `CORS_ORIGINS` | Exact domain(s), never `*` |
| File uploads | Magic-byte validated, 20 MB limit, decompression-bomb guard |
| Rate limiting | `slowapi` required in production |
| Security headers | CSP, HSTS (prod), X-Frame-Options via `SecurityHeadersMiddleware` — **a bug in this middleware that crashed every response was found and fixed this session; see `docs/technical/SECURITY_AUDIT.md`** |
| Login | 5 failures → 15-minute lockout |
| Dependency scanning | `pip-audit` + `npm audit` in CI |
| DB migrations | **Alembic, added this session** — was previously `create_all()` only |

---

## Upgrading

```bash
git pull origin main
docker compose up --build -d
docker compose exec backend alembic upgrade head # added this session
```
