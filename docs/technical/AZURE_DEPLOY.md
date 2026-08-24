# AZURE_DEPLOY.md — Hosting OphthalmoAI on Azure

Full step-by-step Azure Container Apps deployment, compatible with the GitHub Student
Developer Pack's $100 Azure credit.

**The authoritative, runnable version of every command below is `deploy.sh`** — this
document explains what that script does and why; run the script itself rather than
copy-pasting commands out of this file, so the two never drift apart.

---

## 1. What Gets Deployed

| Resource | Azure Service |
|----------|--------------|
| Backend API | Azure Container Apps (Consumption) |
| Frontend | Azure Container Apps (Consumption) |
| Container Registry | Azure Container Registry (Basic) |
| Database | Azure Database for PostgreSQL Flexible Server (Burstable B1ms) |
| Secrets | Azure Container Apps Secrets |
| Logs | Azure Log Analytics |
| CI/CD | GitHub Actions (`k8s/azure-deploy.yml`) |
| HTTPS | Azure Container Apps built-in managed certificate |

Estimated cost: ~$37/month, covering 2–3 months on the Student credit. Stop the
PostgreSQL server when idle (`az postgres flexible-server stop`) — Container Apps in
Consumption mode already scale to zero.

---

## 2. Prerequisites

- GitHub Student Developer Pack (claim Azure credit at education.github.com/pack)
- Docker Desktop, Azure CLI (`az`), `az extension add --name containerapp --upgrade`
- Trained model weights in `models/` (`router.pth`, `specialist_anterior.pth`, `specialist_surface.pth`)
- A Google Gemini API key (free tier — `ai.google.dev`), or an Ollama endpoint

---

## 3. Run the Deployment

```bash
export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export GEMINI_API_KEY="your-gemini-key"
export PG_PASSWORD="YourStr0ng!Pass"
bash deploy.sh
```

`deploy.sh` provisions, in order: resource group → Log Analytics workspace → Container
Apps environment → ACR → builds/pushes both images → PostgreSQL Flexible Server →
backend Container App (with `JWT_SECRET_KEY`/`GEMINI_API_KEY`/`DATABASE_URL` as secrets)
→ frontend Container App → updates backend `CORS_ORIGINS` to the deployed frontend URL.

**Added this session — run once after `deploy.sh` completes and before serving real
traffic:**

```bash
az containerapp exec \
  --name ophthalmoai-backend --resource-group ophthalmoai-rg \
  --command "alembic upgrade head"
```

`deploy.sh`'s `create_tables()` call at app startup creates tables on a fresh database,
but does not apply schema *changes* to an existing one — `alembic upgrade head` is the
real migration path (see `alembic/` and `docs/planning/TRD.md`).

---

## 4. GitHub Actions CD

`k8s/azure-deploy.yml` runs on every push to `main`: tests (including the Alembic
migration round-trip check added this session) → build & push both images → deploy →
health-check poll. Requires these repository secrets: `AZURE_CLIENT_ID`,
`AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `ACR_SERVER`,
`ACR_USERNAME`, `ACR_PASSWORD`, `RESOURCE_GROUP`, `BACKEND_APP_NAME`, `FRONTEND_APP_NAME`.

Create the service principal:

```bash
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
az ad sp create-for-rbac \
  --name "ophthalmoai-github-actions" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/ophthalmoai-rg \
  --json-auth
```

---

## 5. Verify

```bash
curl https://<backend-fqdn>/health # {"ok": true, "device": "cpu"}
curl https://<backend-fqdn>/ready # 503 until router_loaded is true
curl https://<backend-fqdn>/ | python3 -m json.tool
```

If `router_loaded` is `false`, the model weights weren't in `models/` at build time —
verify, rebuild, and push again.

---

## 6. Tear Down

```bash
az postgres flexible-server stop --name <pg-server> --resource-group ophthalmoai-rg
az containerapp update --name ophthalmoai-backend --resource-group ophthalmoai-rg --min-replicas 0 --max-replicas 0
az group delete --name ophthalmoai-rg --yes --no-wait # permanent — dump data first if needed
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `router_loaded: false` | Models not baked into image | Confirm weights exist before `docker build` |
| `502` on frontend | Backend not ready yet | Wait 60–90s, check backend logs |
| CORS error in browser | `CORS_ORIGINS` missing/wrong | Update backend env var with the exact frontend URL |
| PostgreSQL connection refused | Firewall rule missing | Add Container Apps outbound IP to PG firewall |
| Alembic step fails in CI | Migration doesn't apply cleanly | Run `alembic upgrade head` locally against a fresh SQLite DB first |
