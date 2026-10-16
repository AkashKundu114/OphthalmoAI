# OphthalmoAI: Deployment & Operations Guide

> **Research Prototype Operational Instructions**: For research laboratories, academic benchmark environments, and clinical pilot demonstrations.

---

## 1. System Requirements

### Hardware Targets
- **GPU Inference (Recommended)**: NVIDIA GeForce RTX 3060+ / RTX 4060+ / RTX 5060 Laptop or Datacenter A100/H100 with >= 6GB VRAM.
- **CPU Inference**: Multi-core x86_64 / ARM64 CPU with >= 8GB RAM (set `FORCE_CPU=true`).
- **Edge Inference**: Standard modern web browser (Chrome, Edge, Firefox, Safari) with WebAssembly and WebGL/WebGPU support.

### Software Dependencies
- **Python**: 3.11 or 3.12 (CPython)
- **Node.js**: >= 18.0.0 (Node 24 LTS verified)
- **CUDA**: 12.1+ / 12.4+ (when utilizing GPU acceleration)
- **Operating Systems**: Windows 11 / 10, Ubuntu Linux 22.04 / 24.04 LTS

---

## 2. Environment Configuration

Copy `env.example` to `.env` in the repository root:

```bash
cp env.example .env
```

### Essential Configuration Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Set to `production` for strict security controls. |
| `HOST` | `0.0.0.0` | Bind IP for Uvicorn server. |
| `PORT` | `8000` | Port for FastAPI service. |
| `JWT_SECRET_KEY` | *(Must be set in prod)* | Cryptographic random hex key for session tokens. |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed CORS origins (`*` blocked in prod). |
| `FORCE_CPU` | `false` | Set to `true` to force CPU inference even if CUDA is available. |
| `MAX_CONCURRENT_INFERENCES`| `4` | Concurrency throttle semaphore for GPU VRAM protection. |
| `MAX_FILE_SIZE_BYTES` | `20971520` (20MB) | Maximum allowed image upload size. |
| `ENABLE_IQA` | `true` | Enables Image Quality Assessment prior to inference. |
| `ENABLE_UNCERTAINTY` | `true` | Enables MC-dropout epistemic uncertainty estimation. |
| `PERSIST_SCANS` | `true` | Persists scan records and audit logs to SQLite/PostgreSQL. |
| `GEMINI_API_KEY` | `""` | Optional Google Gemini key for multilingual AI doctor chat. |
| `SARVAM_API_KEY` | `""` | Optional Sarvam AI key for Indic speech translation and TTS. |
| `OLLAMA_URL` | `""` | Optional local LLM endpoint (e.g. `http://localhost:11434`). |

---

## 3. Local Installation & Launch

### 3.1 Backend Setup
```powershell
# Create and activate virtual environment
python -m venv .venv_gpu
.\.venv_gpu\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start backend service
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3.2 Frontend Setup
```powershell
cd frontend
npm install
npm run dev
```
The user interface is accessible at `http://localhost:5173`.

---

## 4. Production Build & Serving

### 4.1 Build Static Frontend
```powershell
cd frontend
npm run build
```
Build output is emitted to `frontend/dist/`.

### 4.2 Production Docker Deployment
```bash
# Build and run GPU-accelerated container
docker compose up --build -d
```

### 4.3 Kubernetes Deployment
Kubernetes manifests are provided in `k8s/`:
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## 5. Offline Edge Deployment (Zero Network Egress)

For remote primary health centres (PHC), outreach camps, and rural clinics without internet connectivity:
1. Copy the built `frontend/dist/` bundle onto any USB thumb drive or static web server.
2. Ensure `frontend/dist/models/edge_model.onnx` is present in the distribution bundle.
3. Open `index.html` in any modern browser.
4. Edge mode executes entirely in the client's WebAssembly sandbox with **zero HTTP requests** sent across the network.

---

## 6. Health Checks & Observability

- **Liveness Probe**: `GET /health` or `GET /api/health` $\to$ Returns `{"ok": true, "device": "cuda"}`
- **Readiness Probe**: `GET /ready` or `GET /api/ready` $\to$ Returns `{"ok": true, "model_loaded": true}` (HTTP 503 if models unavailable)
- **Prometheus Metrics**: `GET /metrics` $\to$ Scrapes real-time request counts, latency quantiles, GPU VRAM usage, and active tenants.
