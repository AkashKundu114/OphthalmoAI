# OphthalmoAI Production Deployment Guide

This guide details requirements and operational procedures for deploying **OphthalmoAI** in production enterprise environments using Vercel, Hugging Face Spaces, Docker Compose, Kubernetes, or cloud VMs.

**Maintainer:** Akash Kundu ([@AkashKundu114](https://github.com/AkashKundu114))

---

## 🌐 Live Production Hosts & Mirrors

- **Primary Web Application (Vercel)**: [https://ophthalmo-ai-mu.vercel.app/](https://ophthalmo-ai-mu.vercel.app/)
- **Hugging Face Space**: [https://huggingface.co/spaces/AkashKundu114/ophthalmoai-demo](https://huggingface.co/spaces/AkashKundu114/ophthalmoai-demo)
- **Hugging Face Static App Mirror**: [https://akashkundu114-ophthalmoai-demo.static.hf.space](https://akashkundu114-ophthalmoai-demo.static.hf.space)

---

## 1. Required Model Artifacts in `models/`

Ensure the following trained weights and calibration artifacts are placed in the `models/` directory before startup:

- `models/densenet201_best.pth`: DenseNet-201 weights.
- `models/convnext_best.pth`: ConvNeXt-Small weights.
- `models/efficientnet_v2_best.pth`: EfficientNet-V2-M weights.
- `models/efficientnet_b4_best.pth`: EfficientNet-B4 weights (dedicated Grad-CAM engine).
- `models/calibration.json`: Platt temperature scaling values ($T \in [1.06, 1.34]$).
- `models/ensemble.onnx` *(Optional / Recommended)*: Compiled FP16 ONNX model for 2.15x lower latency.

*Note: If any model weights are unavailable, OphthalmoAI automatically falls back gracefully to synthetic testing mode without service interruption.*

---

## 2. Environment Configuration

Copy `env.example` to `.env` and configure production variables:

```env
# Runtime
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Security & Auth
JWT_SECRET_KEY=generate-secure-32-byte-hex-key
ACCESS_TOKEN_EXPIRE_MINUTES=60
CORS_ORIGINS=https://ophthalmo-ai-mu.vercel.app,https://your-hospital-domain.com

# Database (PostgreSQL with SSL)
DATABASE_URL=postgresql+asyncpg://user:password@db-host:5432/ophthalmoai?ssl=require

# Tenancy & RLS Isolation
TENANT_ENFORCEMENT=true

# AI Assistant Grounding
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash

# Observability
ENABLE_OPENTELEMETRY=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

---

## 3. High-Performance Serving Options

### Option A: Compiled ONNX Runtime FP16 Serving
For high-concurrency production deployments, execute with ONNX acceleration:
```bash
python backend/onnx_inference.py --benchmark
python backend/main.py --use-onnx
```
- **Throughput:** ~17.3 QPS (up from 5.4 QPS PyTorch).
- **Latency (p50):** 84.2 ms.

### Option B: Asynchronous WebSocket Screening Queue
For batch or multi-user clinical queues, utilize the decoupled async queue:
- Ingestion endpoint: `POST /api/v1/screen/async` -> Returns `202 Accepted` job ticket.
- Real-time stream: `WebSocket /ws/jobs/{ticket_id}` -> Streams 5 processing stages directly to the client.

---

## 4. Docker Compose Deployment

```bash
# Build and run backend and frontend containers
docker compose up --build -d

# Execute database migrations
docker compose exec backend alembic upgrade head
```

---

## 5. Kubernetes (K8s) Deployment

Production Helm charts and manifests are provided in [`k8s/`](k8s/README.md):

```bash
# Apply ingress, deployment, and service manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## 6. Health, Telemetry & Monitoring Probes

- **Liveness Probe:** `GET /health` (Returns `200 OK`)
- **Readiness Probe:** `GET /ready` (Verifies database and model weight accessibility)
- **Prometheus Metrics:** `GET /metrics` (Standard Prometheus exposition for request counts, latency quantiles, and GPU memory)
- **Hardware Telemetry:** `GET /metrics/system` (GPU VRAM, CPU load, and host memory)
- **Distributed Traces:** `GET /api/v1/traces/recent` (OpenTelemetry span waterfalls)
