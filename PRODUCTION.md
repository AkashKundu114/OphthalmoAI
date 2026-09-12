# OphthalmoAI Production Deployment Guide

This guide details requirements and procedures for deploying OphthalmoAI in production environments using Docker Compose, Kubernetes, or cloud VMs.

---

## 1. Required Model Artifacts in `models/`
Ensure the following trained weights and calibration artifacts are placed in the `models/` directory before startup:
- `models/densenet201_best.pth`: DenseNet-201 weights.
- `models/convnext_best.pth`: ConvNeXt-Small weights.
- `models/efficientnet_v2_best.pth`: EfficientNet-V2-M weights.
- `models/efficientnet_b4_best.pth`: EfficientNet-B4 weights (dedicated Grad-CAM engine).
- `models/calibration.json`: Platt temperature scaling values.

*Note: If any model weights are unavailable, OphthalmoAI automatically falls back gracefully to single-model mode without service interruption.*

---

## 2. Environment Configuration
Copy `env.example` to `.env` and set production values:
```env
ENVIRONMENT=production
JWT_SECRET_KEY=generate-secure-32-byte-hex-key
DATABASE_URL=postgresql://user:pass@db:5432/ophthalmoai?sslmode=require
CORS_ORIGINS=https://your-domain.com
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash
```

---

## 3. Docker Compose Deployment
```bash
# Build and run backend and frontend containers
docker compose up --build -d

# Execute database migrations
docker compose exec backend alembic upgrade head
```

---

## 4. Health & Monitoring Probes
- Liveness: `GET /health`
- Readiness: `GET /ready`
- Telemetry: `GET /metrics/system`
