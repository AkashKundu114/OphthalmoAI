---
title: OphthalmoAI API
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# OphthalmoAI Backend Inference Engine

Point-of-care retinal disease screening API powered by a Calibrated Tri-Backbone Soft Ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M), Optical Aperture & Chromophore Domain Guardrails (OAC-DG), dedicated Grad-CAM saliency heatmaps (EfficientNet-B4), and a Red-Team hardened AI clinical assistant (Gemini 2.0 Flash).

### Key Highlights
- **85.18% Test Accuracy & 0.9805 Macro AUROC** on held-out clinical fundus test split ($n = 938$).
- **Deterministic Non-Fundus Rejection**: Optical aperture and chromophore backscatter checks block non-medical imagery with HTTP 422.
- **Explainable Saliency**: High-resolution Grad-CAM overlays highlighting pathognomonic microvascular lesions.

### API Endpoints
- `GET /health` - Liveness health check
- `GET /ready` - Readiness check for models and database
- `POST /predict` - Retinal fundus disease screening, domain verification & Grad-CAM analysis
- `POST /api/v1/screen/async` - Asynchronous task queue yielding 202 job tickets
- `POST /chat` - Grounded clinical reasoning assistant with red-team guardrails
- `POST /api/v1/cases/similar` - CBMIR vector search for historical reference cases
- `GET /conditions` - List detectable retinal conditions (6 diagnostic classes)
- `GET /metrics` - Prometheus metrics exposition
- `GET /metrics/system` - Live hardware telemetry (VRAM, RAM, throughput)
- `GET /api/v1/traces/recent` - OpenTelemetry distributed span traces
