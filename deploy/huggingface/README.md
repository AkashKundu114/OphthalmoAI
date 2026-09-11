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

Point-of-care eye disease screening API with an evidential multi-backbone ensemble (ConvNeXt-Small + DenseNet-201 + EfficientNet-V2-M), conformal risk control, Grad-CAM saliency extraction, and multimodal clinical reporting.

### API Endpoints
- `GET /health` - Health check
- `POST /predict` - Disease screening & Grad-CAM analysis
- `POST /chat` - Clinical reasoning assistant
- `GET /conditions` - List detectable eye conditions
