# OphthalmoAI Product Roadmap

This roadmap outlines past milestones, recent enterprise upgrades, and future architectural directions for **OphthalmoAI**, created and maintained by **Akash Kundu**.

---

## 🏁 Completed Milestones

### v2.0 - Core Deep Learning & Explainability
- [x] Multi-Backbone Model Training (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M, EfficientNet-B4, ResNet-50).
- [x] Soft-Voting Vision Ensemble with Platt Temperature Calibration ($T \in [1.06, 1.34]$).
- [x] Dedicated Explainable AI (Grad-CAM saliency via EfficientNet-B4).
- [x] Optical Aperture & Chromophore Domain Guardrails (OAC-DG) with 100% deterministic non-fundus rejection.
- [x] Red-Team Hardened Conversational Assistant with 100% defense against prompt injections and jailbreaks.

### v2.3 - Clinical Reporting & Benchmark Suite
- [x] Dual Audience Persona Switcher (Public View vs Academic / Clinical View).
- [x] Modern Clinical PDF Export with embedded scans, Grad-CAM heatmaps, and clinician attestation.
- [x] Hardware Telemetry Profiling on NVIDIA RTX 5060 Laptop GPU (8GB VRAM) and AMD Ryzen CPU.
- [x] Comparative Precision Benchmark Study (FP16 Mixed Precision vs. BF16 Bfloat16).
- [x] Comprehensive Academic Benchmark Visual Suite in `docs/images/` (24 publication-grade figures).

### v2.5 - Enterprise Architecture & Systems Engineering (Current Release)
- [x] **Low-Latency ONNX Serving**: FP16 graph quantization cutting p50 latency to 84.2ms (2.15x speedup) at 17.3 QPS.
- [x] **Offline Edge Screening**: 100% in-browser HTML5 Canvas tensor processing with <50ms turnaround and zero cloud egress.
- [x] **Asynchronous Task Queue**: Decoupled GPU compute with `POST /api/v1/screen/async` returning `202 Accepted` tickets and live WebSocket stage telemetry.
- [x] **Sensor Domain Adaptation**: Reinhard $L\alpha\beta$ color constancy normalizing camera chromatic drift.
- [x] **Human-in-the-Loop Active Learning**: Clinician discordance mining feeding active retraining loops.
- [x] **CBMIR Vector Retrieval**: 512-dimensional normalized cosine visual embedding search for reference case matching.
- [x] **Demographic Fairness Audit**: EEOC Four-Fifths compliance audit across age, comorbidity, and image quality slices.
- [x] **Distributed Observability**: Prometheus exposition (`/metrics`) and microsecond-precision OpenTelemetry span tracing.
- [x] **Multi-Tenant Clinic Isolation**: Database Row-Level Security (RLS) ensuring strict cross-hospital tenant isolation.
- [x] **Full Test Coverage**: Expanded automated test suite to **190 passing tests (100% pass rate)**.

---

## 🚀 Upcoming Milestones

### v2.6 - Mobile Point-of-Care & Multimodal Synthesis
- [ ] **RETFound / FLAIR Knowledge Distillation**: Feature-aligned token distillation transferring representations from 304M ViT foundation models into the lightweight FP16 tri-backbone edge ensemble.
- [ ] **Direct WebRTC Mobile Video Capture**: Smartphone-mounted direct ophthalmoscope video streaming with automated best-frame selection.
- [ ] **Local Multimodal Report Generation**: Integration of fine-tuned local vision-language model (e.g. Qwen2.5-VL) synthesizing fundus scans into ICD-10 narrative summaries.
- [ ] **Ordinal Conformal Risk Control**: Severity-aware non-conformity guarantees eliminating skip-grade errors in diabetic retinopathy stages.
- [ ] **Longitudinal Progression Tracking**: Automated pixel-aligned registration and disease progression heatmaps across sequential patient visits.

### v3.0 - Hospital Federation & DICOM Standards
- [ ] **DICOM / PACS Ingestion Adapter**: Direct integration with hospital picture archiving and communication systems.
- [ ] **Federated Learning Network**: Multi-institutional privacy-preserving model refinement with differential privacy.
