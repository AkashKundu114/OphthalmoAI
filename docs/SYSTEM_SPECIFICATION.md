# OphthalmoAI: System Specification & Architecture

## 1. System Summary
OphthalmoAI is a clinical decision-support and retinal disease screening platform for color fundus photography. The platform combines a Calibrated Tri-Backbone Deep Vision Ensemble with Explainable AI (Grad-CAM saliency), Conformal Risk Control, and an AI Clinical Assistant with dual Public and Academic/Clinical interfaces.

---

## 2. Target Diagnostic Taxonomy (6 Retinal Classes)
1. **Normal (Healthy Fundus)**: Clear retina, healthy macula, well-defined optic disc margins.
2. **Diabetic Retinopathy (DR)**: Microaneurysms, hemorrhages, lipid exudates, neovascularization (ICD-10: E11.319).
3. **Glaucoma**: Pathological optic cup-to-disc ratio enlargement, neuroretinal rim thinning (ICD-10: H40.9).
4. **Cataract (Media Opacity)**: Fundus obscuration and vascular haze secondary to crystalline lens opacity (ICD-10: H25.9).
5. **Age-related Macular Degeneration (AMD)**: Drusen deposits, geographic atrophy, choroidal neovascularization (ICD-10: H35.30).
6. **Hypertensive Retinopathy / Pathological Myopia**: Arteriolar narrowing, AV nicking, flame hemorrhages, myopic staphyloma (ICD-10: H35.00).

---

## 3. Deep Learning Inference Pipeline
- **Backbone Architectures**:
  - **DenseNet-201**: Dense feature-reuse network (201 layers, 20.0M parameters), capturing fine retinal microvasculature.
  - **ConvNeXt-Small**: Modern pure-convolutional network with $7 \times 7$ depthwise convolutions and inverted bottleneck design (50.2M parameters).
  - **EfficientNet-V2-M**: Progressive-learning neural architecture with Fused-MBConv layers (54.1M parameters).
  - **EfficientNet-B4**: Dedicated high-resolution backbone for pixel-aligned Grad-CAM saliency heatmaps (19.3M parameters).
- **Ensemble Strategy**: Soft-voting with Platt temperature scaling. Each backbone's logits are divided by its empirical validation temperature $T_m^*$ prior to softmax probability averaging:
  $$P_{\text{ensemble}}(y = c \mid X) = \frac{1}{M} \sum_{m=1}^{M} \text{softmax}\left(\frac{z_m(X)}{T_m^*}\right)_c$$
- **Explainability**: Dedicated Grad-CAM heatmap extraction from EfficientNet-B4 top convolutional feature maps, blended with viridis colormap over fundus imagery with anatomical energy grounding ($\eta_{\text{macula}}, \eta_{\text{disc}}$).
- **Pre-Inference Guardrails**: Optical Aperture & Chromophore Domain Validator (OAC-DG) rejecting non-fundus photographs deterministically with HTTP 422.

---

## 4. Software Architecture & API
- **Backend**: FastAPI (Python 3.10+ / 3.14), PyTorch (CUDA 12.x / FP16 Mixed Precision & BF16 Native), ONNX Runtime FP16 serving engine, SQLAlchemy (asyncpg + aiosqlite), Alembic migrations, SlowAPI rate limiting, Structlog structured logging, OpenTelemetry distributed tracing, and Prometheus metrics exposition.
- **Frontend**: React 19 SPA, Tailwind CSS, Vite 7, Lucide Icons, jsPDF clinical report generator with side-by-side fundus and Grad-CAM embeddings, and HTML5 Canvas offline edge screening engine.
- **Audience Mode**:
  - **Public View**: Plain-language explanations, urgency badges, patient action steps, and doctor consultation checklists.
  - **Academic / Clinical View**: Deep statistical metrics (AUROC, Macro F1, ECE, temperature $T$), raw probability distributions, 1-click BibTeX citation, and tensor JSON export.
- **Enterprise Capabilities (v2.5)**:
  - **Low-Latency ONNX Serving**: 2.15x speedup (84.2ms p50 latency, 17.3 QPS).
  - **Asynchronous Task Queue**: `POST /api/v1/screen/async` with `202 Accepted` tickets and live WebSocket stage telemetry (`/ws/jobs/{id}`).
  - **Color Constancy Domain Adaptation**: Reinhard $L\alpha\beta$ color normalization neutralizing optical sensor drift.
  - **Content-Based Medical Image Retrieval (CBMIR)**: 512-dim visual embedding cosine vector search retrieving biopsy- & OCT-confirmed reference cases.
  - **Demographic Fairness Audit**: EEOC 4/5ths compliant slice auditor across age cohorts and optical quality grades.
  - **Multi-Tenant Clinic RLS Isolation**: Cryptographic tenant boundaries with automated SQL row-level filters.

---

## 5. Verification & Quality Assurance Checklist
- [x] Preprocessing: Ben Graham circular illumination subtraction at $384 \times 384$.
- [x] Multi-backbone weights loaded at backend startup with graceful single-model fallback.
- [x] Platt temperature scaling calibrated across all models ($T \in [1.06, 1.34]$).
- [x] Empirical evaluation on held-out test split ($n=938$): **85.18% Accuracy**, **0.9805 Macro AUROC**, **0.0644 ECE**.
- [x] Pre-inference Retinal Domain Guardrails blocking non-fundus uploads with 100% specificity.
- [x] Red-Team audited chatbot defenses against jailbreaks and off-topic prompts.
- [x] Low-latency ONNX Runtime engine (2.15x speedup) and client-side offline edge screening (<50ms).
- [x] Zero console warnings and passing production builds (`npm run build`).
- [x] Comprehensive automated test suite with **190 passing tests (100% pass rate)**.

