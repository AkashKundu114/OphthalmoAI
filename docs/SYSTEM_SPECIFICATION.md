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
  - **DenseNet-201**: Dense feature-reuse network ($ layers, 7.0M parameters), capturing fine retinal microvasculature.
  - **ConvNeXt-Small**: Modern pure-convolutional network with  	imes 7$ depthwise convolutions and inverted bottleneck design.
  - **EfficientNet-V2-M**: Progressive-learning neural architecture with Fused-MBConv layers.
  - **EfficientNet-B4**: Dedicated high-resolution backbone for pixel-aligned Grad-CAM saliency heatmaps.
- **Ensemble Strategy**: Soft-voting with Platt temperature scaling. Each backbone\'s logits are divided by its empirical validation temperature $ prior to softmax probability averaging:
  P_{\\text{ensemble}}(y = c \\mid X) = \\frac{1}{M} \\sum_{m=1}^{M} \\text{softmax}\\left(\\frac{z_m(X)}{T_m}\\right)_c
- **Explainability**: Dedicated Grad-CAM heatmap extraction from EfficientNet-B4 top convolutional feature maps, blended with viridis colormap over fundus imagery.

---

## 4. Software Architecture & API
- **Backend**: FastAPI (Python 3.10+), PyTorch (CUDA 12.x / FP16 Mixed Precision), SQLAlchemy (asyncpg + aiosqlite), Alembic migrations, SlowAPI rate limiting, Structlog structured logging.
- **Frontend**: React 19 SPA, Tailwind CSS, Vite 7, Lucide Icons, jsPDF clinical report generator with side-by-side fundus and Grad-CAM embeddings.
- **Audience Mode**:
  - **Public View**: Plain-language explanations, urgency badges, patient action steps, and doctor consultation checklists.
  - **Academic / Clinical View**: Deep statistical metrics (AUROC, Macro F1, ECE, temperature $), raw probability distributions, 1-click BibTeX citation, and tensor JSON export.

---

## 5. Verification & Quality Assurance Checklist
- [x] Preprocessing: Ben Graham circular illumination subtraction at  \\times 384$.
- [x] Multi-backbone weights loaded at backend startup with graceful single-model fallback.
- [x] Platt temperature scaling calibrated across all models ( \\in [1.06, 1.34]$).
- [x] Empirical evaluation on held-out test split (=938$): **85.18% Accuracy**, **0.9805 Macro AUROC**, **0.0644 ECE**.
- [x] Zero console warnings and passing production builds (
pm run build).
