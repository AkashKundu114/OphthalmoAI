# OphthalmoAI: Point-of-Care Retinal Disease Screening Platform

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![React 19](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Accuracy 85.18%](https://img.shields.io/badge/Test%20Accuracy-85.18%25-brightgreen)
![AUROC 0.9805](https://img.shields.io/badge/Macro%20AUROC-0.9805-blue)

**OphthalmoAI** is an AI-powered retinal disease screening and clinical decision-support platform. It utilizes a **Calibrated Tri-Backbone Soft-Voting Ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M)** with **Platt Temperature Scaling**, paired with a dedicated **EfficientNet-B4 Explainable AI (Grad-CAM)** engine and an interactive **dual-mode interface (Public vs Academic/Clinical)**.

---

> **MEDICAL DISCLAIMER**: OphthalmoAI is designed strictly for research, educational, and screening-aid purposes. It is not an FDA-cleared or CE-marked medical device. All findings must be confirmed by a licensed ophthalmologist or optometrist.

---

## Key Features

1. **Calibrated Tri-Backbone Vision Ensemble**:
   - Concurrently executes **DenseNet-201**, **ConvNeXt-Small**, and **EfficientNet-V2-M**.
   - Applies post-hoc **Platt Temperature Scaling** ($T \in [1.06, 1.34]$) to eliminate neural overconfidence.
   - Achieves **85.18% empirical test accuracy** and **0.9805 Macro AUROC** on 938 held-out clinical fundus images.
2. **Pixel-Level Interpretability (Grad-CAM)**:
   - Dedicated **EfficientNet-B4** backbone generates high-resolution saliency maps overlaid directly on fundus imagery.
3. **Dual Audience Persona Modes**:
   - **Public View**: Patient-friendly explanations, triage urgency indicators, personalized action plans, and "Questions for Your Doctor".
   - **Academic / Clinical View**: In-depth statistical metrics (AUROC, Macro F1, calibration temperature $T$, ECE), multi-class probability distribution, 1-click BibTeX citation copy, and raw tensor JSON export.
4. **Modern Clinical PDF Generation**:
   - Clean, professional vector PDF export containing side-by-side color fundus scans, Grad-CAM overlays, ICD-10/SNOMED-CT codes, confidence bars, and clinician attestation blocks.
5. **AI Clinical Assistant**:
   - Natural language conversational helper grounded in visual findings and physical biomarkers, powered by Google Gemini 2.0 Flash with local Ollama fallback.

---

## Empirical Benchmark Performance

<p align="center">
  <img src="docs/images/benchmark_accuracy_comparison.png" alt="Benchmark Accuracy Comparison" width="90%" />
</p>

| Architecture / Model | Test Accuracy | Macro AUROC | Macro F1 | Calibration $T$ | Calibrated ECE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Tri-Backbone Ensemble (SOTA)** | **85.18%** | **0.9805** | **0.8292** | **Ensemble** | **0.0644** |
| DenseNet-201 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0519 |
| ConvNeXt-Small | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0614 |
| EfficientNet-V2-M | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0268 |
| EfficientNet-B4 (Grad-CAM Engine) | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0582 |
| ResNet-50 (Baseline) | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0412 |

<p align="center">
  <img src="docs/images/multiclass_roc_curves.png" alt="ROC Curves" width="48%" />
  <img src="docs/images/calibration_temperatures_chart.png" alt="Calibration Temperatures" width="48%" />
</p>

---

## Target Retinal Conditions (6 Classes)

| Diagnostic Class | Clinical Urgency | Target Retinal Pathology | ICD-10 Code | SNOMED-CT |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | None | Healthy retina, clear optic disc, crisp foveal reflex | `Z01.00` | `17621005` |
| **Diabetic Retinopathy** | **Urgent** | Microaneurysms, blot hemorrhages, hard exudates | `E11.319` | `4855003` |
| **Glaucoma** | **Urgent** | Cup-to-disc ratio enlargement, neuroretinal rim loss | `H40.9` | `23986001` |
| **Cataract** | Elective | Optical scattering and vascular attenuation on fundus | `H25.9` | `193570009` |
| **Age-related Macular Degeneration** | **Urgent** | Macular drusen, geographic atrophy, CNV | `H35.30` | `267718000` |
| **Hypertensive Retinopathy / Myopia** | **Urgent** | Arteriolar narrowing, AV nicking, staphyloma | `H35.00` | `39934008` |

---

## Quick Start (Local Setup)

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/AkashKundu114/OphthalmoAI.git
cd OphthalmoAI

# Create and activate virtual environment
python -m venv venv
# On Windows: .\venv\Scripts\activate
# On Linux/macOS: source venv/bin/activate

# Install PyTorch and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt

# Start backend server
python backend/main.py
```
> Backend API serves at `http://localhost:8000` (Swagger UI at `/docs`).

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
> Frontend SPA serves at `http://localhost:5173`.

---

## Documentation Suite

- **[System Specification](docs/SYSTEM_SPECIFICATION.md)**: Technical architecture, pipeline stages, and QA checklist.
- **[Clinical Evaluation & Safety](docs/clinical/CLINICAL_EVALUATION_AND_SAFETY.md)**: Intended use, clinical risk controls, and validation protocols.
- **[Performance & Telemetry](docs/PERFORMANCE_METRICS.md)**: Comprehensive empirical metrics, ROC curves, calibration charts, and GPU profiling.
- **[Technical White Paper](docs/OphthalmoAI_Technical_White_Paper.md)**: Engineering methodology, ensemble formulations, and explainability.
- **[Production Guide](PRODUCTION.md)**: Deployment guidelines for Docker, Kubernetes, and cloud environments.

---

## License
Apache License 2.0. Copyright (c) 2026 Akash Kundu. See `LICENSE` for details.
