<div align="center">
  
# OphthalmoAI 
### **Point-of-Care Eye Disease Screening Platform**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

OphthalmoAI is a comprehensive full-stack ophthalmology platform that **enhances diagnostic screening capabilities across 12 visible eye conditions**. It employs a novel **Monolithic EfficientNet-B4 Architecture** to deliver high-accuracy inference, backed by **interpretable Grad-CAM heatmaps** and an **AI-driven clinical conversational assistant**.

</div>

---

> **MEDICAL DISCLAIMER:** OphthalmoAI is provided strictly for **research, educational, and informational purposes**. It is **not** an FDA-cleared or CE-marked medical device. It does **not** constitute professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider or ophthalmologist. See the `LICENSE` file for full liability details.

---

## Architecture & Design Principles
1. **Meta-Classifier Ensemble Architecture:** Uses a state-of-the-art meta-classifier to intelligently combine predictions from three vision models (ConvNeXt-Small, DenseNet-201, and EfficientNet-V2). This allows for robust clinical diagnostic capabilities by capturing both fine-grained vascular anomalies and robust structural features.
2. **Comprehensive 12-Disease Screening:** Supports diagnosis across 12 distinct conditions: `['cataract', 'conjunctivitis', 'ptosis', 'normal', 'pterygium', 'uveitis', 'blepharitis', 'chalazion', 'keratitis', 'stye', 'subconjunctival_hemorrhage', 'jaundice']`.
3. **Hardware Optimized for 8GB VRAM:** Employs Automatic Mixed Precision (AMP), gradient scaling, and aggressive VRAM garbage collection to train three state-of-the-art models natively on a single NVIDIA RTX 5060 Laptop GPU.
4. **LLM Structural Guardrails:** Integrates Gemini 2.0 Flash into the point-of-care interface, restricted strictly to answering questions and contextualizing the deterministic vision pipeline results, preventing clinical hallucination.
5. **Comprehensive Dockerized Test Suite:** Automated frontend (Vitest + JSDOM) and backend (Pytest + FastAPI TestClient) integration testing, running entirely in isolated ephemeral Docker containers to ensure zero environment pollution.

---

## Detectable Conditions

| Condition | Anatomical Group | Clinical Urgency |
|-----------|-----------------|------------------|
| **Cataract** | Anterior Segment | Elective |
| **Uveitis** | Anterior Segment | **Urgent** |
| **Conjunctivitis** | Ocular Surface | Non-urgent |
| **Jaundice** *(Scleral Icterus)* | Ocular Surface | **Emergency** |
| **Pterygium** | Ocular Surface | Elective |
| **Ptosis** | Adnexal/Oculoplastic | Non-urgent |
| **Blepharitis** | Adnexal/Oculoplastic | Non-urgent |
| **Chalazion** | Adnexal/Oculoplastic | Non-urgent |
| **Stye** | Adnexal/Oculoplastic | Non-urgent |
| **Keratitis** | Anterior Segment | **Emergency** |
| **Subconjunctival Hemorrhage**| Ocular Surface | Non-urgent |
| **Normal** | All Groups | None |

---

## Getting Started (Docker / GPU)

The easiest and most performant way to run OphthalmoAI is via our pre-configured Docker pipeline.

```bash
# 1. Clone the repository
git clone https://github.com/AkashKundu114/Eye-Disease-AI-Diagnosis.git
cd Eye-Disease-AI-Diagnosis

# 2. Configure Environment (Set GEMINI_API_KEY)
cp env.example .env           

# 3. Spin up the cluster
docker compose up --build -d

# 4. Initialize Database Schema
docker compose exec backend alembic upgrade head
```

Navigate to **[http://localhost:8080](http://localhost:8080)** to access the platform.

---

## Local Development Environment (Development)

If you wish to build the FastAPI backend and React frontend locally:

### 1. Backend API
```bash
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

# Install PyTorch (CUDA 12.4 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt

# Run migrations and start the server
alembic upgrade head
python backend/main.py    
```
> The API will be served at `http://localhost:8000`, with interactive Swagger Docs at `/docs`.

### 2. Frontend SPA
```bash
cd frontend
npm install
npm run dev               
```
> The React app will be served at `http://localhost:5173`.

---

## Project Architecture

```
OphthalmoAI/
 backend/
    main.py, auth.py, db_async.py, routes_admin.py
    security.py, clinical_codes.py, calibration.py
    audit.py, logging_config.py, requirements.txt
 frontend/
    src/App.jsx, ChatBox.jsx, index.css
    nginx.conf, vite.config.js
 scripts/ # Training, Telemetry, and Evaluation tools
 alembic/ # Database schema migrations
 docs/ # PRD, Technical Specs, Clinical Guidelines
 docker-compose.yml # Orchestration
```

---

## Performance Benchmarks & Telemetry

OphthalmoAI features an integrated hardware telemetry pipeline designed to profile training runs across different compute architectures and precision formats. Below are the verified benchmark results:

| Model Architecture | Precision | Batch Size | Time per Epoch | Peak VRAM | Final Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Meta-Classifier Ensemble (state-of-the-art)** | **FP16** | **32** | **20.62s** | **0.96 GB** | **99.72%** |
| **Meta-Classifier Ensemble** | **BF16** | **32** | **22.51s** | **1.21 GB** | **99.67%** |
| **ConvNeXt-Small** | **FP16** | **32** | **19.32s** | **3.64 GB** | **99.32%** |
| **DenseNet-201** | **BF16** | **32** | **25.00s** | **3.45 GB** | **99.49%** |
| **EfficientNet-V2-M** | **FP16** | **32** | **24.91s** | **4.62 GB** | **99.21%** |
| *EfficientNet-B4 (Docker Single)* | *FP16* | *16* | *33.68s* | *2.00 GB* | *98.61%* |
| *ResNet50 (Bare-Metal GPU)* | *FP32* | *16* | *52.09s* | *1.73 GB* | *92.96%* |
| *ResNet50 (CPU Baseline)* | *FP32* | *16* | *460.79s* | *0.00 GB* | *81.61%* |

<p align="center">
  <img src="docs/images/architecture_evolution_summary.png" alt="Architecture Evolution Summary" width="95%" />
</p>

### Detailed Comparative Analysis
* **Base Monolith Models (ConvNeXt vs DenseNet vs EfficientNet):**
  <p align="center">
    <img src="docs/images/base_monolith_models_comparison.png" alt="Base Monolith Comparison" width="95%" />
  </p>
* **Meta-Classifier Ensemble Optimization (BS=4 vs BS=32 Scaling):**
  <p align="center">
    <img src="docs/images/meta_classifier_comparison.png" alt="Meta Classifier Comparison" width="90%" />
  </p>

> **Note**: For detailed telemetry charts spanning Model Convergence, GPU VRAM Consumption, and Thermal Performance, please read [`docs/PERFORMANCE_METRICS.md`](docs/PERFORMANCE_METRICS.md).

---

## Core API Reference

| Method | Path | Auth Required | Description |
|--------|------|---------------|-------------|
| `GET` | `/health` / `/ready` | | Liveness and readiness probes |
| `POST` | `/predict` | | Run an eye scan inference (rate-limited) |
| `POST` | `/chat` | | AI Doctor chat (Gemini Flash free tier / Ollama) |
| `POST` | `/auth/register` / `/token` | | Account creation and login |
| `POST` | `/scans/{id}/override` | Clinician/Admin | Record a second opinion |
| `GET` | `/admin/audit-logs` | Admin | Query the administrative audit trail |

*(For the complete schema, refer to `docs/technical/BACKEND_SCHEMA.md`)*

---

## Technical Documentation

If you want to understand the clinical design, telemetry tracking, or backend architecture, check out our comprehensive documentation suite:

- **[OphthalmoAI Technical White Paper](docs/OphthalmoAI_Technical_White_Paper.md)** - Explains the Monolithic Vision Pipeline and LLM Structural Guardrails.
- **[Performance Metrics](docs/PERFORMANCE_METRICS.md)** - Details on the PyTorch NGC Docker performance and VRAM optimizations.
- **[Implementation Plan](docs/planning/IMPLEMENTATION_PLAN.md)** - Project checklist and phase tracking.
- **[Technical Issues & Resolutions](docs/technical/ISSUES.md)** - Deep dive into resolved architecture and dependency conflicts.
- **[Clinical Safety Specs](docs/clinical/CLINICAL_SAFETY.md)** - Intended use and safety mechanisms.

---

## License & Copyright

**Apache License 2.0** - Copyright © 2026 Akash Kundu.

Please carefully review the `LICENSE` file for the strict **Medical and Clinical Liability Disclaimer**. This software cannot be used for clinical decision-making without independent validation.
