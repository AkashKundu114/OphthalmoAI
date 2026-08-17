<div align="center">
  
# 👁️ OphthalmoAI 
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

> ⚕️ **MEDICAL DISCLAIMER:** OphthalmoAI is provided strictly for **research, educational, and informational purposes**. It is **not** an FDA-cleared or CE-marked medical device. It does **not** constitute professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider or ophthalmologist. See the `LICENSE` file for full liability details.

---

## 🌟 Key Innovations
1. **Meta-Classifier Ensemble Architecture:** Uses a state-of-the-art meta-classifier to intelligently combine predictions from three massive Vision models (ConvNeXt-Small, DenseNet-201, and EfficientNet-V2). This allows for near-perfect point-of-care clinical diagnostics by capturing both fine-grained vascular anomalies and robust structural features.
2. **Comprehensive 12-Disease Screening:** Supports diagnosis across 12 distinct conditions: `['cataract', 'conjunctivitis', 'ptosis', 'normal', 'pterygium', 'uveitis', 'blepharitis', 'chalazion', 'keratitis', 'stye', 'subconjunctival_hemorrhage', 'jaundice']`.
3. **Hardware Optimized for 8GB VRAM:** Employs Automatic Mixed Precision (AMP), gradient scaling, and aggressive VRAM garbage collection to train three SOTA models natively on a single NVIDIA RTX 5060 Laptop GPU.
4. **LLM Structural Guardrails:** Integrates Gemini 2.0 Flash into the point-of-care interface, restricted strictly to answering questions and contextualizing the deterministic vision pipeline results, preventing clinical hallucination.
5. **Comprehensive Dockerized Test Suite:** Automated frontend (Vitest + JSDOM) and backend (Pytest + FastAPI TestClient) integration testing, running entirely in isolated ephemeral Docker containers to ensure zero environment pollution.

---

## 🩺 Detectable Conditions

| Condition | Anatomical Group | Clinical Urgency |
|-----------|-----------------|------------------|
| **Cataract** | Anterior Segment | Elective |
| **Uveitis** | Anterior Segment | 🔴 **Urgent** |
| **Conjunctivitis** | Ocular Surface | Non-urgent |
| **Jaundice** *(Scleral Icterus)* | Ocular Surface | 🔴 **Emergency** |
| **Pterygium** | Ocular Surface | Elective |
| **Ptosis** | Adnexal/Oculoplastic | Non-urgent |
| **Blepharitis** | Adnexal/Oculoplastic | Non-urgent |
| **Chalazion** | Adnexal/Oculoplastic | Non-urgent |
| **Stye** | Adnexal/Oculoplastic | Non-urgent |
| **Keratitis** | Anterior Segment | 🔴 **Emergency** |
| **Subconjunctival Hemorrhage**| Ocular Surface | Non-urgent |
| **Normal** | All Groups | None |

---

## ⚡ Quick Start (Docker / GPU)

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

## 💻 Bare-Metal Setup (Development)

If you wish to build the FastAPI backend and React frontend locally:

### 1. Backend API
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

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

## 🏗️ Project Architecture

```
OphthalmoAI/
├── backend/
│   ├── main.py, auth.py, db_async.py, routes_admin.py
│   ├── security.py, clinical_codes.py, calibration.py
│   ├── audit.py, logging_config.py, requirements.txt
├── frontend/
│   ├── src/App.jsx, ChatBox.jsx, index.css
│   ├── nginx.conf, vite.config.js
├── scripts/              # Training, Telemetry, and Evaluation tools
├── alembic/              # Database schema migrations
├── docs/                 # PRD, Technical Specs, Clinical Guidelines
└── docker-compose.yml    # Orchestration
```

---

## 📊 Performance Benchmarks & Telemetry

OphthalmoAI features an integrated hardware telemetry pipeline designed to profile training runs across different compute architectures. Below are the verified benchmark results for the **EfficientNet-B4** and **ResNet50** models:

| Environment | Model | Time per Epoch | Peak VRAM | Final Accuracy | CPU Usage |
|-------------|-------|----------------|-----------|----------------|-----------|
| **Ryzen 9 8940HX (Bare-Metal)** | ResNet50 | `451.7s` | N/A | 81.61% | ~60% |
| **RTX 5060 (Bare-Metal)** | ResNet50 | `55.0s` | 1.77 GB | 92.96% | ~63% |
| **RTX 5060 (Bare-Metal)** | EfficientNet-V2-S | `51.9s` | 2.74 GB | 91.30% | ~45% |
| **RTX 5060 (Bare-Metal)** | EfficientNet-B4 | `38.4s` | 2.05 GB | 96.27% | ~10% |
| **RTX 5060 (Docker/NGC 26.07)** | EfficientNet-B4 | `23.5s` | 2.05 GB | **98.61%** | ~8% |

> **Note**: For detailed telemetry charts spanning Model Convergence, GPU VRAM Consumption, and Thermal Performance, please see the `docs/images/` directory or read `docs/PERFORMANCE_METRICS.md`.

---

## 📡 Core API Reference

| Method | Path | Auth Required | Description |
|--------|------|---------------|-------------|
| `GET` | `/health` / `/ready` | ❌ | Liveness and readiness probes |
| `POST` | `/predict` | ❌ | Run an eye scan inference (rate-limited) |
| `POST` | `/chat` | ❌ | AI Doctor chat (Gemini Flash free tier / Ollama) |
| `POST` | `/auth/register` / `/token` | ❌ | Account creation and login |
| `POST` | `/scans/{id}/override` | 🔒 Clinician/Admin | Record a second opinion |
| `GET` | `/admin/audit-logs` | 🔒 Admin | Query the administrative audit trail |

*(For the complete schema, refer to `docs/technical/BACKEND_SCHEMA.md`)*

---

## 📚 Deep Dive Documentation

If you want to understand the clinical design, telemetry tracking, or backend architecture, check out our comprehensive documentation suite:

- **[OphthalmoAI Technical White Paper](docs/OphthalmoAI_Technical_White_Paper.md)** - Explains the Monolithic Vision Pipeline and LLM Structural Guardrails.
- **[Performance Metrics](docs/PERFORMANCE_METRICS.md)** - Details on the PyTorch NGC Docker performance and VRAM optimizations.
- **[Implementation Plan](docs/planning/IMPLEMENTATION_PLAN.md)** - Project checklist and phase tracking.
- **[Technical Issues & Resolutions](docs/technical/ISSUES.md)** - Deep dive into resolved architecture and dependency conflicts.
- **[Clinical Safety Specs](docs/clinical/CLINICAL_SAFETY.md)** - Intended use and safety mechanisms.

---

## 📜 License & Copyright

**Apache License 2.0** — Copyright © 2026 Akash Kundu.

Please carefully review the `LICENSE` file for the strict **Medical and Clinical Liability Disclaimer**. This software cannot be used for clinical decision-making without independent validation.
