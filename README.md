# OphthalmoAI — Eye Disease Screening

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
OphthalmoAI is a comprehensive full-stack ophthalmology platform that **enhances diagnostic screening capabilities across 7 visible eye conditions** through the deployment of a **MobileNetV3 routing layer and specialized EfficientNet-B4 models**, providing **interpretable Grad-CAM heatmaps and AI-driven clinical cross-checks**.

> ⚕ **Medical Disclaimer:** OphthalmoAI is a research and educational screening tool. It is **not a substitute** for professional medical diagnosis, advice, or treatment. Always consult a qualified ophthalmologist.

---

## 🚀 Key Impact & Accomplishments (XYZ Format)

- **Accelerated initial screening triage** by providing **near-instant preliminary assessments** through the implementation of an **automated multi-model inference pipeline (MobileNetV3 router + EfficientNet-B4 experts)**.
- **Improved clinical trust and explainability** by **generating visual evidence of AI decisions** using **Grad-CAM heatmaps and rigorous uncertainty calibration**.
- **Streamlined patient understanding** as measured by **accessible follow-up interactions** by integrating a **Gemini 2.0 Flash-powered medical chat assistant with strict safety guardrails**.
- **Ensured enterprise-grade security and reliability** by **establishing an robust audit trail and scalable state management** using **FastAPI, AsyncSQLAlchemy, and role-based JWT authentication**.

---

## 🩺 Detectable Conditions

| Condition | Anatomical Group | Urgency |
|-----------|-----------------|---------|
| **Cataract** | Anterior Segment | Elective |
| **Uveitis** | Anterior Segment | 🔴 Urgent |
| **Conjunctivitis** | Ocular Surface | Non-urgent |
| **Jaundice** *(Scleral Icterus)* | Ocular Surface | 🔴 Emergency |
| **Pterygium** | Ocular Surface | Elective |
| **Eyelid Conditions** | Adnexal/Oculoplastic | Non-urgent |
| **Normal** | All Groups | None |

---

## 🏗️ Project Architecture

```
OphthalmoAI/
├── backend/
│   ├── main.py, auth.py, db.py, db_async.py, routes_admin.py
│   ├── security.py, validators.py, medical_data.py, clinical_codes.py
│   ├── calibration.py, uncertainty.py, iqa.py, model_registry.py
│   ├── audit.py, storage.py, logging_config.py, requirements.txt, Dockerfile
├── frontend/
│   ├── src/App.jsx, ChatBox.jsx, cropImage.js, index.css
│   ├── nginx.conf, vite.config.js, Dockerfile
├── scripts/              # Training, calibration, and evaluation pipelines
├── alembic/              # Database migrations
├── tests/backend/        # Comprehensive Pytest suites
├── k8s/                  # Kubernetes manifests & Azure CD workflow
├── docs/                 # Planning, design, technical, and clinical specs
└── docker-compose.yml    # Local multi-container orchestration
```

---

## ⚡ Quick Start — Docker

To run the application locally using Docker:

```bash
git clone <this-repo>
cd OphthalmoAI
cp env.example .env           # Configure GEMINI_API_KEY or OLLAMA_URL
docker compose up --build -d
docker compose exec backend alembic upgrade head
```

Navigate to **http://localhost:8080**. *(Note: Model weights are not included by default — mount `models/` or run the training scripts).*

---

## 💻 Local Development Setup

To build and run the backend locally (requires a compatible Python environment with PyTorch):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install PyTorch (CUDA 12.4 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt

# Run migrations and start the server
alembic upgrade head
python backend/main.py    # API served at http://localhost:8000, Swagger at /docs
```

To run the frontend:
```bash
cd frontend
npm install
npm run dev               # App served at http://localhost:5173
```

---

## 📡 API Reference

| Method | Path | Auth Required | Description |
|--------|------|---------------|-------------|
| `GET` | `/health` / `/ready` | ❌ | Liveness and readiness probes |
| `GET` | `/conditions` | ❌ | Retrieve metadata for all 7 conditions |
| `POST` | `/predict` | ❌ | Run an eye scan inference (rate-limited) |
| `POST` | `/chat` | ❌ | AI Doctor chat (Gemini Flash free tier / Ollama) |
| `POST` | `/auth/register` / `/token` | ❌ | Account creation and login |
| `GET` | `/auth/me` | 🔒 Any | Retrieve current user profile |
| `POST` | `/scans/{id}/override` | 🔒 Clinician/Admin | Record a second opinion |
| `GET` | `/admin/audit-logs` | 🔒 Admin | Query the administrative audit trail |
| `POST` | `/admin/model-registry/activate` | 🔒 Admin | Promote a new model version |

*(For the complete schema, refer to `docs/technical/BACKEND_SCHEMA.md`)*

---

## 🧪 Testing & Validation

The backend is backed by an extensive test suite ensuring clinical safety and operational stability.

```bash
# Run the backend tests with coverage
pytest tests/backend/ -v --cov=backend
```

---

## 📚 Documentation Directory

| Document | Description |
|----------|-------------|
| `WIRING.md` | Change logs and recent verification notes |
| `ROADMAP.md` | Upgrade roadmap and planned features |
| `PRODUCTION.md` | Deployment guide (Docker / Azure / Kubernetes) |
| `docs/technical/AZURE_DEPLOY.md` | Step-by-step Azure hosting instructions |
| `docs/technical/BACKEND_SCHEMA.md` | API schema and data models |
| `docs/technical/SECURITY_AUDIT.md` | Security findings and mitigation strategies |
| `docs/clinical/*` | Intended use, safety mechanisms, validation report template |
| `docs/planning/*` | PRD, TRD, application flow, implementation plans |

---

## 📜 License

Apache License 2.0 — see `LICENSE`. Copyright © 2026 Akash Kundu.
