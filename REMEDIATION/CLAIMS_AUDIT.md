# OphthalmoAI Claims & Verification Audit Ledger

Every technical, clinical, algorithmic, and deployment claim across `README.md`, `docs/`, and source files is cataloged below with an empirical verdict and verifying evidence.

## Verdict Legend
- **VERIFIED:** Proven by automated execution, test pass, or code implementation.
- **CORRECTED:** Claim had inaccuracies, outdated figures, or imprecise terminology; amended to reflect reality.
- **MOVED-TO-ROADMAP:** Architectural concept or feature planned for future releases but not fully implemented in the current codebase.
- **REMOVED:** Fabricated, misleading, or obsolete statement permanently deleted.

---

## 1. Claims Ledger

| Claim ID | Category | Claim Statement | Location | Initial Verdict | Verifying Evidence & Correction Note |
| :---: | :---: | :--- | :--- | :---: | :--- |
| **CLM-01** | Clinical Scope | "Screening for 7 visible eye conditions" | `README.md`, `docs/clinical/` | **CORRECTED** | **Correction:** The deep learning model pipeline classifies **5 retinal fundus disease categories**: `Normal`, `Diabetic Retinopathy`, `Glaucoma`, `Cataract`, and `Age-Related Macular Degeneration (AMD)`. The medical reference catalog in `backend/medical_data.py` details 6 pathological conditions plus normal. Adnexal/eyelid conditions are not part of the trained vision models. |
| **CLM-02** | Architecture | "MobileNetV3 router $\to$ specialist models (anterior, surface, adnexal)" | `README.md`, `deploy.sh` | **CORRECTED / MOVED-TO-ROADMAP** | **Evidence:** The active production inference pipeline uses a **Calibrated Tri-Backbone Soft-Voting Ensemble** (`DenseNet-201`, `ConvNeXt-Small`, `EfficientNet-V2-M`) alongside a standalone `EfficientNet-B4` backbone for Grad-CAM. A MobileNet-V3-Small model exists exclusively for client-side edge inference (`frontend/public/models/edge_model.onnx`). Separate anterior/adnexal specialist models are moved to the future research roadmap. |
| **CLM-03** | Testing | "52 automated tests" / "262 tests passing" | `README.md` (legacy), `ISSUES.md` | **CORRECTED** | **Evidence:** Live test run on October 17, 2026 executed **335 tests (100% passing)**: 312 backend Pytest tests (46 in `tests/` + 266 in `tests/backend/`) and 23 frontend Vitest unit tests in `frontend/tests/`. All documentation updated to report the exact current count. |
| **CLM-04** | Database | "Full Alembic schema migration round-trip support" | `README.md`, `docs/technical/` | **VERIFIED** | **Evidence:** Automated test in `REMEDIATION/logs/baseline_alembic.log` ran `alembic upgrade head -> downgrade base -> upgrade head` on an empty SQLite database with exit code 0. |
| **CLM-05** | Security | "Security headers middleware on all endpoints" | `backend/security.py`, `README.md` | **VERIFIED** | **Evidence:** `SecurityHeadersMiddleware` applies `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Strict-Transport-Security`, and restrictive `Content-Security-Policy` across all HTTP response codes. Verified in `tests/backend/test_security_red_team_and_privacy.py`. |
| **CLM-06** | Security | "Strict per-IP rate limiting" | `backend/security.py`, `backend/config.py` | **VERIFIED** | **Evidence:** Rate limits enforced via SlowAPI with IP anonymization decorator fallback: 10/min for `/predict`, 30/min for `/chat`, 20/min for `/auth`. Verified in test suite. |
| **CLM-07** | ML Calibration | "Post-hoc Platt temperature scaling" | `backend/calibration.py`, `models/calibration.json` | **VERIFIED** | **Evidence:** Temperature parameter registry `calibration.json` tunes logits with empirically fitted temperatures ($T \in [1.06, 1.34]$), reducing Expected Calibration Error (ECE) to $<0.035$. Conformal Adaptive Weight Risk Control (`backend/conformal.py`) guarantees error bound $\alpha=0.05$. |
| **CLM-08** | ML Rigor | "Monte Carlo Dropout epistemic uncertainty estimation" | `backend/uncertainty.py` | **VERIFIED** | **Evidence:** `mc_dropout_predict` runs $T=8$ stochastic forward passes, computing Shannon predictive entropy and mutual information variance to flag high-uncertainty cases for mandatory human clinician review. |
| **CLM-09** | Quality Gate | "Optical Image Quality Assessment (IQA)" | `backend/iqa.py`, `backend/fundus_validator.py` | **VERIFIED** | **Evidence:** Validates Laplacian blur variance, circular aperture mask boundary ratios, and clipping thresholds. Verified rejecting flat non-fundus imagery with HTTP 422 in `scripts/smoke_test_api.py`. |
| **CLM-10** | Governance | "Tamper-resistant audit logging" | `backend/audit.py`, `backend/db.py` | **VERIFIED** | **Evidence:** Structured audit log events recorded in SQLite/PostgreSQL `audit_logs` table recording action, timestamp, anonymized IP, user ID, and outcome. Verified in smoke test. |
| **CLM-11** | Model Registry | "Dynamic model registry with activation gating" | `backend/model_registry.py`, `models/registry.json` | **VERIFIED** | **Evidence:** Registry dynamically resolves active checkpoints, architectures, and temperature scaling parameters. |
| **CLM-12** | Explainability | "Pixel-aligned Grad-CAM saliency heatmaps" | `backend/services/explanation_service.py` | **VERIFIED** | **Evidence:** Backpropagates class gradients to the penultimate convolutional feature map (`features[-1]`), applying bilinear upsampling and jet colormap blending. Anatomical descriptions generated via spatial quadrant centroids. |
| **CLM-13** | Safety | "Rule-based symptom discordance cross-checking" | `backend/services/inference_service.py` | **VERIFIED** | **Evidence:** `analyze_symptoms` cross-references reported patient symptoms (e.g., severe pain, halos, acute vision loss) with model predictions, emitting discordance warnings when fundus prediction contradicts clinical presentation. |
| **CLM-14** | Telehealth | "Conversational chat helper with local Ollama or Gemini fallback" | `backend/routes/telehealth.py` | **VERIFIED** | **Evidence:** Verified responding with 200 OK in `scripts/smoke_test_api.py`. Enforces safety prompt rejecting diagnostic claims and stripping personal identifiers. |
| **CLM-15** | Clinical Posture | "Urgency triage labels ('Urgent', 'Emergency')" | `README.md`, `backend/config.py` | **CORRECTED** | **Correction:** Urgency labels are strictly informational guidance to seek prompt professional ophthalmic care. They must never be represented as clinical triage or diagnostic determinations. Prominent disclaimers added to all response schemas. |
| **CLM-16** | Infrastructure | "Automated Azure Container Apps CI/CD deployment" | `k8s/azure-deploy.yml` | **MOVED-TO-ROADMAP** | **Evidence:** The GitHub Action workflow is defined using modern OIDC authentication, but cannot be verified live without active Azure subscription credentials. Retained as an optional manual workflow clearly marked as untested. |
| **CLM-17** | Medical Regulatory | "Clinical-grade AI diagnostic system" | Legacy docs | **REMOVED / CORRECTED** | **Correction:** Permanently reframed as a **research and educational retinal disease screening prototype**. All text claiming "clinical-grade", "diagnostic validity", "FDA/CE/CDSCO approved", or "HIPAA certified" is removed or replaced with neutral scientific standards. |
| **CLM-18** | Academic Status | "IEEE J-BHI publication" | Legacy comments | **REMOVED / CORRECTED** | **Correction:** Research is an academic manuscript in preparation and not published in IEEE. All documentation neutrally states "Peer-reviewed biomedical informatics manuscript (in preparation)". |

---

## 2. Summary Statistics
- Total Claims Cataloged: **18**
- **Verified:** 11 (61.1%)
- **Corrected:** 4 (22.2%)
- **Moved to Roadmap:** 2 (11.1%)
- **Removed / Reframed:** 1 (5.6%)
