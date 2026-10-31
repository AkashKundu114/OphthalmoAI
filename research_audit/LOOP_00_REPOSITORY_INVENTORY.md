# LOOP 00 — REPOSITORY INVENTORY & AUDIT BASELINE

**Audit Run Date**: 2026-10-31  
**Repository Root**: `d:\OphthalmoAI`  
**Active Git Branch**: `main`  
**HEAD Commit**: `5387d2c63d2e906cdca3b11b129a90c8b0ce6201`  
**Git Working Tree Status**: Clean (0 uncommitted changes)  
**Execution Environment**: Python 3.14.6, PyTorch 2.16.0.dev20261017+cu134, Torchvision 0.30.0.dev20261030+cu134, NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 13.4 available).

---

## 1. REPOSITORY STRUCTURE & DIRECTORY TOPOLOGY

```
d:\OphthalmoAI\
├── .agents/                    # Multi-agent configurations & skill definitions
├── .github/                    # CI/CD workflows and dependabot configuration
├── alembic/                    # Database migration version scripts
├── backend/                    # FastAPI core service, ML models, triage & security
│   ├── routes/                 # Specialized API routers (clinical, screening, system, auth, telehealth)
│   ├── services/               # Inference service, model loader, explanation, preprocessor
│   ├── config.py               # Central environment configuration & hyperparameter constants
│   ├── conformal.py            # Conformal calibration & Urgency-Stratified Risk Control (US-CRC)
│   ├── fundus_validator.py     # Deterministic biophysical optical admissibility guardrail
│   ├── iqa.py                  # Image quality assessment & compression
│   ├── models.py               # Model architecture definitions
│   ├── onnx_inference.py       # ONNX runtime wrapper & latency benchmarking suite
│   ├── uncertainty.py          # Monte Carlo Dropout & Dirichlet evidential vacuity
│   └── validators.py           # Input payload sanitizers & boundary guards
├── benchmarks/                 # Latency, memory, and throughput profiling suite
├── dataset/                    # Fundus data repository
│   ├── raw/                    # Raw source archives (color_fundus_eye, odir5k, aptos_dr)
│   ├── processed/              # Partitioned cohorts (manifest_v2.csv, train_v2, val_v2, test_v2)
│   ├── metadata/               # Label harmonization and mapping definitions
│   └── logs/                   # Training hardware telemetry and GPU monitoring logs
├── deploy/                     # Docker, Kubernetes, and Hugging Face deployment manifests
├── docs/                       # Architecture diagrams, model cards, audit history
├── frontend/                   # React 18 / Vite / Tailwind clinical web dashboard
├── k8s/                        # Kubernetes deployment, HPA, ingress, and network policies
├── models/                     # PyTorch checkpoints (.pth), ONNX models, and calibration JSONs
├── research/                   # Manuscripts, LaTeX tables, figures, reproducible scripts
│   ├── submissions/            # Submission dossiers (ieee, arxiv, iccece2027)
│   ├── results/                # Canonical evaluation results & raw predictions (.npz)
│   ├── tables/                 # Modular LaTeX table source files
│   └── scripts/                # Manuscript figure synchronization & packaging scripts
├── research_audit/             # Prior forensic audit reports (00 to 17) & issue registers
├── REMEDIATION/                # Prior remediation logs, coverage reports, and baseline audits
├── scripts/                    # 70+ automation, training, benchmarking, and forensic scripts
└── tests/                      # 317 collected test cases (backend unit, integration, forensics)
```

---

## 2. IMPORTANT ENTRY POINTS

### Application & API Entry Points
1. **FastAPI Application Backend**: `backend/main.py`
   - Initializes database, logs, routes, and model inference semaphores.
   - Entry command: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
2. **Unified CLI App Launcher**: `scripts/start_app.py`
   - Boots backend Uvicorn server and Vite frontend concurrently.
3. **Primary Screening Endpoint**: `POST /predict` & `POST /api/predict` in `backend/routes/screening.py`
   - Full pipeline: validation -> domain adaptation -> multi-backbone forward pass -> AW-CRC conformal set -> Grad-CAM -> symptom discordance analysis -> review payload.
4. **Edge ONNX Inference**: `backend/onnx_inference.py`
   - Client/edge low-latency ONNX runtime runner with FP32/INT8 execution.

### Training & Calibration Entry Points
1. **Individual Backbone Training**: `scripts/train_model.py`
   - Supports `--arch [convnext_small|densenet201|efficientnet_v2_m|resnet50|efficientnet_b4]`.
2. **Tri-Backbone Meta-Ensemble Training**: `scripts/train_ensemble.py`
   - Stacking MLP meta-classifier on penultimate feature representations.
3. **Evidential Meta-Classifier**: `scripts/train_evidential_meta.py`
   - Dirichlet prior network for epistemic vacuity estimation.
4. **AW-CRC Conformal Calibration**: `scripts/run_aw_crc_calibration.py`
   - Computes empirical risk quantiles modulated by biophysical optical scores $S(X)$.
5. **Probability Calibration**: `scripts/calibrate_models.py`
   - Temperature scaling (Platt scaling) on holdout validation logits.

### Evaluation & Forensic Entry Points
1. **Canonical Ensemble Evaluation**: `scripts/evaluate_ensemble.py`
   - Evaluates tri-backbone ensemble on sealed test sets (`test_v2.csv`).
2. **Independent Metric Recalculator**: `scripts/reproduce_evaluation.py`
   - Recalculates confusion matrices, AUROC, AUPRC, ECE, Brier score, and bootstrap CIs.
3. **Dataset Forensic Deduplication**: `scripts/forensic_deduplication_v2.py`
   - SHA-256 exact matching, perceptual hashing (dHash), and split verification.
4. **Adversarial Guardrail Stress Benchmark**: `scripts/red_team_guardrail_test.py`
   - Adversarial perturbation, noise injection, and OOD non-fundus evaluation.
5. **External Generalization Benchmark**: `scripts/run_external_validation.py`
   - Evaluates trained models on external cohorts (IDRiD, RIM-ONE DL).

---

## 3. DATASET INVENTORY

### A. Raw Source Cohorts (`dataset/raw/`)
1. **Color Fundus Eye (CFE)**: `dataset/raw/color_fundus_eye/`
   - Origin: Kaggle Retinal Disease Classification archive.
   - Total raw images: ~16,140 fundus photographs.
   - Disease classes: Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertensive Retinopathy / Myopia.
2. **ODIR-5K**: `dataset/raw/odir5k/`
   - Origin: Peking University International Competition on Ocular Disease Intelligent Recognition.
   - Contains paired left/right fundus images with age, sex, and diagnostic keywords.
3. **APTOS 2019 Blindness Detection**: `dataset/raw/aptos_dr/`
   - Origin: Aravind Eye Hospital retinal photography cohort.
   - Diabetic Retinopathy severity grades 0 to 4.

### B. Processed Cohorts & Split Manifests (`dataset/processed/`)
| Manifest File | Sample Count ($N$) | Group Count | Role / Split | Description |
|---|---|---|---|---|
| `manifest_v2.csv` | 13,892 | 13,889 | Master Corpus | Clean, deduplicated primary study cohort |
| `train_v2.csv` | 9,725 | 9,715 | Train (70%) | Primary model optimization partition |
| `validation_v2.csv` | 1,389 | 1,388 | Validation (10%) | Hyperparameter & early-stopping selection |
| `calibration_v2.csv` | 1,389 | 1,388 | Calibration (10%) | Disjoint holdout for Platt & Conformal calibration |
| `test_v2.csv` | 1,389 | 1,388 | Sealed Test (10%) | Independent, unperturbed final benchmark |
| `duplicate_clusters_v2.csv`| 884 clusters | - | Deduplication | Exact SHA-256 & perceptual hash clusters |
| `excluded_samples_v2.csv` | 8,477 | - | Quarantine | 37.86% excluded due to duplicates/conflicts |
| `master_patient_manifest.csv`| Legacy | - | Legacy Split | Historical manifest with heuristic patient clustering |

---

## 4. MODEL INVENTORY (`models/`)

| Checkpoint Name | Architecture | Size | Parameter Count | Reported Accuracy | Status / Notes |
|---|---|---|---|---|---|
| `convnext_small.pth` | ConvNeXt-Small | 197.9 MB | 50.2M | 97.98% | Constituent tri-backbone member |
| `densenet201.pth` | DenseNet-201 | 73.8 MB | 20.0M | 98.13% | Constituent tri-backbone member |
| `efficientnet_v2_m.pth` | EfficientNet-V2-M | 213.0 MB | 54.1M | 98.63% | Constituent tri-backbone member |
| `meta_classifier.pth` | MLP Stacking Classifier | 484.8 MB | ~124.3M total | 98.70% | Stacking network on concatenated backbones |
| `efficientnet_b4.pth` | EfficientNet-B4 | 71.0 MB | 19.3M | ~85.2% | Monolithic legacy / Grad-CAM target |
| `resnet50.pth` | ResNet-50 | 94.4 MB | 25.6M | 96.12% | Baseline comparator |
| `retfound_finetuned.pth` | RETFound (ViT-Large) | 1.21 GB | 304.0M | 98.42% | Foundation model benchmark comparator |
| `mobilenet_v3_small_edge.pth`| MobileNetV3-Small | 6.2 MB | 2.5M | ~91.2% | PyTorch edge training checkpoint |
| `edge_model_fp32.onnx` | MobileNetV3-Small ONNX | 6.1 MB | 2.5M | - | Exported ONNX runtime deployment model |
| `aw_crc_calibration.json` | AW-CRC Quantiles | 1.45 KB | - | - | $q_{emerg}=0.7017, q_{rout}=0.3055$ |
| `calibration.json` | Platt Temperatures | 137 B | - | - | Per-backbone temperature scaling coefficients |

---

## 5. PIPELINES INVENTORY

### A. Training Pipeline
- Preprocessing: Crop to retinal FOV, resize to $384 \times 384$, Ben Graham local color constancy enhancement ($\sigma=10$), ImageNet normalization.
- Augmentations: Random horizontal/vertical flip, random rotation ($\pm 15^\circ$), affine scaling, color jitter.
- Optimization: AdamW ($\text{lr}=10^{-4}$, weight decay $10^{-2}$), CosineAnnealingLR with warmup, CrossEntropyLoss with class inverse-frequency weights.
- Precision: PyTorch AMP (FP16 / BF16 autocast with GradScaler).

### B. Inference Pipeline (Live API `/predict`)
1. **Tier-0 Optical Guardrail**: `validate_fundus_image()` tests dimensions, visual variance, contrast, and Laplacian sharpness. Computes admissibility score $S(X) \in [0.50, 1.0]$.
2. **Domain Adaptation**: `adapt_fundus_domain()` balances color histogram channels.
3. **Multi-Backbone Forward Pass**: DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M generate individual logits.
4. **Soft-Voting Ensemble**: Ensembles calibrated probabilities: $P_{ens} = \frac{1}{3} \sum_m \text{softmax}(z_m / T_m)$.
5. **AW-CRC Conformal Prediction**: Evaluates candidate inclusion threshold $P_{ens}(c) \ge 1 - (q_c / S(X)^\gamma)$.
6. **Explainability**: Generates Grad-CAM visual heatmaps and spatial anatomical landmark descriptions.
7. **Clinical Rules Engine**: Evaluates symptom discordance alerts and routes emergency cases.

### C. Evaluation Pipeline
- Standard metrics: Multiclass Accuracy, Balanced Accuracy, Macro F1, Weighted F1, One-vs-Rest Macro AUROC, OvR AUPRC.
- Calibration metrics: Expected Calibration Error (ECE), Brier Score, Platt temperature reliability diagrams.
- Statistical significance: 2,000-iteration bootstrap resampling, paired McNemar's test for ensemble vs. constituent models.
- Conformal metrics: Empirical marginal coverage, stratum coverage (Emergency $\ge 99\%$, Routine $\ge 95\%$), mean set cardinality, singleton proportion, empty-set rate.

### D. Deployment Pipeline
- Client-side WebAssembly / ONNX Runtime: MobileNetV3-Small running in React dashboard via `frontend/src/edgeInference.js`.
- Docker Containerization: Multi-stage Dockerfiles (`Dockerfile`, `Dockerfile.test`, `frontend/Dockerfile`).
- Kubernetes Orchestration: Full manifests in `k8s/` with Horizontal Pod Autoscaler (HPA), Ingress, and NetworkPolicy.

---

## 6. TEST SUITE INVENTORY

- Total Collected Test Cases: **317 tests** across `tests/` and `tests/backend/`.
- Coverage areas:
  - Analytics & reporting (`test_analytics.py`)
  - Asynchronous screening queue (`test_async_screening.py`)
  - Visual biomarker extraction (`test_biomarker_extractor.py`)
  - Probability calibration (`test_calibration.py`)
  - Conformal prediction & AW-CRC bounds (`test_conformal.py`)
  - Optical guardrails & biophysical boundaries (`test_boundary_conditions.py`, `test_guardrails_and_safety.py`)
  - ONNX runtime inference & numerical parity (`test_onnx_inference.py`, `test_onnx_parity_and_security.py`)
  - Dataset v2 forensics & split leakage (`test_dataset_v2_forensics.py`)
  - Security, privacy, and RBAC (`test_security_and_auth.py`, `test_tenancy_and_rbac.py`)

---

## 7. MISSING OR UNTRACKED ARTIFACTS

1. **U-Net Segmentation Weights**: PASG-GradCAM claims an optic disc/macula U-Net segmentation guide, but no `.pth` weights for U-Net exist in `models/`.
2. **Authentic Longitudinal Clinical EHR Cohort**: The multimodal synergy experiments relied on synthetic data generated via `scripts/generate_patient_biodata.py`; real clinical EHR tables are absent.
3. **Negative Stress Non-Fundus Corpus**: The 2,500 non-fundus medical images (CheXpert, ISIC) are partially hardcoded in benchmark mocks rather than stored as raw image files on disk.

---

## 8. EXISTING CLAIMS REQUIRING INDEPENDENT RE-VERIFICATION

| Claim ID | Claimed Value | Source Location | Audit Priority | Verification Status |
|---|---|---|---|---|
| **CLM-01** | Meta-Ensemble Accuracy: 98.70% (1,371/1,389) | `research/numbers_revised.tex` | CRITICAL | UNVERIFIED — Requires independent re-evaluation |
| **CLM-02** | Macro AUROC: 0.9985 | `research/numbers_revised.tex` | CRITICAL | UNVERIFIED — Requires recalculation from raw probabilities |
| **CLM-03** | Zero Data Leakage Across v2 Splits | `research/manuscript.tex` | CRITICAL | UNVERIFIED — Requires SHA-256 & perceptual hash cross-check |
| **CLM-04** | Patient-Level Independence | `research/manuscript.tex` | HIGH | CONTRADICTED — Source Kaggle CFE dataset lacks clinical patient IDs |
| **CLM-05** | AW-CRC Coverage: Emergency 99.31%, Routine 96.73% | `research/numbers_revised.tex` | HIGH | SUSPECT — Code replaces empty prediction sets with argmax before evaluating coverage |
| **CLM-06** | Meta-Ensemble Superiority over Soft-Voting | `research/numbers_revised.tex` | HIGH | CONTRADICTED — McNemar p-value = 0.6875 indicates no statistically significant difference |
| **CLM-07** | External Generalization (IDRiD / RIM-ONE) | `research/numbers_revised.tex` | HIGH | CONTRADICTED — Specificity collapsed to 16.42% (IDRiD) and 1.05% (RIM-ONE); 100% referral rate |
| **CLM-08** | Edge ONNX Model Parity & Latency (1.57 ms) | `research/numbers_revised.tex` | MEDIUM | UNVERIFIED — Must benchmark on live hardware |
| **CLM-09** | Grad-CAM Adebayo Sanity Check ($r = -0.0818$) | `research/numbers_revised.tex` | MEDIUM | UNVERIFIED — Requires live parameter randomization execution |

---

*Phase 0 Repository Discovery Complete. No production code has been modified. Proceeding to subagent audit integration and forensic verification.*
