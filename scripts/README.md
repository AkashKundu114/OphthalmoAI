# OphthalmoAI: Scripts & Automation Directory

This directory contains the end-to-end data preparation, model training, post-hoc calibration, external clinical validation, and telemetry utilities for **OphthalmoAI**.

All scripts are optimized for execution on modern NVIDIA GPUs (CUDA with Automatic Mixed Precision FP16) and include automatic fallbacks to CPU when running in resource-constrained environments.

---

## Directory Organization

```
scripts/
├── 1. Data Pipeline
│   ├── prepare_dataset.py              # Core PyTorch Dataset & DataLoader utilities
│   ├── prepare_fundus_dataset.py       # Ingests base multicenter corpus (ODIR, APTOS) with Ben Graham crops
│   └── ingest_external_train_data.py   # Downloads and preprocesses external IDRiD training scans
│
├── 2. Training & Domain Adaptation
│   ├── train_model.py                  # Universal backbone trainer (ConvNeXt, DenseNet, EfficientNet, ResNet)
│   ├── train_ensemble.py               # Fuses backbone logits into the soft-voting / meta-classifier ensemble
│   ├── train_evidential_meta.py        # Trains Dirichlet evidential deep learning uncertainty head
│   ├── fine_tune_external_ensemble.py  # Layer-selective domain adaptation on external sensor data
│   └── run_all_trainings.py            # Master training orchestrator and smoke-testing suite
│
├── 3. Evaluation & Publication Suite
│   ├── evaluate_models.py              # Evaluates individual backbones (AUROC, Macro F1, ECE)
│   ├── evaluate_ensemble.py            # Evaluates the tri-backbone ensemble on held-out test split
│   ├── evaluate_external_dataset.py    # Independent external clinical evaluation (IDRiD & RIM-ONE DL)
│   └── generate_external_figures.py    # Generates 300 DPI IEEE/Nature Medicine academic figures
│
├── 4. Calibration & Risk Control
│   ├── calibrate_models.py             # Computes optimal Platt temperature scaling (T*) per backbone
│   └── run_conformal_calib.py          # Calibrates conformal risk prediction sets (alpha=0.01 guarantee)
│
├── 5. System Operations & Telemetry
│   ├── metric_logger.py                # Dual-memory telemetry logger (VRAM, Host RAM, execution time)
│   ├── start_app.py                    # Local developer launcher (FastAPI backend + Vite React frontend)
│   ├── start_public_gpu.py             # Launches cloud GPU tunnel for remote telemedicine screening
│   ├── deploy_to_huggingface.py        # Deploys demo models and weights to Hugging Face Space
│   ├── red_team_guardrail_test.py      # Automated red-team test suite for optical & LLM prompt injection
│   ├── sync_db_schema.py               # Database schema synchronization utility
│   └── launchers/                      # Native shell scripts (.bat, .sh, .ps1)
```

---

## 1. Data Pipeline

### `prepare_dataset.py`
The foundational PyTorch dataset module. Defines `RetinalFundusDataset`, data augmentations (random flips, rotations, color jitter), and the canonical DataLoader factory `prepare_fundus_dataloaders()` across the 6 target classes.

### `prepare_fundus_dataset.py`
Scans raw image directories (`dataset/raw/`), standardizes image dimensions to $384 \times 384$, applies **Ben Graham local Gaussian illumination subtraction** with circular masking, and generates stratified `train.csv`, `val.csv`, and `test.csv` manifests.
```bash
python scripts/prepare_fundus_dataset.py
```

### `ingest_external_train_data.py`
Downloads the 413 official training scans from the **IDRiD** benchmark, processes them with circular Ben Graham cropping, and merges them into `dataset/processed/train_augmented.csv` ($N = 4,786$).
```bash
python scripts/ingest_external_train_data.py
```

---

## 2. Model Training & Fine-Tuning

### `train_model.py`
The universal model training engine. Supports all backbones, multiple floating-point precisions (`fp16`, `bf16`, `fp32`), dynamic learning rate scheduling, and hardware telemetry logging.
```bash
# Train ConvNeXt-Small under FP16 AMP on GPU
python scripts/train_model.py --model convnext_small --precision fp16 --batch-size 16 --epochs 10

# Train DenseNet-201
python scripts/train_model.py --model densenet201 --precision fp16 --batch-size 16 --epochs 10

# Train EfficientNet-V2-M
python scripts/train_model.py --model efficientnet_v2_m --precision fp16 --batch-size 16 --epochs 10
```

### `train_ensemble.py`
Extracts frozen penultimate logits from the three backbones and trains a lightweight fusion meta-classifier with LayerNorm and Dropout to maximize ensemble accuracy.
```bash
python scripts/train_ensemble.py --device cuda
```

### `fine_tune_external_ensemble.py`
Executes layer-selective transfer learning on the augmented dataset (`train_augmented.csv`). Freezes early convolutional feature detectors while fine-tuning top stages and linear classification heads to neutralize camera sensor illumination drift.
```bash
python scripts/fine_tune_external_ensemble.py --epochs 2 --batch-size 16 --lr 5e-5
```

---

## 3. Evaluation & Academic Visualization

### `evaluate_ensemble.py`
Benchmarks the tri-backbone soft-voting ensemble against the strictly held-out empirical test split ($n = 938$). Computes Accuracy, Macro AUROC, Class-specific Sensitivity/Specificity, Macro F1, and Calibrated ECE.
```bash
python scripts/evaluate_ensemble.py --mode soft_voting --device cuda
```

### `evaluate_external_dataset.py`
Runs an independent clinical evaluation on external datasets never seen during training:
- **IDRiD** ($n = 103$, Kowa VX-10 camera, India)
- **RIM-ONE DL** ($n = 447$, Nidek AFC-210 camera, Spain)
Logs full diagnostic metrics, DR severity breakdown, and conformal safety escalation rates to `docs/benchmarks/external_benchmark_results.json`.
```bash
python scripts/evaluate_external_dataset.py --device cuda
```

### `generate_external_figures.py`
Compiles 5 publication-ready academic figures formatted to **IEEE Transactions on Medical Imaging** and **Nature Medicine** standards (300 DPI, 95% Wilson binomial confidence intervals, accessible scientific color palettes):
```bash
python scripts/generate_external_figures.py
```
Outputs saved to `docs/images/`:
- `external_vs_internal_benchmark.png`
- `external_severity_detection_breakdown.png`
- `external_human_review_uncertainty.png`
- `external_fov_sensor_shift.png`
- `external_adaptation_gain.png`

---

## 4. Calibration & Conformal Risk Control

### `calibrate_models.py`
Performs post-hoc Platt temperature scaling ($T^*$) on the validation split via negative log-likelihood minimization, saving tuned temperatures to `models/calibration.json`.
```bash
python scripts/calibrate_models.py --model all --device cuda
```

### `run_conformal_calib.py`
Calibrates non-conformity scores under inductive Conformal Prediction to construct prediction sets $\mathcal{C}(X)$ guaranteeing $\ge 99.0\%$ empirical coverage ($\alpha = 0.01$) for sight-threatening retinal conditions.
```bash
python scripts/run_conformal_calib.py --device cuda
```

---

## 5. Telemetry, Operations & Security

### `start_app.py`
Launches the full-stack OphthalmoAI platform locally (FastAPI backend on port 8000 + React Vite frontend on port 5173).
```bash
python scripts/start_app.py
```

### `red_team_guardrail_test.py`
Comprehensive security and safety audit testing deterministic rejection of non-fundus imagery (natural scenery, pet photos, adversarial noise) and conversational LLM prompt injection defenses.
```bash
pytest scripts/red_team_guardrail_test.py
```

### `deploy_to_huggingface.py`
Syncs the ONNX runtime model and web assets to the official Hugging Face Space repository.
```bash
python scripts/deploy_to_huggingface.py
```
