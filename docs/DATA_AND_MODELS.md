# OphthalmoAI: Data & Model Provenance

> **Academic Research Notice**: This documentation provides technical reproducibility details for an academic research manuscript in preparation. All models and datasets described herein are for scientific research and benchmark evaluation.

---

## 1. Dataset Provenance & Patient-Level Splitting

To ensure clinical reproducibility and prevent data leakage across image crops or fellow eyes of the same individual, OphthalmoAI enforces strict patient-level disjoint partitioning.

### 1.1 Development & Benchmark Corpus
- **Primary Source**: Multi-center color fundus retinal photography collection spanning 6 canonical diagnostic classes:
  1. `Normal` (Unremarkable posterior pole, crisp optic rim, healthy macula)
  2. `Diabetic Retinopathy` (Microaneurysms, hemorrhages, hard exudates, cotton-wool spots)
  3. `Glaucoma` (Optic cup-to-disc ratio enlargement, neuroretinal rim thinning, focal notching)
  4. `Cataract` (Diffuse media opacification, lenticular scatter obscuring posterior pole)
  5. `Age-related Macular Degeneration` (Confluent soft drusen, subretinal pigment changes, geographic atrophy)
  6. `Hypertensive Retinopathy / Pathological Myopia` (Arteriolar narrowing, AV nicking, flame hemorrhages, tilted disc, chorioretinal atrophy)
- **Data Partitions**:
  - `dataset/processed/train.csv`: Training split for feature extraction and fine-tuning.
  - `dataset/processed/test.csv`: Independent held-out test split reserved strictly for final statistical evaluation.
  - Data splitting enforced via `scripts/split_patient_level.py` with stratification across diagnostic classes.

### 1.2 Independent External Clinical Benchmarks
To evaluate domain shift robustness across geographic populations, camera sensors, and ethnicity cohorts, OphthalmoAI was evaluated against two completely unseen external clinical datasets:
1. **IDRiD (Indian Diabetic Retinopathy Image Dataset)**:
   - **Origin**: Clinical ophthalmology clinic in Nanded, Maharashtra, India.
   - **Optical Sensor**: Kowa VX-10 fundus camera (50° field-of-view).
   - **Evaluation Purpose**: Assesses generalizability across rural Indian demographic cohorts and distinct sensor color profiles.
2. **RIM-ONE DL (Retinal Image Database for Optic Nerve Evaluation)**:
   - **Origin**: University Hospital of the Canary Islands and Hospital Universitario Miguel Servet, Spain.
   - **Optical Sensor**: Nidek AFC-210 non-mydriatic fundus camera (45° field-of-view).
   - **Evaluation Purpose**: Assesses glaucoma optic disc cup-to-disc segmentation and classification under European clinical conditions.

---

## 2. Model Zoo & Checkpoint Specifications

All model checkpoints are stored in `models/` with strict verification:

| Model Architecture | File Name | Checkpoint Size | Target Resolution | Primary Role |
| :--- | :--- | :--- | :--- | :--- |
| **DenseNet-201** | `densenet201.pth` | 73.8 MB | 384 × 384 | Dense feature re-use backbone (Ensemble) |
| **ConvNeXt-Small** | `convnext_small.pth` | 198.0 MB | 384 × 384 | Modern 7×7 depthwise conv inductive bias (Ensemble) |
| **EfficientNet-V2-M** | `efficientnet_v2_m.pth` | 213.0 MB | 384 × 384 | Progressive learning & fused MBConv (Ensemble) |
| **EfficientNet-B4** | `efficientnet_b4.pth` | 71.0 MB | 384 × 384 | Primary Grad-CAM backbone & single-model fallback |
| **MobileNetV3-Small** | `mobilenet_v3_small_edge.pth` | 6.2 MB | 224 × 224 | PyTorch source for edge export |
| **Edge ONNX (FP32)** | `edge_model_fp32.onnx` | 6.1 MB | 224 × 224 | Intermediate ONNX graph |
| **Edge ONNX (INT8)** | `frontend/public/models/edge_model.onnx` | 1.8 MB | 224 × 224 | In-browser zero-egress edge inference (WASM) |

---

## 3. Probability Calibration & Conformal Parameters

The calibrated ensemble uses empirical parameter files stored in `models/`:

- **`models/calibration.json`**:
  Stores learned Platt scaling temperature parameters ($T > 0$) fitted via negative log-likelihood minimization on validation logits:
  ```json
  {
    "densenet201": 0.8841,
    "convnext_small": 0.9125,
    "efficientnet_v2_m": 0.8654,
    "efficientnet_b4": 0.8912,
    "monolith": 0.8912
  }
  ```
- **`models/conformal_calibration.json`**:
  Stores non-conformity quantiles $\hat{q}_{1-\alpha}$ and admissibility weight coefficients for the 3-tier Conformal Triage Policy:
  - High Confidence (Single-label prediction set $\to$ Routine Care)
  - Differential Uncertainty (Multi-label prediction set $\to$ Priority Clinician Review)
  - Severe OOD / High Epistemic Vacuity (Expanded prediction set $\to$ Immediate Human Specialist Over-read)

---

## 4. Checkpoint Integrity & Security Enforcements

1. **`weights_only=True` Enforced**:
   All `torch.load` calls explicitly enforce `weights_only=True` across `backend/services/model_loader.py` and evaluation scripts. This prevents arbitrary code execution vulnerabilities from pickled Python byte streams.
2. **Missing Checkpoint Behavior**:
   If a checkpoint is missing, the system emits structured telemetry logs (`model_loader.checkpoint_missing`) and falls back gracefully to the single-model baseline or returns HTTP 503 Service Unavailable without crashing the process.
3. **Licensing & Usage**:
   - Model weights and scripts are distributed strictly under academic open-research terms.
   - Redistribution must retain medical disclaimer notices: *Not approved by FDA/CE for autonomous clinical diagnosis.*
