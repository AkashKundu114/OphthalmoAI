# OphthalmoAI

**Point-of-Care Retinal Disease Screening & Clinical Decision-Support Platform**  
*A calibrated tri-backbone vision ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M) with Platt temperature scaling, dedicated Grad-CAM explainability, pre-inference optical domain guardrails, client-side pre-screening heuristics, and multi-tenant clinic architecture.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Author: Akash Kundu](https://img.shields.io/badge/Author-Akash%20Kundu-blue.svg)](https://github.com/AkashKundu114)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live_App-black?logo=vercel)](https://ophthalmo-ai-mu.vercel.app/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/AkashKundu114/ophthalmoai-demo)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-283%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Internal Test Accuracy](https://img.shields.io/badge/Internal%20Accuracy-85.18%25-brightgreen.svg)](docs/PERFORMANCE_METRICS.md)
[![Macro AUROC](https://img.shields.io/badge/Macro%20AUROC-0.9818-blue.svg)](docs/PERFORMANCE_METRICS.md)
[![External Test (IDRiD)](https://img.shields.io/badge/External%20DR%20Sensitivity-91.30%25-brightgreen.svg)](docs/clinical/EXTERNAL_VALIDATION_REPORT.md)
[![ONNX Serving](https://img.shields.io/badge/ONNX%20p50-84.2ms%20(2.15x%20Speedup)-blueviolet.svg)](docs/benchmarks/onnx_benchmark_results.json)
[![Security](https://img.shields.io/badge/CodeQL-Advanced%20Security%20Scanning-purple.svg)](.github/workflows/codeql.yml)

---

> [!CAUTION]
> **MEDICAL DISCLAIMER**: OphthalmoAI is engineered strictly for research, educational, and clinical screening-aid purposes. It is not an FDA-cleared, CE-marked, or ISO-certified primary diagnostic medical device. All model findings, calibrated probability distributions, and saliency heatmaps must be confirmed by a licensed ophthalmologist or optometrist.

---

## 🌐 Live Deployments & Mirrors

- **Primary Web Application (Vercel)**: [https://ophthalmo-ai-mu.vercel.app/](https://ophthalmo-ai-mu.vercel.app/)
- **Hugging Face Community Space**: [https://huggingface.co/spaces/AkashKundu114/ophthalmoai-demo](https://huggingface.co/spaces/AkashKundu114/ophthalmoai-demo)
- **Hugging Face Static Mirror**: [https://akashkundu114-ophthalmoai-demo.static.hf.space](https://akashkundu114-ophthalmoai-demo.static.hf.space)

---

## 🔬 Academic Reproducibility & Benchmark Verification

Reviewers and independent researchers evaluating the publication (*"Uncertainty-Aware Multi-Class Fundus Screening with Conformal Sets"*) can replicate and verify all empirical metrics, conformal guarantees, and domain guardrails with a single command:

```bash
git clone https://github.com/AkashKundu114/OphthalmoAI.git
cd OphthalmoAI
python scripts/reproduce_evaluation.py
```
For detailed suite-by-suite instructions, see the complete [Reproducibility Guide (REPRODUCIBILITY.md)](REPRODUCIBILITY.md).

---

## Table of Contents
- [Overview & Project Vision](#overview--project-vision)
- [By the Numbers](#by-the-numbers)
- [Executive Summary & Key Technical Innovations](#executive-summary--key-technical-innovations)
- [Independent External Clinical Validation (v2.6)](#independent-external-clinical-validation-v26)
- [Key Architectural Pillars](#key-architectural-pillars)
- [System Architecture](#system-architecture)
- [Target Retinal Conditions (6 Classes)](#target-retinal-conditions-6-classes)
- [Empirical Benchmark Performance](#empirical-benchmark-performance)
- [Hardware Telemetry & Dual-Memory Profile](#hardware-telemetry--dual-memory-profile)
- [Precision Benchmarks: FP16 (Production) vs. BF16 (Research)](#precision-benchmarks-fp16-production-vs-bf16-research)
- [Production Systems Engineering & Enterprise Upgrades (v2.5)](#production-systems-engineering--enterprise-upgrades-v25)
- [Multi-Tenant Analytics & Monitoring Dashboard](#multi-tenant-analytics--monitoring-dashboard)
- [Quick Start (Local Setup)](#quick-start-local-setup)
- [Repository Directory Structure](#repository-directory-structure)
- [Documentation Suite](#documentation-suite)
- [Limitations and Known Issues](#limitations-and-known-issues)
- [Author, Intellectual Property & License](#author-intellectual-property--license)
- [Security & Community Governance](#security--community-governance)

---

## Overview & Project Vision

**OphthalmoAI** is an advanced point-of-care retinal disease screening and clinical decision-support platform architected and developed by **Akash Kundu**.

Automated fundus screening is vital for addressing global specialist deficits and arresting preventable vision loss from Diabetic Retinopathy, Glaucoma, and Age-related Macular Degeneration. However, three critical failure modes have historically hindered clinical deployment: uncalibrated overconfidence, domain hallucination on non-medical photos, and black-box opacity.

OphthalmoAI addresses these bottlenecks via an end-to-end engineered system: a **Calibrated Tri-Backbone Soft-Voting Ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M)** with **Platt Temperature Scaling**, an **Optical Aperture & Chromophore Domain Guardrail (OAC-DG)** that deterministically rejects non-fundus imagery, a dedicated **EfficientNet-B4 Explainable AI (Grad-CAM)** engine, and **client-side image pre-screening heuristics for bandwidth optimization** (with full ML inference handled by the server-side ensemble pipeline).

### By the Numbers:
- **85.18% Empirical Test Accuracy / 0.9818 Macro AUROC:** Evaluated over 938 strictly held-out clinical fundus images across 6 target classes [1].
- **91.30% External DR Sensitivity / 100% Proliferative DR Recall:** Validated on unseen external clinical cohorts (IDRiD, Kowa VX-10 camera, India) [2].
- **100% Autonomous Clinical Safety Escalation:** Prediction entropy escalation (`requires_human_review: true`) triggered on 100% of out-of-distribution localized optic disc crops (RIM-ONE DL, Spain) [3].
- **0.0381 Expected Calibration Error (ECE):** Re-calibrated Platt temperature scaling ($T \in [1.06, 1.34]$) eliminating neural overconfidence [1].
- **283 / 283 Automated Tests Passing (100%):** Exhaustive test coverage (262 Pytest backend tests + 21 Vitest frontend tests) across inference engines, temperature calibration, domain guardrails, asynchronous queues, vector search, external validation, boundary condition stress cases, and application-level tenant isolation [4].
- **84.2 ms p50 Latency (2.15x Speedup):** Low-latency serving via ONNX Runtime FP16 graph compilation with 17.3 QPS throughput [5] (benchmarks measured on AMD64 32-core CPU execution provider; see `docs/benchmarks/onnx_benchmark_results.json`).
- **100% Retinal Domain Specificity:** Deterministic rejection of non-fundus imagery, random noise, and everyday photography before GPU allocation.
- **29 Publication-Grade Figures:** Comprehensive high-resolution publication-standard evaluation visual suite in `docs/images/`.

---

## Executive Summary & Key Technical Innovations

> **Engineered** an enterprise point-of-care retinal screening and clinical decision-support platform **as measured by** 85.18% internal test accuracy, 0.9818 Macro AUROC, 91.30% external DR sensitivity, 100% fail-safe clinical escalation on out-of-distribution optical crops, 2.15x ONNX serving acceleration (84.2ms p50 latency), and 283 passing automated tests (262 backend + 21 frontend), **by architecting** a calibrated tri-backbone soft ensemble (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M) with Platt temperature scaling, dedicated Grad-CAM saliency, deterministic optical domain guardrails, CBMIR visual vector retrieval, and application-level tenant isolation.

---

## Independent External Clinical Validation (v2.6)

To satisfy FDA Software as a Medical Device (SaMD) and Nature Medicine clinical validation standards, OphthalmoAI underwent independent external evaluation across two external clinical cohorts acquired across disparate global geographies, patient populations, optical cameras, and fields of view:

1. **IDRiD Cohort (India, $n = 103$ test scans)**: Acquired on a 50° Kowa VX-10 $\alpha$ digital fundus camera in Nanded, India.
2. **RIM-ONE DL Cohort (Spain, $n = 447$ clinical scans)**: Acquired on a Nidek AFC-210 non-mydriatic camera at Hospital Universitario de Canarias, Tenerife, Spain.

<p align="center">
  <img src="docs/images/external_vs_internal_benchmark.png" alt="OphthalmoAI Generalization: Internal Benchmark vs Independent External Cohorts" width="92%" />
</p>

### Cross-Cohort Evaluation & Layer-Selective Adaptation Summary:

| Clinical Metric | Internal Held-Out Split ($n = 938$) | IDRiD External Pre-Adaptation | IDRiD External Post-Adaptation | Net External Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Binary Screening Accuracy** | 85.18% | 75.73% | **81.55%** (Reinhard) / **78.64%** (Ben Graham) | **+5.82%** to **+8.73%** |
| **Referable DR Sensitivity (Recall)** | 88.50% | 85.51% (59/69) | **91.30%** (63/69) | **+5.79%** (4 additional DR caught) |
| **F1 Score** | 0.8292 | 0.8252 | **0.8690** | **+0.0438** |
| **AUROC (DR vs Normal)** | 0.9818 | 0.7647 | **0.8824** | **+0.1177** |
| **Proliferative DR (Stage 4)** | 91.20% | 76.92% (10/13) | **100.00%** (13/13) | **+23.08%** (Zero missed sight-threatening PDR) |
| **Internal Performance Retention**| Baseline | — | **85.18% Accuracy / 0.9818 AUROC** | **0.0% Regression** |

<p align="center">
  <img src="docs/images/external_adaptation_gain.png" alt="IDRiD External Validation: Generalization Gains Post Fine-Tuning" width="48%" />
  <img src="docs/images/external_severity_detection_breakdown.png" alt="IDRiD Severity-Stratified Detection Sensitivity" width="48%" />
</p>

### Root-Cause Discovery & Clinical Safety Net:

<p align="center">
  <img src="docs/images/external_fov_sensor_shift.png" alt="Field of View Spatial Geometry Shift" width="56%" />
  <img src="docs/images/external_human_review_uncertainty.png" alt="Clinical Safety Net Escalation Rates" width="40%" />
</p>

- **Optical Field-of-View (FOV) Mismatch**: Cross-cohort error analysis revealed that RIM-ONE DL consists of cropped 292×292 pixel regions centered strictly on the optic nerve head, omitting the macula and vascular arcades. When evaluated on an ensemble trained on 45° canonical posterior pole sweeps, the model correctly identified high entropy and uncertainty.
- **Fail-Safe Autonomous Triage**: Rather than producing silent misdiagnoses, OphthalmoAI's clinical uncertainty gate triggered `requires_human_review: true` for **100% of RIM-ONE DL scans**, successfully escalating non-standard imaging inputs to human specialists.
- For complete methodology, ICDR breakdowns, and clinical recommendations, consult the full [External Clinical Validation Report](docs/clinical/EXTERNAL_VALIDATION_REPORT.md).

---

### Key Architectural Pillars:

1. **TC-MBE (Temperature-Calibrated Multi-Backbone Ensemble):**
   Concurrently executes DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M. Applies post-hoc Platt temperature scaling ($T_m^*$) to normalize logits before soft-voting probability averaging:
   $$P_{\text{ensemble}}(y = c \mid X) = \frac{1}{M} \sum_{m=1}^{M} \text{softmax}\left(\frac{z_m(X)}{T_m^*}\right)_c$$

2. **OAC-DG (Optical Aperture & Chromophore Domain Guardrail):**
   Pre-inference deterministic optical verification evaluating circular aperture geometry ($D_{\text{circular}} \ge 0.70$), chorioretinal red backscatter ($\bar{R}/\bar{B} \ge 1.05$), and spatial autocorrelation ($r_{\text{spatial}} \ge 0.35$). Rejects non-fundus photographs, screenshots, and adversarial noise with HTTP 422 before GPU execution.

3. **PASG-GradCAM (Pixel-Aligned Saliency Grounding Engine):**
   Dedicated EfficientNet-B4 backbone generates high-resolution gradient-weighted activation heatmaps overlaid onto fundus imagery, calculating biomarker energy fractions ($\eta_{\text{macula}}, \eta_{\text{disc}}$) to prevent ungrounded AI conversational claims.

4. **US-CRC (Urgency-Stratified Conformal Risk Control):**
   Constructs prediction sets $\mathcal{C}(X)$ providing provable finite-sample coverage guarantees ($\alpha = 0.01$ for sight-threatening emergencies such as DR, Glaucoma, and AMD).

5. **Low-Latency ONNX Serving & Client-Side Pre-Screening Heuristic:**
   Compiled graph execution with FP16 quantization reducing p50 serving latency to 84.2ms at 17.3 QPS (`backend/onnx_inference.py`). On the frontend, client-side image pre-screening heuristics (`frontend/src/edgeInference.js`) validate optical chromophore ratios and spatial autocorrelation in-browser (<50ms) to reject non-fundus uploads and optimize bandwidth before transmission. *Note: Full multi-class ML inference requires the server-side ensemble pipeline.*

6. **Cross-Dataset Sensor Domain Adaptation:**
   Reinhard $L\alpha\beta$ color constancy mapping matches chromatic distribution moments across disparate camera vendors (Zeiss, Topcon, Canon, handheld lenses), neutralizing optical sensor drift.

7. **CBMIR Vector Engine & Demographic Fairness Auditing:**
   512-dimensional visual embedding cosine search retrieving verified historical reference cases (`POST /api/v1/cases/similar`), verified balanced across standard demographic parity benchmarks ($0.982 \ge 0.80$) across age cohorts and optical quality grades.

8. **Multi-Tenant Clinic Application-Level Isolation:**
   Application-level tenant isolation via SQLAlchemy query filtering with X-Tenant-ID header validation (`backend/tenancy.py`) ensuring strict data isolation across healthcare providers with hierarchical RBAC (Technician $\rightarrow$ Clinician $\rightarrow$ Admin).

---

## System Architecture

> 🗺️ **Interactive Architecture Map:** Explore the interactive, standalone architecture diagram generated with Archify at [`docs/architecture.html`](docs/architecture.html).
> - **Interactive Capabilities**: Pan & zoom canvas, click nodes to inspect runtime responsibilities, trace directional data flows (`R`), step through the multi-stage clinical lifecycle via story playback (`P`), toggle fullscreen presentation mode (`F`), search components (`/`), and switch between Dark/Light themes (`T`).

<p align="center">
  <img src="docs/images/end_to_end_pipeline_architecture.png" alt="Figure 1: End-to-End Trustworthy Point-of-Care Retinal Disease Screening Pipeline Architecture" width="96%" />
  <br />
  <em><strong>Figure 1: End-to-End Trustworthy Point-of-Care Retinal Disease Screening Pipeline Architecture.</strong> Multi-stage clinical workflow integrating multi-device fundus acquisition, deterministic biophysical optical aperture guardrail &Phi;(X), sensor domain adaptation, temperature-calibrated tri-backbone soft-voting ensemble (TC-MBE: DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M), urgency-stratified conformal risk control (US-CRC), and Pixel-Aligned Saliency Grounding (PASG-GradCAM) with CBMIR reference case retrieval.</em>
</p>

<p align="center">
  <img src="docs/images/architecture_evolution_summary.png" alt="Figure 2: Empirical Architectural Evolution Across Model Generations" width="92%" />
  <br />
  <em><strong>Figure 2: Empirical Architectural Evolution Across Model Generations.</strong> Left: Training throughput progression from multi-threaded CPU baseline to GPU acceleration on RTX 5060 silicon. Right: Multi-center held-out clinical test accuracy progression from ResNet-50 baseline (75.69%) to the SOTA TC-MBE ensemble (85.18%, 0.9818 Macro AUROC, Calibrated ECE = 0.0644).</em>
</p>

```text
                                  ┌───────────────────────────┐
                                  │   React 19 Frontend SPA   │
                                  │   (Tailwind CSS + Vite 7) │
                                  └─────────────┬─────────────┘
                                                │
                                    REST API / WebSockets
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ FASTAPI BACKEND (Python 3.10+ / 3.14 / PyTorch CUDA 12.x / ONNX Runtime)                    │
│                                                                                             │
│  ┌─────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐  │
│  │ Optical Domain Filter   │ ──> │ Tenancy Isolation Guard│ ──> │ Reinhard Color Normal. │  │
│  │ (Aperture + Chromophore)│     │ (ORM-Level Filtering)  │     │ (Lαβ Sensor Transfer)  │  │
│  └────────────┬────────────┘     └────────────────────────┘     └───────────┬────────────┘  │
│               │                                                             │               │
│               └──────────────────────────────┬──────────────────────────────┘               │
│                                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ CALIBRATED TRI-BACKBONE SOFT-VOTING ENSEMBLE                                          │  │
│  │  ┌───────────────────────┐   ┌───────────────────────┐   ┌─────────────────────────┐  │  │
│  │  │ DenseNet-201 (FP16)   │   │ ConvNeXt-Small (FP16) │   │ EfficientNet-V2-M (FP16)│  │  │
│  │  │ T = 1.2616            │   │ T = 1.3407            │   │ T = 1.0654              │  │  │
│  │  └───────────┬───────────┘   └───────────┬───────────┘   └────────────┬────────────┘  │  │
│  │              └───────────────────────────┼────────────────────────────┘               │  │
│  │                                          ▼                                            │  │
│  │                         Soft-Voting Probability Averaging (ECE = 0.0644)              │  │
│  └──────────────────────────────────────────┬────────────────────────────────────────────┘  │
│                                             │                                               │
│               ┌─────────────────────────────┴─────────────────────────────┐                 │
│               ▼                                                           ▼                 │
│  ┌─────────────────────────┐                             ┌───────────────────────────────┐  │
│  │ Dedicated Grad-CAM      │                             │ CBMIR Vector Search Engine    │  │
│  │ (EfficientNet-B4)       │                             │ (512-Dim Cosine Similarity)   │  │
│  └────────────┬────────────┘                             └───────────────┬───────────────┘  │
│               │                                                           │                 │
│               ▼                                                           ▼                 │
│  ┌─────────────────────────┐                             ┌───────────────────────────────┐  │
│  │ Clinical Report Engine  │                             │ Asynchronous Task Queue       │  │
│  │ (Side-by-Side Vector PDF│                             │ (202 Accepted + WS Streaming) │  │
│  └─────────────────────────┘                             └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

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

## Empirical Benchmark Performance

<p align="center">
  <img src="docs/images/benchmark_accuracy_comparison.png" alt="Benchmark Accuracy Comparison" width="90%" />
</p>

| Architecture / Model | Test Accuracy | Macro AUROC | Macro F1 | Calibration $T$ | Calibrated ECE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Tri-Backbone Ensemble (SOTA)** | **85.18%** | **0.9818** | **0.8292** | **Ensemble** | **0.0644** |
| DenseNet-201 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0519 |
| ConvNeXt-Small | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0614 |
| EfficientNet-V2-M | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0268 |
| EfficientNet-B4 (Grad-CAM Engine) | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0582 |
| ResNet-50 (Baseline) | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0412 |

<p align="center">
  <img src="docs/images/multiclass_roc_curves.png" alt="ROC Curves" width="48%" />
  <img src="docs/images/calibration_temperatures_chart.png" alt="Calibration Temperatures" width="48%" />
</p>

<p align="center">
  <img src="docs/images/ensemble_sensitivity_specificity.png" alt="Ensemble Sensitivity and Specificity" width="48%" />
  <img src="docs/images/confusion_matrix_ensemble.png" alt="Confusion Matrix" width="48%" />
</p>

---

## Hardware Telemetry & Dual-Memory Profile

<p align="center">
  <img src="docs/images/memory_usage_comparison.png" alt="Memory Usage (VRAM + RAM)" width="48%" />
  <img src="docs/images/training_time_comparison.png" alt="Training Time Comparison (25x Speedup)" width="48%" />
</p>

- **Dual-Resource Allocation**: Measures Dedicated GPU VRAM and Host System RAM concurrently across all architectures. Peak VRAM utilization tops out at 6.30 GB GDDR6 (EfficientNet-V2-M), leaving comfortable headroom on standard 8GB GPUs (RTX 5060 Laptop GPU).
- **25x GPU Speedup**: Hardware-accelerated mixed-precision training executes an epoch in ~78.5s (RTX 5060) compared to 1,949.2s on multi-threaded CPU baseline.

---

## Precision Benchmarks: FP16 (Production) vs. BF16 (Research)

<p align="center">
  <img src="docs/images/bf16_vs_fp16_accuracy_comparison.png" alt="BF16 vs FP16 Accuracy Comparison" width="48%" />
  <img src="docs/images/bf16_vs_fp16_calibration_comparison.png" alt="BF16 vs FP16 Calibration Comparison" width="48%" />
</p>

- **Production Decision**: All architectures were independently trained in FP16 and BF16. FP16 achieves **85.18% test accuracy** (+4.16% over BF16's 81.02%) due to higher mantissa precision (10 bits vs 7 bits) preserving micro-vascular lesion gradients. FP16 is deployed in production; BF16 weights and calibrations are preserved for research.

---

## Production Systems Engineering & Enterprise Upgrades (v2.5)

<p align="center">
  <img src="docs/images/onnx_latency_throughput_benchmark.png" alt="ONNX Runtime Serving Benchmarks" width="48%" />
  <img src="docs/images/edge_vs_cloud_performance.png" alt="Edge vs Cloud Performance" width="48%" />
</p>

- **Low-Latency ONNX Runtime Serving & Quantization**:
  - Implements graph compilation, operator fusion, and FP16 quantization (`backend/onnx_inference.py`).
  - Achieves a **2.15x serving speedup** (reducing p50 latency from 181.0ms to 84.2ms) and increases throughput from 5.4 to 17.3 QPS on multi-core architectures [5].
  - Benchmarks measured on AMD64 (32-core CPU execution provider, Windows 11). See `docs/benchmarks/` for reproducible benchmark scripts (`scripts/benchmark_onnx.py`) and persistent JSON output (`docs/benchmarks/onnx_benchmark_results.json`). *Note: When compiled ONNX model weights are not locally present, scripts output clearly flagged synthetic reference projections.*
- **Client-Side Image Pre-Screening Heuristics for Bandwidth Optimization**:
  - In-browser client-side optical verification via HTML5 Canvas pixel extraction (`frontend/src/edgeInference.js`).
  - Turnaround time <50ms with zero cloud bandwidth consumed on invalid uploads, deterministically verifying chromophore ratios ($R/B \ge 1.05$) and spatial autocorrelation ($r_{\text{spatial}} \ge 0.30$) before network transmission.
  - *Note: Full multi-class ML inference requires the server-side ensemble pipeline.*

<p align="center">
  <img src="docs/images/async_task_architecture.png" alt="Asynchronous Task Queue & WebSocket Streaming" width="96%" />
</p>

- **Asynchronous Task Queue & Real-Time WebSocket Streaming**:
  - Decouples heavy multi-backbone GPU compute from the HTTP request cycle (`backend/async_screening.py`).
  - `POST /api/v1/screen/async` returns an immediate `202 Accepted` job ticket. Continuous stage telemetry is streamed to clients via `WebSocket /ws/jobs/{id}` across 5 discrete execution stages.

<p align="center">
  <img src="docs/images/sensor_domain_adaptation_analysis.png" alt="Sensor Domain Shift Adaptation" width="48%" />
  <img src="docs/images/hitl_active_learning_loop.png" alt="Human-in-the-Loop Active Learning" width="48%" />
</p>

- **Cross-Dataset Generalization & Reinhard Color Constancy**:
  - Automatically identifies optical sensor drift across camera vendors (Zeiss, Topcon, handheld lenses) using chromatic distribution moments.
  - Normalizes color balance in $L\alpha\beta$ space (`backend/domain_adaptation.py`), preventing feature extractor degradation.
- **Human-in-the-Loop (HITL) & Active Learning Pipeline**:
  - Clinician override and attestation system (`backend/routes_admin.py`).
  - Measures clinical concordance rate (90.5%), logs diagnostic discordance, and mines high-confidence AI error modes into candidate sets for active learning retraining loops.

<p align="center">
  <img src="docs/images/vector_search_cbmir.png" alt="CBMIR Vector Search" width="48%" />
  <img src="docs/images/fairness_slice_audit.png" alt="Demographic Fairness Audit" width="48%" />
</p>

- **Content-Based Medical Image Retrieval (CBMIR) Vector Engine**:
  - Implements dense 512-dimensional visual embedding indexing and normalized cosine similarity search (`backend/vector_search.py`).
  - Endpoint `POST /api/v1/cases/similar` retrieves top-$k$ reference cases from historical archives with multimodal ophthalmic & OCT-confirmed pathology and longitudinal patient outcomes, grounding deep learning predictions with empirical case history.
- **Demographic Fairness, Algorithmic Bias & Slice Auditing**:
  - Comprehensive clinical slice disparity auditor (`backend/fairness_audit.py`).
  - Evaluates Equalized Odds across demographic cohorts (Age: $<45$, $45-65$, $>65$; Optical Quality Grades A/B; Systemic Comorbidities).
  - Confirms compliance with FDA SaMD fairness guidelines and demographic parity benchmarks (Disparate Impact Ratio = $0.982 \ge 0.80$, Equalized Odds Disparity = $0.016 \le 0.10$).

<p align="center">
  <img src="docs/images/observability_opentelemetry.png" alt="Prometheus & OpenTelemetry Observability" width="48%" />
  <img src="docs/images/multitenant_clinic_isolation.png" alt="Multi-Tenant Clinic Application-Level Isolation" width="48%" />
</p>

- **Prometheus Telemetry & OpenTelemetry Distributed Tracing**:
  - Standard Prometheus exposition exporter (`GET /metrics`, `backend/metrics.py`) tracking inference requests, latency quantiles (p50/p90/p99), GPU VRAM memory gauges, and optical domain shift counters.
  - Microsecond-precision distributed span tracing (`backend/tracing.py`, `GET /api/v1/traces/recent`) providing end-to-end latency waterfall visibility across ingestion, preprocessing, inference, and serialization.
- **Multi-Tenant Clinic Architecture & Application-Level Isolation**:
  - Application-level tenant isolation via SQLAlchemy query filtering with X-Tenant-ID header validation (`backend/tenancy.py`, `backend/db.py`).
  - Resolves clinic context via `X-Tenant-ID` header or authenticated JWT claims, automatically binding tenant query filtering (`WHERE tenant_id = :tenant_id`) across scans, users, and audit trails to guarantee zero cross-hospital data leakage without requiring PostgreSQL-native `CREATE POLICY` database engine configuration.
  - Hierarchical Role-Based Access Control (`ROLE_HIERARCHY`: Technician $\rightarrow$ Clinician $\rightarrow$ Admin).

---

## Multi-Tenant Analytics & Monitoring Dashboard
OphthalmoAI includes a production-grade analytics module with:
- **Tenant-Level KPIs**: Screening volumes, confidence trends,
  inference performance, and diagnosis distribution — isolated per tenant
- **Anomaly Detection**: Z-score, moving average, SLA breach, and
  distribution shift detection on screening time-series data
- **Real-Time Alerts**: Automated anomaly notifications with severity
  classification and actionable clinical recommendations

### Ad-Tech Engineering Patterns
The analytics module intentionally demonstrates patterns used in
advertising technology platforms:
| Screening Metric | Ad-Tech Equivalent |
|-----------------|-------------------|
| Screening volume anomaly | Impression volume anomaly |
| Confidence score decline | CTR/conversion rate decline |
| Inference SLA breach | Bid response latency in RTB |
| Diagnosis distribution shift | Audience composition drift |

These patterns are directly transferable to campaign monitoring,
performance analytics, and automated optimization systems.

---

## Quick Start (Local Setup)

### 1-Click Launch (Recommended)
Launch both FastAPI backend and Vite frontend with automatic GPU detection and browser launch:

```bash
# Windows (Double-click or run from CMD):
start.bat

# PowerShell:
powershell -ExecutionPolicy Bypass -File .\start.ps1

# Linux / macOS / WSL:
chmod +x start.sh && ./start.sh

# Public GPU Mode (RTX 5060 + Cloudflare Tunnel + Hugging Face & Vercel Sync):
start_public_gpu.bat
```

---

### Manual Setup & Execution

#### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/AkashKundu114/OphthalmoAI.git
cd OphthalmoAI

# Create and activate virtual environment
python -m venv venv
# On Windows: .\venv\Scripts\activate
# On Linux/macOS: source venv/bin/activate

# Install PyTorch and dependencies (CUDA 12.4+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r backend/requirements.txt

# Start backend server
python backend/main.py
```
> Backend API serves at `http://localhost:8000` (Swagger UI at `http://localhost:8000/docs`).

#### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
> Frontend SPA serves at `http://localhost:5173`.

#### 3. Run Quality Gates & Tests
```bash
# Run all 262 automated Pytest unit and integration tests
pytest tests -q

# Run all 21 frontend Vitest unit tests
cd frontend && npm test -- --run

# Run frontend build verification
cd frontend && npm run build
```

#### 4. Database Architecture & Multi-Backend Configuration
Supports PostgreSQL (production), MS SQL Server (enterprise), and SQLite (development). Configure via `DATABASE_URL` environment variable.

| Backend | Driver / Dialect | Intended Environment | Connection String Example |
| :--- | :--- | :--- | :--- |
| **SQLite** | `sqlite` | **Local Development & Testing (Default)** | `DATABASE_URL=sqlite:///./ophthalmoai.db` |
| **PostgreSQL** | `postgresql+psycopg` / `postgresql` | **Production & Multi-Instance Clusters** | `DATABASE_URL=postgresql://ophthalmo:password@localhost:5432/ophthalmoai` |
| **MS SQL Server** | `mssql+pyodbc` | **Enterprise Hospital Networks & EHR Integration** | `DATABASE_URL=mssql+pyodbc://sa:password@localhost/ophthalmoai?driver=ODBC+Driver+18+for+SQL+Server` |

- **Development vs. Production**: The platform defaults to SQLite for zero-config local development, issuing a clear runtime log warning (`WARNING: Using SQLite for development. Set DATABASE_URL for production.`). For production deployments, PostgreSQL or MS SQL Server is required with production connection pooling (`pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`).
- **PostgreSQL via Docker Compose**: Launch production PostgreSQL with automated health checks and initialization extensions (`scripts/init_db.sql`):
  ```bash
  docker compose up -d db
  ```
- **Alembic Database Migrations**: Run migrations across any supported backend with automatic batch mode for SQLite:
  ```bash
  alembic upgrade head
  ```
- **Database Connection Verification**: Diagnostic utility in `backend/db_utils.py` validates connectivity and logs database engine metadata:
  ```python
  from backend.db_utils import verify_database_connection
  verify_database_connection()
  ```

---

## Repository Directory Structure

```text
OphthalmoAI/
├── backend/                       # FastAPI backend, ensemble models, routes, services
│   ├── domain_validator.py        # Optical aperture & chromophore backscatter guardrails
│   ├── onnx_inference.py          # Low-latency ONNX Runtime FP16 execution engine
│   ├── async_screening.py         # Asynchronous job queue & WebSocket telemetry
│   ├── domain_adaptation.py       # Reinhard Lαβ color constancy transfer
│   ├── vector_search.py           # 512-dim CBMIR normalized cosine embedding index
│   ├── fairness_audit.py          # EEOC Four-Fifths demographic slice disparity auditor
│   ├── tenancy.py                 # Multi-tenant clinic application-level isolation & RBAC
│   ├── tracing.py                 # OpenTelemetry microsecond span tracing
│   └── routes_admin.py            # HITL overrides, active learning, and audit logs
├── frontend/                      # Standalone React 19 SPA (Tailwind CSS + Vite 7)
│   ├── src/                       # React components, clinical persona switcher, PDF export
│   └── src/edgeInference.js       # Client-side image pre-screening heuristic for bandwidth optimization
├── models/                        # Trained PyTorch weights & Platt calibration JSONs
├── scripts/                       # Standardized 5-pillar operational & automation scripts
│   ├── README.md                  # Complete operational scripts catalog & usage reference
│   ├── evaluate_external_dataset.py # Automated external multi-cohort validation pipeline
│   ├── generate_external_figures.py # Zero-overlap 300 DPI publication visual generator
│   └── fine_tune_external_ensemble.py # Layer-selective fine-tuning with AMP FP16
├── docs/                          # Comprehensive technical and clinical documentation suite
│   ├── architecture.html          # Interactive Archify SVG architecture map (pan/zoom, keyboard nav)
│   ├── images/                    # 29 publication-grade academic figures
│   ├── benchmarks/                # Benchmark result JSONs (onnx_benchmark_results.json)
│   ├── clinical/                  # Clinical safety, intended use, and external validation reports
│   │   ├── CLINICAL_EVALUATION_AND_SAFETY.md # Intended use & risk mitigation
│   │   └── EXTERNAL_VALIDATION_REPORT.md     # Multi-cohort external validation & generalization study
│   ├── design/                    # UI/UX brief and user application flow
│   ├── research/                  # Formal research paper draft and mathematical derivations
│   └── technical/                 # System architecture, schemas, and security audits
├── deploy/                        # Production deployment manifests (Hugging Face, Docker)
├── k8s/                           # Production Kubernetes manifests and ingress configs
└── tests/                         # 283 automated tests (262 backend pytest + 21 frontend vitest)
```

---

## Documentation Suite

- **[Interactive Architecture Map](docs/architecture.html)**: Interactive Archify SVG system architecture map with keyboard shortcuts and flow inspection.
- **[System Specification](docs/SYSTEM_SPECIFICATION.md)**: Technical architecture, pipeline stages, and QA checklist.
- **[Performance & Telemetry](docs/PERFORMANCE_METRICS.md)**: Comprehensive empirical metrics, ROC curves, calibration charts, and GPU profiling.
- **[External Clinical Validation Report](docs/clinical/EXTERNAL_VALIDATION_REPORT.md)**: Independent multi-cohort validation on IDRiD ($n=103$) and RIM-ONE DL ($n=447$).
- **[Clinical Evaluation & Safety](docs/clinical/CLINICAL_EVALUATION_AND_SAFETY.md)**: Intended use, clinical risk controls, and validation protocols.
- **[Scripts Catalog & Operations Guide](scripts/README.md)**: Reference guide for the consolidated 5-pillar script architecture.
- **[Technical White Paper](docs/OphthalmoAI_Technical_White_Paper.md)**: Engineering methodology, ensemble formulations, and explainability.
- **[Production Guide](PRODUCTION.md)**: Deployment guidelines for Docker, Kubernetes, and cloud environments.
- **[Roadmap](ROADMAP.md)**: Product roadmap, completed milestones, and upcoming v2.6 / v3.0 horizons.

---

## Benchmark Methodology & Metric Sources

1. **Internal Test Metrics (85.18% Accuracy, 0.9818 Macro AUROC, 0.0381 ECE):** Evaluated over strictly held-out clinical fundus test scans ($n=938$) across 6 target classes. Calibrated via Platt temperature scaling ($T \in [1.06, 1.34]$). Detailed in `docs/PERFORMANCE_METRICS.md` and `REPRODUCIBILITY.md`.
2. **External Clinical Validation (IDRiD):** 91.30% sensitivity across $n=103$ test scans acquired on a Kowa VX-10 $\alpha$ digital fundus camera in Nanded, India. Detailed in `docs/clinical/EXTERNAL_VALIDATION_REPORT.md`.
3. **Autonomous Escalation Net (RIM-ONE DL):** 100% fail-safe escalation rate (`requires_human_review: true`) triggered across $n=447$ localized optic disc crops from Hospital Universitario de Canarias, Spain. Detailed in `docs/clinical/EXTERNAL_VALIDATION_REPORT.md`.
4. **Automated Test Suite Verification:** 283 passed tests (100% pass rate) verified live: 262 backend Pytest tests (216 in `tests/backend/` + 46 in `tests/test_analytics.py` and `tests/test_boundary_conditions.py`) and 21 frontend Vitest unit tests (5 test suites in `frontend/tests/`).
5. **ONNX Runtime Serving & Latency:** Measured on AMD64 (32-core CPU execution provider, Windows 11). Reproducible via `scripts/benchmark_onnx.py`; persistent benchmark telemetry logged in `docs/benchmarks/onnx_benchmark_results.json`. Note: When compiled ONNX weights are not locally present, benchmark scripts output clearly flagged synthetic reference projections.

---

## Limitations and Known Issues

To ensure full technical defensibility under source-code audit and interview scrutiny, the following engineering boundaries and active constraints are documented:

1. **Client-Side Screening vs. Full ML Inference:**
   - The in-browser screening module (`frontend/src/edgeInference.js`) provides lightweight optical pre-screening heuristics (chromophore ratios and spatial autocorrelation) designed for client-side bandwidth optimization and instant non-fundus rejection.
   - Full diagnostic multi-class classification, conformal prediction sets, and Grad-CAM saliency generation require the server-side PyTorch / ONNX ensemble pipeline.

2. **Field-of-View (FOV) Sensor Shift:**
   - As documented in the external clinical validation study (`docs/clinical/EXTERNAL_VALIDATION_REPORT.md`), localized optic disc crops (e.g., RIM-ONE DL, 292×292 px) lacking the macula and temporal arcade trigger the clinical uncertainty gate (`requires_human_review: true`), as the ensemble requires canonical 45° posterior pole fundus photographs.

3. **Application-Level Multi-Tenancy:**
   - Multi-tenant clinic data isolation is implemented at the application/ORM layer via SQLAlchemy query filtering and JWT/header validation (`backend/tenancy.py`), rather than through PostgreSQL-native database `CREATE POLICY` (Row-Level Security) rules. Direct database queries bypassing the application layer do not enforce tenant filtering.

4. **ONNX Weights & Benchmark Fallback:**
   - In environments where large compiled ONNX weight files (~500 MB+) are not downloaded, the benchmarking utility (`scripts/benchmark_onnx.py`) outputs synthetic reference estimates clearly flagged as `is_synthetic: true` (`docs/benchmarks/onnx_benchmark_results.json`).

5. **Database Concurrency in Development:**
   - SQLite is configured with WAL mode for local zero-config testing but is limited to a single concurrent writer; multi-user clinical production deployments require PostgreSQL or MS SQL Server via `DATABASE_URL`.

6. **Regulatory Status:**
   - OphthalmoAI is a clinical decision-support and screening-aid research platform (academic research manuscript in preparation). It is not FDA 510(k) cleared or CE-marked as a primary diagnostic medical device. All findings must be corroborated by licensed clinicians.

---

## Author, Intellectual Property & License

**OphthalmoAI** is an **independent clinical AI decision-support platform** architected, developed, and maintained by **Akash Kundu**.

- **Copyright:** Copyright &copy; 2026 Akash Kundu.
- **License:** Distributed under the **Apache License 2.0**. See the [`LICENSE`](LICENSE) file for complete terms.
- **Permitted Operations:** Free for academic research, education, and point-of-care clinical evaluation with proper author attribution.

---

## Security & Community Governance

- **Security Policy & Vulnerability Disclosure:** Consult [`SECURITY.md`](SECURITY.md) for vulnerability reporting and HIPAA threat models.
- **Code of Conduct:** Review [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community participation and impact guidelines.
- **Contributing Guidelines:** Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for quality gates, clinical data rules, and Google XYZ PR requirements.
- **Support & FAQ:** Visit [`SUPPORT.md`](SUPPORT.md) for documentation guides and issue routing.
- **Issue Guidelines:** Review [`ISSUES.md`](ISSUES.md) for reporting bugs and clinical domain false alarms.
- **Privacy Policy:** Read [`PRIVACY_POLICY.md`](PRIVACY_POLICY.md) for HIPAA, GDPR, and offline edge screening protections.
- **Terms and Conditions:** Review [`TERMS_AND_CONDITIONS.md`](TERMS_AND_CONDITIONS.md) for medical decision-support disclaimers and liability limitations.
