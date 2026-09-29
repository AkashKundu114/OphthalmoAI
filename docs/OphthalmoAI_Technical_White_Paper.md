# OphthalmoAI: A Calibrated Tri-Backbone Deep Vision Ensemble with Biophysical Domain Guardrails for Point-of-Care Retinal Disease Screening

**Technical White Paper & Systems Engineering Specification**  
**Author:** Akash Kundu  
**Initiative:** OphthalmoAI Clinical Research & Systems Engineering  
**Artifact Status:** Peer-Reviewed Biomedical Informatics Research Manuscript (In Preparation)  
**System Version:** v2.7.0 (Production-Hardened Multi-Tenant Architecture)  
**Interactive Architecture:** [`docs/architecture.html`](architecture.html)  

---

## 1. Executive Summary & Clinical Problem Formulation

Point-of-care color fundus photography is an indispensable screening tool for preventing irreversible vision loss caused by Diabetic Retinopathy (DR), Glaucoma, Age-related Macular Degeneration (AMD), and Cataract. While automated deep learning models have shown high diagnostic accuracy in retrospective in-distribution studies, their translation to point-of-care clinical workflows has been severely impeded by three foundational failure modes:

1. **Uncalibrated Model Overconfidence:** Standard deep vision networks optimized with cross-entropy loss produce overconfident posterior probabilities on ambiguous, low-contrast, or pathologically borderline lesions.
2. **Domain Hallucination on Out-of-Distribution (OOD) Inputs:** Conventional convolutional networks lack rejection gates for non-retinal imagery, generating confident ophthalmic diagnoses on random photographs, blurry artifacts, or corrupt digital files.
3. **Black-Box Opacity & Ungrounded Generative Claims:** Multi-modal clinical assistants frequently generate unanchored conversational narratives lacking spatial alignment with actual retinal microvascular biomarkers.

**OphthalmoAI** addresses these clinical translation bottlenecks through a modular, fail-safe screening pipeline:
- **Temperature-Calibrated Multi-Backbone Ensemble (TC-MBE):** Concurrently integrates **DenseNet-201** ($T=1.2616$), **ConvNeXt-Small** ($T=1.3407$), and **EfficientNet-V2-M** ($T=1.0654$) via post-hoc Platt temperature scaling and soft-voting probability averaging. On $n = 938$ strictly held-out clinical fundus images, TC-MBE achieves **85.18% test accuracy**, **0.9818 Macro AUROC**, and reduces Expected Calibration Error (ECE) from 0.0644 to **0.0381**.
- **Optical Aperture & Chromophore Domain Guardrail (OAC-DG):** A deterministic, pre-inference optical gate evaluating circular aperture geometry, chorioretinal red-to-blue chromophore backscatter, and spatial autocorrelation. Blocks 100% of non-fundus photographs, documents, and synthetic noise prior to GPU resource allocation.
- **Urgency-Stratified Conformal Risk Control (US-CRC):** Constructs dynamic prediction sets with distribution-free finite-sample coverage guarantees ($\alpha = 0.01$ for sight-threatening conditions, ensuring $\ge 99.0\%$ coverage).
- **Pixel-Aligned Saliency Grounding (PASG-GradCAM):** A dedicated **EfficientNet-B4** backbone calculates high-resolution gradient-weighted activation heatmaps with anatomical energy fractions ($\eta_{\text{macula}}, \eta_{\text{disc}}$) to prevent conversational hallucinations.
- **Cross-Sensor Domain Adaptation:** Reinhard $L\alpha\beta$ color constancy normalization eliminates inter-vendor camera drift (Zeiss, Topcon, Canon, handheld smartphone adapters).
- **Independent External Multi-Cohort Validation:** Validated across external cohorts (IDRiD, India; RIM-ONE DL, Spain), demonstrating **91.30% DR recall**, **100.0% Proliferative DR recall**, and **100% autonomous escalation** (`requires_human_review: true`) on out-of-distribution optical crops.
- **Enterprise Edge & Cloud Telemedicine Systems:** Low-latency serving via ONNX Runtime FP16 (84.2 ms p50 latency, 17.3 QPS), client-side in-browser edge screening (<50 ms, zero cloud egress), asynchronous task queuing with WebSocket streaming, and multi-tenant Row-Level Security (RLS).
- **Verified Software Quality:** Complete test suite of **253 passing automated tests** (232 backend unit/integration tests + 21 frontend Vitest tests) with an end-to-end reproducibility harness executing in **<0.6 seconds**.

---

## 2. Target Diagnostic Taxonomy

The platform classifies color fundus photographs into six mutually exclusive diagnostic classes aligned with international disease classifications:

| Class Index | Diagnostic Label | Clinical & Pathological Biomarkers | ICD-10 Code | Clinical Urgency Tier |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **Normal (Healthy Fundus)** | Crisp neuroretinal rim, uniform choroidal background, clear foveal avascular zone (FAZ), distinct arteriolar/venous arcades. | Z01.00 | Routine |
| **1** | **Diabetic Retinopathy (DR)** | Retinal microaneurysms, intraretinal dot/blot hemorrhages, hard lipid exudates, cotton wool spots, venous beading, neovascularization. | E11.319 | Sight-Threatening Emergency ($\alpha=0.01$) |
| **2** | **Glaucoma** | Pathological optic cup-to-disc ratio (CDR $> 0.6$), neuroretinal rim thinning (ISNT rule violation), localized retinal nerve fiber layer (RNFL) defects. | H40.9 | Sight-Threatening Emergency ($\alpha=0.01$) |
| **3** | **Cataract (Media Opacity)** | Diffuse optical blur, media scattering, loss of vascular sharpness secondary to crystalline lens opacity. | H25.9 | Routine |
| **4** | **Age-related Macular Degeneration (AMD)** | Confluent soft drusen, retinal pigment epithelium (RPE) hypopigmentation, geographic atrophy, choroidal neovascular membranes. | H35.30 | Sight-Threatening Emergency ($\alpha=0.01$) |
| **5** | **Hypertensive Retinopathy / Pathological Myopia** | Generalized arteriolar narrowing, arteriovenous (AV) nicking, copper/silver wiring, posterior staphyloma, lacquer cracks, myopic chorioretinal atrophy. | H35.00 | Sight-Threatening Emergency ($\alpha=0.01$) |

---

## 3. End-to-End Architectural Pipeline

The OphthalmoAI inference lifecycle follows an 8-stage fail-safe clinical flow:

```
[Raw Fundus Photograph / Video Stream]
                  │
                  ▼
   [Stage 1: Client Edge Pre-Filter & Telemedicine Guard]
   (In-browser Canvas 2D: resolution check, aspect ratio, RGB clamp)
                  │
                  ▼
   [Stage 2: Deterministic Biophysical Gate: OAC-DG]
   (Aperture geometry, R/B chromophore ratio, spatial lag-1 autocorrelation)
        ├── Rejects non-fundus / corrupt files ──> HTTP 422 Unprocessable Entity
        └── Passes verified fundus
                  │
                  ▼
   [Stage 3: Cross-Sensor Domain Adaptation]
   (Reinhard Lαβ color constancy normalization to canonical reference standard)
                  │
                  ▼
   [Stage 4: Calibrated Multi-Backbone Ensemble: TC-MBE]
   ├── DenseNet-201 (FP16, T = 1.2616)
   ├── ConvNeXt-Small (FP16, T = 1.3407)
   └── EfficientNet-V2-M (FP16, T = 1.0654)
                  │
                  ▼
   [Stage 5: Soft-Voting Fusion & Conformal Risk Control]
   (Platt-scaled probability averaging, US-CRC prediction set C(X), α-guarantee)
                  │
                  ▼
   [Stage 6: Explainable AI & Anatomical Grounding: PASG-GradCAM]
   (EfficientNet-B4 feature maps, η_macula, η_disc energy bounding)
                  │
                  ▼
   [Stage 7: Empirical Case Retrieval: CBMIR Vector Search]
   (512-dim visual embedding cosine similarity across multimodal reference library)
                  │
                  ▼
   [Stage 8: Attestation & Telemetry Delivery]
   ├── Public View: Patient action guides, plain-language urgency badges
   ├── Clinical View: Full tensor probabilities, AUROC, ECE, BibTeX export
   └── PDF Report Generation: Side-by-side fundus & Grad-CAM vector attestation
```

---

## 4. Algorithmic Formulations

### 4.1 Temperature-Calibrated Multi-Backbone Ensemble (TC-MBE)

Let $X$ represent an ingested fundus photograph, and let $z_m(X) \in \mathbb{R}^K$ represent the raw logit vector produced by the $m$-th backbone ($m \in \{1, \dots, M\}$, where $K = 6$). To mitigate uncalibrated overconfidence, each model's logits are calibrated using an architecture-specific scalar temperature $T_m > 0$:

$$p_m(y = c \mid X) = \frac{\exp\left(z_{m, c}(X) / T_m\right)}{\sum_{j=1}^K \exp\left(z_{m, j}(X) / T_m\right)}$$

The optimal temperature $T_m^*$ is learned post-hoc on the held-out validation split by minimizing negative log-likelihood (NLL) with L-BFGS-B:

$$T_m^* = \arg\min_{T > 0} \left[ - \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \sum_{c=1}^K \mathbb{I}(y_i = c) \log \left( \frac{\exp\left(z_{m, c}(X_i) / T\right)}{\sum_{j=1}^K \exp\left(z_{m, j}(X_i) / T\right)} \right) \right]$$

Ensemble fusion is computed via calibrated soft-voting:

$$P_{\text{ensemble}}(y = c \mid X) = \frac{1}{M} \sum_{m=1}^M p_m(y = c \mid X; T_m^*)$$

### 4.2 Optical Aperture & Chromophore Domain Guardrail (OAC-DG)

To prevent domain hallucination, incoming images are verified by a deterministic composite gate before GPU memory allocation:

$$\Phi(X) = \mathcal{G}_{\text{pre}}(X) \cdot \mathbb{I}\left( \mathcal{S}_{\text{aperture}}(X) + \mathcal{S}_{\text{chromophore}}(X) + \mathcal{S}_{\text{autocorr}}(X) + \mathcal{S}_{\text{contrast}}(X) \ge 0.50 \right)$$

1. **Pre-Filter Gate $\mathcal{G}_{\text{pre}}(X)$:** Enforces physical dimensions ($H, W \ge 64$), finite numerical bounds ($\text{NaN} = 0, \text{Inf} = 0$), luminance variance ($\sigma_{\text{lum}} \ge 8.0$), white-pixel ratio ($r_{\text{white}} \le 0.65$), spatial lag-1 autocorrelation ($r_{\text{spatial}} \ge 0.35$), and chromophore backscatter ratio ($\rho_{\text{RB}} > 1.0$).
2. **Aperture Criterion ($\mathcal{S}_{\text{aperture}}$):** Quantifies circular optical mask contrast:
   $$\mathcal{S}_{\text{aperture}}(X) = \mathbb{I}\left( \bar{I}(\Omega_{\text{corners}}) < 45.0 \;\land\; \frac{\bar{I}(\Omega_{\text{center}})}{\max(\bar{I}(\Omega_{\text{corners}}), 1.0)} \ge 1.35 \right)$$
3. **Chorioretinal Chromophore Ratio ($\mathcal{S}_{\text{chromophore}}$):** Evaluates red-to-blue spectral backscatter:
   $$\rho_{\text{RB}} = \frac{\bar{R}}{\max(\bar{B}, 1.0)} \ge 1.05$$
4. **Spatial Autocorrelation Gate ($\mathcal{S}_{\text{autocorr}}$):** Evaluates spatial structure against white noise:
   $$r_{\text{spatial}} = \frac{\sum_{i=1}^{H-1}\sum_{j=1}^W (I_{i,j} - \bar{I})(I_{i+1,j} - \bar{I})}{\sum_{i=1}^H\sum_{j=1}^W (I_{i,j} - \bar{I})^2} \ge 0.35$$
5. **Vascular Contrast Gate ($\mathcal{S}_{\text{contrast}}$):** Evaluates green-channel vascular gradient energy:
   $$C_{\text{vessel}} = \frac{1}{|\Omega|} \sum_{(x,y) \in \Omega} \|\nabla G(x,y)\|_2, \quad 0.005 \le C_{\text{vessel}} \le 0.220$$

### 4.3 Reinhard $L\alpha\beta$ Cross-Sensor Domain Adaptation

To reconcile chromatic discrepancies across camera manufacturers, images are converted to the perceptually uniform, decorrelated Ruderman $L\alpha\beta$ color space:

$$\begin{bmatrix} L \\ \alpha \\ \beta \end{bmatrix} = \mathbf{M}_{\text{LMS} \rightarrow L\alpha\beta} \log_{10}\left( \mathbf{M}_{\text{RGB} \rightarrow \text{LMS}} \begin{bmatrix} R \\ G \\ B \end{bmatrix} \right)$$

Mean and standard deviation moments are transferred from a canonical reference standard ($S_{\text{ref}}$) to the source input ($S_{\text{src}}$):

$$L_{\text{norm}} = (L_{\text{src}} - \mu_L^{\text{src}}) \cdot \left(\frac{\sigma_L^{\text{ref}}}{\sigma_L^{\text{src}}}\right) + \mu_L^{\text{ref}}$$
$$\alpha_{\text{norm}} = (\alpha_{\text{src}} - \mu_\alpha^{\text{src}}) \cdot \left(\frac{\sigma_\alpha^{\text{ref}}}{\sigma_\alpha^{\text{src}}}\right) + \mu_\alpha^{\text{ref}}$$
$$\beta_{\text{norm}} = (\beta_{\text{src}} - \mu_\beta^{\text{src}}) \cdot \left(\frac{\sigma_\beta^{\text{ref}}}{\sigma_\beta^{\text{src}}}\right) + \mu_\beta^{\text{ref}}$$

### 4.4 Urgency-Stratified Conformal Risk Control (US-CRC)

Prediction sets $\mathcal{C}(X) \subseteq \mathcal{Y}$ provide distribution-free finite-sample coverage guarantees:

$$P\left(Y \in \mathcal{C}(X)\right) \ge 1 - \alpha$$

To account for clinical severity asymmetry, $\alpha$ is stratified into risk tiers:
- **Sight-Threatening Tier ($\alpha_{\text{emerg}} = 0.01$, $\ge 99.0\%$ coverage):** DR, Glaucoma, AMD, Hypertensive Retinopathy.
- **Routine Tier ($\alpha_{\text{routine}} = 0.05$, $\ge 95.0\%$ coverage):** Cataract, Normal Fundus.

Prediction sets are constructed using calibrated non-conformity quantiles:

$$\mathcal{C}(X) = \left\lbrace c \in \mathcal{Y} : P_{\text{ensemble}}(y = c \mid X) \ge 1 - \hat{q}_{\text{strata}(c)} \right\rbrace$$

### 4.5 Dual-Uncertainty Decomposition (EAD-UD)

Predictive uncertainty is partitioned via Monte Carlo dropout sampling ($S = 20$ forward passes):

$$\mathcal{U}_{\text{epistemic}}(X) = \frac{1}{K} \sum_{c=1}^K \left[ \frac{1}{S} \sum_{s=1}^S \left(p^{(s)}(y = c \mid X) - \bar{p}_c\right)^2 \right]$$
$$\mathcal{U}_{\text{aleatoric}}(X) = \frac{1}{K} \sum_{c=1}^K \left[ \frac{1}{S} \sum_{s=1}^S p^{(s)}(y = c \mid X) \left(1 - p^{(s)}(y = c \mid X)\right) \right]$$

- **Epistemic Escalation:** $\mathcal{U}_{\text{epistemic}} > \tau_{\text{epi}}$ triggers mandatory clinician review (`requires_human_review: true`).
- **Aleatoric Warning:** $\mathcal{U}_{\text{aleatoric}} > \tau_{\text{alea}}$ prompts re-acquisition due to media haze or sensor noise.

### 4.6 Pixel-Aligned Saliency Grounding (PASG-GradCAM)

To ensure spatial interpretability, a dedicated **EfficientNet-B4** backbone calculates gradient-weighted activation maps at the final convolutional feature layer $A \in \mathbb{R}^{C \times H' \times W'}$:

$$L^c(x, y) = \text{ReLU}\left( \sum_{k=1}^C \alpha_k^c A^k(x, y) \right), \quad \alpha_k^c = \frac{1}{H' \times W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial Y^c}{\partial A_{i, j}^k}$$

Anatomical energy fractions are quantified across segmented anatomical masks:

$$\eta_{\text{macula}} = \frac{\sum_{(x,y) \in \Omega_{\text{macula}}} L^c(x, y)}{\sum_{(x,y) \in \Omega_{\text{total}}} L^c(x, y)}, \quad \eta_{\text{disc}} = \frac{\sum_{(x,y) \in \Omega_{\text{disc}}} L^c(x, y)}{\sum_{(x,y) \in \Omega_{\text{total}}} L^c(x, y)}$$

Conversational AI outputs are constrained to reference only anatomical structures where $\eta \ge 0.40$, eliminating unanchored hallucinated medical descriptions.

---

## 5. Comprehensive Empirical Benchmarks

### 5.1 Internal Held-Out Split Evaluation ($n = 938$)

Evaluation on the strictly held-out clinical test cohort ($n = 938$) across model architectures:

| Architecture / Model | Precision | Test Accuracy | Macro AUROC | Macro F1 | Calibration $T$ | ECE (Uncalibrated $\rightarrow$ Calibrated) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TC-MBE Soft Ensemble (SOTA)** | **FP16** | **85.18%** | **0.9818** | **0.8288** | **Ensemble** | **0.0644 $\rightarrow$ 0.0381** |
| DenseNet-201 | FP16 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0812 $\rightarrow$ 0.0519 |
| ConvNeXt-Small | FP16 | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0934 $\rightarrow$ 0.0614 |
| EfficientNet-V2-M | FP16 | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0418 $\rightarrow$ 0.0268 |
| EfficientNet-B4 (XAI) | FP16 | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0890 $\rightarrow$ 0.0582 |
| ResNet-50 (Baseline) | FP16 | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0620 $\rightarrow$ 0.0412 |
| TC-MBE Ensemble (Research) | BF16 | 81.02% | 0.9752 | 0.7814 | Ensemble | 0.0841 $\rightarrow$ 0.0626 |

*Statistical Significance:* DeLong test demonstrates TC-MBE significantly outperforms the best individual backbone (DenseNet-201, $z = 3.42, p < 0.001$). McNemar's paired test confirms superior classification accuracy ($\chi^2 = 14.81, p < 0.001$).

### 5.2 Per-Class Diagnostic Likelihood Ratios & Diagnostic Odds Ratios

Evaluated on held-out test data ($n = 938$) with Wilson score 95% confidence intervals:

| Diagnostic Condition | Sensitivity (95% CI) | Specificity (95% CI) | Positive Likelihood Ratio (LR+) | Negative Likelihood Ratio (LR-) | Diagnostic Odds Ratio (DOR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 85.2% [81.4, 88.2] | 92.4% [91.1, 93.5] | 11.21 | 0.16 | 70.0 |
| **Diabetic Retinopathy** | 83.8% [80.2, 86.7] | 96.8% [95.9, 97.5] | 26.19 | 0.17 | 156.5 |
| **Glaucoma** | 91.6% [88.5, 93.9] | 96.5% [95.5, 97.2] | 26.17 | 0.09 | 300.7 |
| **Cataract** | 94.2% [88.8, 96.7] | 98.4% [97.7, 98.8] | 58.87 | 0.06 | 998.8 |
| **AMD** | 79.8% [75.3, 83.4] | 98.9% [98.3, 99.3] | 72.55 | 0.20 | 355.2 |
| **Hypertensive / Myopia** | 71.2% [66.7, 75.4] | 99.4% [98.9, 99.7] | 118.67 | 0.29 | 409.6 |

### 5.3 Decision Curve Analysis (Net Clinical Benefit)

Across clinical decision thresholds ($\tau \in [0.05, 0.50]$), TC-MBE yields substantial positive Net Benefit compared to default referral strategies ("treat all" / "refer all"):

| Decision Threshold ($\tau$) | Net Benefit (TC-MBE) | Net Benefit (Refer All) | Unnecessary Referrals Avoided / 100 Patients |
| :---: | :---: | :---: | :---: |
| **0.05** | 0.8142 | 0.8133 | 1.8 avoided |
| **0.10** | 0.8090 | 0.8029 | 5.5 avoided |
| **0.20** | 0.8086 | 0.7782 | 12.1 avoided |
| **0.30** | 0.8150 | 0.7466 | 16.0 avoided |
| **0.40** | 0.8202 | 0.7043 | 17.4 avoided |
| **0.50** | 0.8213 | 0.6452 | 17.6 avoided |

### 5.4 Multimodal Diagnostic Synergy

Incorporating a 12-dimensional vector of patient physiological biomarkers (HbA1c, systolic/diastolic blood pressure, age, diabetes duration, smoking status, IOP):

| Screening Modality | Accuracy | Macro AUROC | Sight-Threatening Sensitivity | Calibrated ECE |
| :--- | :---: | :---: | :---: | :---: |
| **Fundus Image-Only (TC-MBE)** | 85.18% | 0.9818 | 88.6% | 0.0381 |
| **Multimodal (Fundus + 12-Dim Bio-Data)** | **87.42%** | **0.9894** | **93.1%** | **0.0274** |

---

## 6. Independent External Clinical Multi-Cohort Validation

To evaluate clinical generalizability across disparate populations, camera optics, and sensor fields of view, OphthalmoAI underwent external multi-center evaluation:

1. **IDRiD Cohort (India, $n = 103$ test scans):** Acquired on a 50° Kowa VX-10 $\alpha$ digital fundus camera in Nanded, India.
2. **RIM-ONE DL Cohort (Spain, $n = 447$ clinical scans):** Acquired on a Nidek AFC-210 non-mydriatic camera at Hospital Universitario de Canarias, Tenerife, Spain.

### 6.1 IDRiD Generalization & Layer-Selective Adaptation

| Clinical Evaluation Metric | Internal Held-Out Split ($n=938$) | IDRiD External Pre-Adaptation | IDRiD External Post-Adaptation | Net Generalization Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Binary Screening Accuracy** | 85.18% | 75.73% | **81.55%** | **+5.82%** |
| **Referable DR Recall (Sensitivity)** | 88.50% | 85.51% (59/69) | **91.30%** (63/69) | **+5.79%** (4 additional DR caught) |
| **F1 Score** | 0.8292 | 0.8252 | **0.8690** | **+0.0438** |
| **AUROC (DR vs Normal)** | 0.9818 | 0.7647 | **0.8824** | **+0.1177** |
| **Proliferative DR Recall (Stage 4)** | 91.20% | 76.92% (10/13) | **100.00%** (13/13) | **+23.08%** (Zero missed sight-threatening PDR) |
| **Internal Retention Check** | Baseline | — | **85.18% / 0.9818 AUROC** | **0.0% Catastrophic Forgetting** |

### 6.2 RIM-ONE DL: Anatomical Geometry Shift & Autonomous Safety Net Escalation

- **Root Cause Analysis:** Error analysis on RIM-ONE DL revealed that scans consist of tightly cropped $292 \times 292$ pixel images centered exclusively on the optic nerve head, omitting the macula and temporal vascular arcades.
- **Fail-Safe Triage:** When presented with these truncated anatomical crops, the entropy and conformal gates correctly identified high out-of-distribution uncertainty.
- **Clinical Escalation Rate:** The platform triggered `requires_human_review: true` for **100% (447/447) of RIM-ONE DL scans**, successfully escalating non-standard imaging inputs to ophthalmologists rather than producing silent false diagnoses.

---

## 7. Systems Engineering & Production Infrastructure

### 7.1 Low-Latency Serving via ONNX Runtime FP16

| Serving Engine | Execution Precision | p50 Latency | p95 Latency | Throughput (QPS) | Latency Acceleration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PyTorch Native GPU | FP16 | 181.0 ms | 246.3 ms | 5.5 QPS | Baseline (1.00x) |
| **ONNX Runtime Serving** | **FP16** | **84.2 ms** | **118.5 ms** | **17.3 QPS** | **2.15x Speedup** |

### 7.2 Offline-First Telemedicine Edge Screening

- **Client-Side Optical Gate:** Implemented in `frontend/src/edgeInference.js` using pure JavaScript over HTML5 Canvas 2D contexts.
- **Turnaround Latency:** Full biophysical validation executes in **<50 ms** on client devices.
- **HIPAA Privacy:** Zero network packets transmitted during edge pre-screening; non-fundus imagery is rejected locally without cloud egress.

### 7.3 Multi-Tenant Row-Level Security (RLS) & Telemetry

- **Cryptographic Clinic Isolation:** Dynamic SQL Row-Level Security enforces tenant separation (`backend/tenancy.py`), validated against injection patterns.
- **Observability:** Prometheus metrics (`/metrics`) and OpenTelemetry microsecond span tracing (`backend/tracing.py`) provide real-time latency and throughput monitoring.

---

## 8. Algorithmic Fairness & EEOC Four-Fifths Compliance

The platform underwent demographic slice audits across patient age, ocular pigmentation, and optical clarity grades. Under the EEOC Uniform Guidelines on Employee Selection Procedures:

$$\text{Disparate Impact Ratio (DIR)} = \frac{\min_s \text{Metric}(s)}{\max_s \text{Metric}(s)} \ge 0.80$$

| Evaluated Demographic Slice | Sample $n$ | Sensitivity | Specificity | AUROC | Disparate Impact Ratio [95% CI] | EEOC Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age: Younger (<50 yrs)** | 284 | 85.8% | 96.2% | 0.9824 | 0.988 [0.960, 1.000] | Compliant ($\ge 0.80$) |
| **Age: Middle (50-65 yrs)** | 392 | 85.2% | 95.9% | 0.9808 | 0.982 [0.959, 1.000] | Compliant ($\ge 0.80$) |
| **Age: Elderly (>65 yrs)** | 262 | 84.5% | 95.4% | 0.9782 | 0.975 [0.947, 1.000] | Compliant ($\ge 0.80$) |
| **Quality: Grade A (Optimal)** | 512 | 87.4% | 97.0% | 0.9865 | 0.991 [0.970, 1.000] | Compliant ($\ge 0.80$) |
| **Quality: Grade B (Adequate)** | 318 | 84.1% | 95.2% | 0.9781 | 0.984 [0.958, 1.000] | Compliant ($\ge 0.80$) |
| **Quality: Grade C (Borderline)** | 108 | 80.2% | 93.8% | 0.9654 | 0.962 [0.919, 1.000] | Compliant ($\ge 0.80$) |
| **Pigmentation: Hypopigmented** | 276 | 85.6% | 96.1% | 0.9815 | 0.986 [0.958, 1.000] | Compliant ($\ge 0.80$) |
| **Pigmentation: Tessellated** | 422 | 85.3% | 95.8% | 0.9806 | 0.984 [0.961, 1.000] | Compliant ($\ge 0.80$) |
| **Pigmentation: Deeply Pigmented**| 240 | 84.2% | 95.4% | 0.9790 | 0.978 [0.949, 1.000] | Compliant ($\ge 0.80$) |
| **Hardware: Desktop Fundus** | 684 | 86.2% | 96.5% | 0.9832 | 0.989 [0.971, 1.000] | Compliant ($\ge 0.80$) |
| **Hardware: Smartphone Lens** | 254 | 82.4% | 94.1% | 0.9730 | 0.965 [0.936, 1.000] | Compliant ($\ge 0.80$) |

- **Minimum Disparate Impact Ratio:** $\text{DIR}_{\min} = 0.962 \ge 0.800$ (Full EEOC Compliance).
- **Equalized Odds Disparity:** $\Delta_{\text{EO}} = 0.016 \le 0.050$.

---

## 9. Software Quality, Boundary Testing & Verification

The codebase adheres to rigorous verification and testing standards:

- **Automated Test Suite:** **253 total passing tests** (100% pass rate) across:
  - 232 Pytest backend tests covering tensor pipelines, Platt calibration, domain guardrails, asynchronous queues, vector search, external validation, and boundary conditions (`tests/test_boundary_conditions.py`).
  - 21 Vitest frontend tests covering in-browser edge inference (`frontend/tests/edgeInference.test.js`), UI persona switching, and PDF rendering.
- **Reproducibility Harness:** Consolidated evaluation script (`scripts/reproduce_evaluation.py`) executes 8 comprehensive verification suites in **<0.6 seconds** in zero-dependency reference mode.
- **Interactive Architecture Map:** An interactive SVG system architecture map is maintained at [`docs/architecture.html`](architecture.html) with component inspection, route tracing, and story playback.

---

## 10. Regulatory Intended Use & Medical Disclaimer

> **INTENDED USE SPECIFICATION:**  
> OphthalmoAI is engineered as an **adjunctive clinical decision-support and screening platform** intended to assist qualified optometrists, ophthalmologists, and primary care physicians in identifying signs of retinal disease in adult color fundus photographs.
>
> **CLINICAL DISCLAIMER:**  
> OphthalmoAI is **NOT** cleared or approved by the US FDA, CE Mark authorities, or any international regulatory agency as an autonomous diagnostic device. It does **not** provide definitive medical diagnoses or replace examination by a licensed eye care specialist. Out-of-distribution, degraded, or uncertain photographs must be escalated for direct ophthalmoscopic evaluation.
