# OphthalmoAI: Clinical Evaluation, Validation & Safety Framework

**Status**: Clinical Research & Engineering Evaluation  
**Target Domain**: Color Fundus Photography & Retinal Disease Screening  
**Reference**: OphthalmoAI Multi-Backbone Calibrated Ensemble Pipeline  

---

## 1. Intended Use & Clinical Scope

### 1.1 Purpose
OphthalmoAI is an artificial intelligence-assisted clinical decision-support software designed to analyze digital color fundus photographs of the posterior pole. It classifies retinal imagery across **six primary diagnostic categories**:
1. **Normal (Healthy Fundus)** — Clear retina, intact neuroretinal rim, crisp foveal reflex.
2. **Diabetic Retinopathy (DR)** — Microaneurysms, blot hemorrhages, hard exudates, neovascularization (ICD-10: E11.319).
3. **Glaucoma** — Pathological cup-to-disc ratio ($>0.6$), vertical cup elongation, neuroretinal rim thinning (ICD-10: H40.9).
4. **Cataract (Media Opacity)** — Diffuse optical scattering and vascular attenuation visible on fundus view (ICD-10: H25.9).
5. **Age-related Macular Degeneration (AMD)** — Drusen accumulation, geographic atrophy, choroidal neovascularization (ICD-10: H35.30).
6. **Hypertensive Retinopathy / Pathological Myopia** — Arteriolar attenuation, copper/silver wiring, AV crossing changes, posterior staphyloma (ICD-10: H35.00).

### 1.2 Clinical Role: Decision Support, Not Autonomous Diagnosis
- OphthalmoAI functions strictly as an **adjunctive screening and triage tool** to prioritize patient backlogs in community health, optometric screenings, and telehealth clinics.
- It does **not** replace full clinical examinations by licensed ophthalmologists (slit-lamp biomicroscopy, optical coherence tomography, or visual field testing).

---

## 2. Empirical Clinical Validation & Benchmark Results

### 2.1 Held-Out Test Set Evaluation ($n = 938$)
All models were benchmarked on a strictly segregated, held-out empirical test split of 938 verified color fundus scans across the 6 target classes.

| Architecture / Model | Precision | Test Accuracy | Macro AUROC | Macro F1 | Calibration $T$ | Calibrated ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Tri-Backbone Soft Ensemble (SOTA)** | **FP16** | **85.18%** | **0.9818** | **0.8292** | **Ensemble** | **0.0644** |
| DenseNet-201 | FP16 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0519 |
| ConvNeXt-Small | FP16 | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0614 |
| EfficientNet-V2-M | FP16 | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0268 |
| EfficientNet-B4 (Grad-CAM Saliency Engine) | FP16 | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0582 |
| ResNet-50 (Baseline) | FP16 | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0412 |
| Tri-Backbone Soft Ensemble (Research) | BF16 | 81.02% | 0.9752 | 0.7814 | Ensemble | 0.0626 |

### 2.2 Per-Class Sensitivity & Specificity (Ensemble, $n = 938$)
- **Normal**: Sensitivity 83.6% | Specificity 91.7% | AUROC 0.9597 | AUPRC 0.8898 | Support $n=225$
- **Diabetic Retinopathy**: Sensitivity 80.9% | Specificity 96.6% | AUROC 0.9717 | AUPRC 0.9332 | Support $n=225$
- **Glaucoma**: Sensitivity 91.2% | Specificity 96.1% | AUROC 0.9855 | AUPRC 0.9591 | Support $n=194$
- **Cataract**: Sensitivity 93.5% | Specificity 98.1% | AUROC 0.9958 | AUPRC 0.9859 | Support $n=200$
- **Age-related Macular Degeneration (AMD)**: Sensitivity 77.5% | Specificity 98.8% | AUROC 0.9912 | AUPRC 0.8846 | Support $n=40$
- **Hypertensive Retinopathy / Pathological Myopia**: Sensitivity 63.0% | Specificity 99.5% | AUROC 0.9865 | AUPRC 0.8666 | Support $n=54$
- **Macro Average**: Sensitivity 81.6% | Specificity 96.8% | Macro AUROC 0.9818 | Total Support $n=938$

---

## 3. Calibrated Uncertainty & Risk Control

### 3.1 Platt Temperature Scaling
Modern neural networks often exhibit overconfidence. OphthalmoAI applies post-hoc temperature scaling ($T > 0$) to align output probabilities with empirical accuracy:
$$\hat{p}_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

Temperatures were tuned via negative log-likelihood minimization on the validation set, successfully decreasing Expected Calibration Error across all models down to $0.0268 - 0.0644$.

### 3.2 Urgency-Stratified Conformal Risk Control & Human Review
The system automatically assigns a `requires_human_review: true` flag when:
1. Top calibrated confidence is $< 75\%$.
2. The conformal prediction set contains multiple candidate classes ($|\mathcal{C}(X)| > 1$).
3. Epistemic uncertainty exceeds clinical triage threshold ($\mathcal{U}_{\text{epistemic}} \ge 0.15$).
4. The image fails pre-inference Image Quality Assessment (IQA blur score $< 100$ or extreme brightness anomalies).
5. Clinician override is submitted via `POST /scans/{id}/override`.

---

## 4. Optical Aperture & Chromophore Domain Guardrails (OAC-DG)
To eliminate catastrophic false positives on out-of-distribution imagery (everyday snapshots, pets, documents, noise), OphthalmoAI executes physical domain checks prior to neural inference:
- **Aperture Circularity**: Asserts optical vignette mask with dark corner margins.
- **Chorioretinal Chromophore Ratio**: Validates red-to-blue backscatter ($\bar{R}/\bar{B} \ge 1.05$) reflecting hemoglobin and melanin pigments.
- **Spatial Autocorrelation**: Evaluates lag-1 correlation ($r_{\text{spatial}} \ge 0.35$) to reject synthetic noise and text screenshots.
- **Rejection Outcome**: Non-fundus uploads are rejected deterministically with HTTP 422, blocking non-medical images from clinical inference.

---

## 5. Explainability & Clinician Verification (Grad-CAM)
OphthalmoAI pairs every prediction with an interpretable Class Activation Map generated via **EfficientNet-B4**:
- High-intensity saliency areas correspond directly to pathological features: microaneurysms in DR, optic disc cupping in Glaucoma, and drusen in AMD.
- Spatial biomarker energy ratios ($\eta_{\text{macula}}, \eta_{\text{disc}}$) quantify anatomical grounding.
- Both original fundus imagery and Grad-CAM overlays are embedded side-by-side in modern, exportable PDF clinical reports.

---

## 6. Independent Multi-Cohort External Clinical Validation (v2.6)

Following FDA SaMD and Nature Medicine guidelines for external generalizability, OphthalmoAI was evaluated against two independent external clinical cohorts:

1. **IDRiD Cohort (India, $n = 103$ test scans, Kowa VX-10 $\alpha$ camera)**:
   - Evaluates multi-center generalizability on Indian diabetic patient populations.
   - Post-adaptation accuracy surged from 75.73% to **81.55%**, achieving **91.30% Referable DR Sensitivity** (catching 63 of 69 cases) and **100% detection of Proliferative DR** (13 of 13 cases).
   - Zero regression observed on the internal test split ($n = 938$, accuracy 85.18%, Macro AUROC 0.9818).

2. **RIM-ONE DL Cohort (Spain, $n = 447$ clinical scans, Nidek AFC-210 camera)**:
   - Uncovered critical anatomical **Field-of-View (FOV) Mismatch**: RIM-ONE DL consists of tight 292×292 optic disc region-of-interest crops lacking the macula and temporal vascular arcades.
   - **Clinical Safety Net Verification**: Rather than issuing ungrounded autonomous predictions, OphthalmoAI's predictive entropy gate escalated **100% of RIM-ONE DL scans** to clinician review (`requires_human_review: true`), demonstrating robust fail-safe behavior on non-canonical imaging inputs.

For comprehensive clinical protocols, Wilson score confidence intervals, and ICDR severity breakdowns, consult the dedicated [External Clinical Validation Report](EXTERNAL_VALIDATION_REPORT.md).
