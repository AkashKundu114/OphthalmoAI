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

### 2.1 Held-Out Test Set Evaluation ( = 938$)
All models were benchmarked on a strictly segregated, held-out empirical test split of 938 verified color fundus scans.

| Architecture / Model | Test Accuracy | Macro AUROC | Macro F1 | Calibration $ | Calibrated ECE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Tri-Backbone Soft Ensemble (SOTA)** | **85.18%** | **0.9805** | **0.8292** | **Ensemble** | **0.0644** |
| DenseNet-201 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0519 |
| ConvNeXt-Small | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0614 |
| EfficientNet-V2-M | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0268 |
| EfficientNet-B4 (Grad-CAM Saliency Engine) | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0582 |
| ResNet-50 (Baseline) | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0412 |

### 2.2 Per-Class Sensitivity & Specificity (Ensemble)
- **Normal**: Sensitivity 89.2% | Specificity 94.5%
- **Diabetic Retinopathy**: Sensitivity 88.5% | Specificity 95.8%
- **Glaucoma**: Sensitivity 82.1% | Specificity 96.2%
- **Cataract**: Sensitivity 86.4% | Specificity 97.1%
- **AMD**: Sensitivity 83.7% | Specificity 96.5%
- **Hypertensive Retinopathy / Myopia**: Sensitivity 81.1% | Specificity 95.9%

---

## 3. Calibrated Uncertainty & Risk Control

### 3.1 Platt Temperature Scaling
Modern neural networks often exhibit overconfidence. OphthalmoAI applies post-hoc temperature scaling ( > 0$) to align output probabilities with empirical accuracy:
\hat{p}_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}
Temperatures were tuned via negative log-likelihood minimization on the validation set, successfully decreasing Expected Calibration Error across all models down to .0268 - 0.0644$.

### 3.2 Conformal Prediction & Human-in-the-Loop Review
The system automatically assigns a 
equires_human_review: true flag when:
1. Top calibrated confidence is $< 75\%$.
2. The conformal prediction set contains multiple candidate classes ($|\mathcal{C}(X)| > 1$).
3. The image fails pre-inference Image Quality Assessment (IQA blur score $< 100$ or extreme brightness anomalies).
4. Clinician override is submitted via POST /scans/{id}/override.

---

## 4. Explainability & Clinician Verification (Grad-CAM)
OphthalmoAI pairs every prediction with an interpretable Class Activation Map generated via **EfficientNet-B4**:
- High-intensity saliency areas correspond directly to pathological features: microaneurysms in DR, optic disc cupping in Glaucoma, and drusen in AMD.
- Both original fundus imagery and Grad-CAM overlays are embedded side-by-side in modern, exportable PDF clinical reports.
