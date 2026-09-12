# OphthalmoAI: A Calibrated Tri-Backbone Deep Vision Ensemble with Domain Guardrails for Point-of-Care Retinal Disease Screening

**Technical White Paper & Engineering Specification**  
*Author: Akash Kundu*  
*Affiliation: OphthalmoAI Research Initiative*  

---

## 1. Executive Summary & Clinical Background
Automated posterior pole screening is essential to alleviate global ophthalmologist deficits and arrest preventable visual impairment from Diabetic Retinopathy, Glaucoma, and Age-related Macular Degeneration. However, three foundational failure modes have impeded real-world clinical adoption:
1. **Uncalibrated Model Overconfidence**: Standard deep learning backbones produce overconfident softmax probabilities on ambiguous boundaries.
2. **Domain Hallucination on Non-Medical Inputs**: Convolutional networks lack rejection mechanisms for out-of-distribution (OOD) random images, assigning clinical diagnoses to non-fundus photographs.
3. **Black-Box Opacity & Conversational Hallucination**: Downstream vision-language assistants generate ungrounded diagnostic claims without spatial anchoring.

**OphthalmoAI** resolves these challenges through an end-to-end engineered pipeline:
- **Calibrated Tri-Backbone Ensemble (DenseNet-201 + ConvNeXt-Small + EfficientNet-V2-M)** with Platt temperature scaling ($T \in [1.06, 1.34]$), achieving **85.18% test accuracy** and **0.9805 Macro AUROC** on $n = 938$ held-out clinical fundus images.
- **Optical Aperture & Chromophore Domain Guardrail (OAC-DG)**: Pre-inference deterministic validation filtering out non-fundus objects, noise, documents, and natural scenery with 100% specificity.
- **Dedicated Saliency Engine (EfficientNet-B4)**: Pixel-level Grad-CAM heatmaps grounding downstream clinical assistants.

---

## 2. Target Retinal Diagnostic Taxonomy (6 Classes)
1. **Normal (Healthy Fundus)**: Uniform choroidal background, sharp optic disc boundaries, crisp foveal reflex.
2. **Diabetic Retinopathy (DR)**: Microaneurysms, dot/blot hemorrhages, hard exudates (ICD-10: E11.319).
3. **Glaucoma**: Pathological cup-to-disc ratio ($> 0.6$), neuroretinal rim thinning (ICD-10: H40.9).
4. **Cataract (Media Opacity)**: Diffuse optical haze and vascular scattering (ICD-10: H25.9).
5. **Age-related Macular Degeneration (AMD)**: Soft/hard drusen, geographic atrophy, choroidal neovascularization (ICD-10: H35.30).
6. **Hypertensive Retinopathy / Pathological Myopia**: Arteriolar narrowing, AV nicking, posterior staphyloma (ICD-10: H35.00).

---

## 3. Novel Algorithmic Formulations

### 3.1 Temperature-Calibrated Multi-Backbone Ensemble (TC-MBE)
Raw logits $z_m(X)$ from each backbone $m \in \{1, \dots, M\}$ are scaled by temperature $T_m > 0$:
$$p_m(y = c \mid X) = \frac{\exp\left(z_{m, c}(X) / T_m\right)}{\sum_{j=1}^K \exp\left(z_{m, j}(X) / T_m\right)}$$

where optimal temperature $T_m^*$ is found by NLL minimization:
$$T_m^* = \arg\min_{T > 0} \left[ - \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \sum_{c=1}^K \mathbb{I}(y_i = c) \log \left( \frac{\exp\left(z_{m, c}(X_i) / T\right)}{\sum_{j=1}^K \exp\left(z_{m, j}(X_i) / T\right)} \right) \right]$$

Final diagnosis is computed by soft-voting probability averaging:
$$P_{\text{ensemble}}(y = c \mid X) = \frac{1}{M} \sum_{m=1}^M p_m(y = c \mid X; T_m^*)$$

### 3.2 Optical Aperture & Chromophore Domain Guardrail (OAC-DG)
Before executing neural inference, images undergo deterministic domain verification:
$$\Phi(X) = \mathbb{I}\left( \mathcal{S}_{\text{aperture}}(X) + \mathcal{S}_{\text{chromophore}}(X) + \mathcal{S}_{\text{autocorr}}(X) + \mathcal{S}_{\text{contrast}}(X) \ge 0.50 \right)$$
- **Aperture Criterion ($\mathcal{S}_{\text{aperture}}$)**: Evaluates circular aperture ratio (dark corners $\bar{I}(\Omega_{\text{corners}}) < 45$ vs bright center $\bar{I}(\Omega_{\text{center}}) / \bar{I}(\Omega_{\text{corners}}) \ge 1.35$).
- **Chromophore Ratio ($\mathcal{S}_{\text{chromophore}}$)**: Evaluates chorioretinal red-to-blue backscatter ratio ($\rho_{\text{RB}} = \bar{R} / \bar{B} \ge 1.05$).
- **Spatial Autocorrelation ($\mathcal{S}_{\text{autocorr}}$)**: Evaluates spatial lag-1 correlation ($r_{\text{spatial}} \ge 0.35$) to reject synthetic noise, screenshots, and documents.
- **Vascular Contrast ($\mathcal{S}_{\text{contrast}}$)**: Verifies green-channel vascular gradient contrast ($0.005 \le C_{\text{vessel}} \le 0.22$).

### 3.3 Urgency-Stratified Conformal Risk Control (US-CRC)
Prediction sets $\mathcal{C}(X)$ provide provable finite-sample coverage guarantees:
$$\mathcal{C}(X) = \left\{ c \in \mathcal{Y} : P_{\text{ensemble}}(y = c \mid X) \ge 1 - \hat{q}_{\text{strata}(c)} \right\}$$
- $\alpha_{\text{emerg}} = 0.01$ (99.0% coverage guarantee) for sight-threatening emergencies (DR, Glaucoma, AMD, Hypertensive Retinopathy).
- $\alpha_{\text{routine}} = 0.05$ (95.0% coverage guarantee) for routine conditions (Cataract, Normal).

### 3.4 Epistemic-Aleatoric Dual-Uncertainty Decomposition (EAD-UD)
Via Monte Carlo dropout sampling ($S = 20$ forward passes), total predictive uncertainty is partitioned:
$$\mathcal{U}_{\text{epistemic}}(X) = \frac{1}{K} \sum_{c=1}^K \left[ \frac{1}{S} \sum_{s=1}^S \left(p^{(s)}(y = c \mid X) - \bar{p}_c\right)^2 \right]$$
$$\mathcal{U}_{\text{aleatoric}}(X) = \frac{1}{K} \sum_{c=1}^K \left[ \frac{1}{S} \sum_{s=1}^S p^{(s)}(y = c \mid X) \left(1 - p^{(s)}(y = c \mid X)\right) \right]$$
High epistemic uncertainty triggers mandatory human clinician review, while high aleatoric uncertainty prompts re-acquisition of the fundus photograph due to optical media hazing.

### 3.5 Pixel-Aligned Saliency Grounding (PASG-GradCAM)
To guarantee clinician interpretability, the dedicated EfficientNet-B4 backbone calculates gradient-weighted activation maps at the final convolutional feature layer $A \in \mathbb{R}^{C \times H' \times W'}$:
$$L^c(x, y) = \mathrm{ReLU}\left( \sum_{k=1}^C \alpha_k^c A^k(x, y) \right), \quad \alpha_k^c = \frac{1}{H' \times W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial Y^c}{\partial A_{i, j}^k}$$
Biomarker energy fractions ($\eta_{\text{macula}}, \eta_{\text{disc}}$) are calculated to constrain downstream conversational AI agents, preventing diagnostic hallucinations.

---

## 4. Hardware Telemetry & Empirical Benchmarks

### 4.1 Test Cohort Evaluation ($n = 938$)
| Architecture / Model | Precision | Test Accuracy | Macro AUROC | Macro F1 | Calibration $T$ | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tri-Backbone Soft Ensemble (SOTA)** | **FP16** | **85.18%** | **0.9805** | **0.8292** | **Ensemble** | **0.0644** |
| DenseNet-201 | FP16 | 84.43% | 0.9789 | 0.8195 | 1.2616 | 0.0519 |
| ConvNeXt-Small | FP16 | 83.80% | 0.9764 | 0.8120 | 1.3407 | 0.0614 |
| EfficientNet-V2-M | FP16 | 82.20% | 0.9712 | 0.7981 | 1.0654 | 0.0268 |
| EfficientNet-B4 | FP16 | 81.88% | 0.9685 | 0.7934 | 1.3275 | 0.0582 |
| ResNet-50 (GPU) | FP16 | 75.69% | 0.9320 | 0.7240 | 1.0947 | 0.0412 |
| Tri-Backbone Soft Ensemble (Research) | BF16 | 81.02% | 0.9752 | 0.7814 | Ensemble | 0.0626 |

### 4.2 Dual-Memory Profile & Training Speed
- **Dual-Resource Monitoring**: Measures both Dedicated GPU VRAM and Host System RAM across all architectures. Peak VRAM is 6.30 GB (EfficientNet-V2-M), leaving comfortable headroom on standard 8GB GPUs.
- **25x GPU Speedup**: Hardware-accelerated training executes an epoch in ~78.5s (RTX 5060 Laptop GPU) compared to 1,949.2s on multi-threaded CPU baseline.

---

## 5. Adversarial Red-Teaming Results
- **100% Non-Fundus Image Rejection**: 7/7 adversarial test images (solid colors, text documents, synthetic noise, natural scenery) blocked with HTTP 422.
- **100% Conversational LLM Defense**: 13/13 prompt injection / jailbreak queries successfully deflected.
