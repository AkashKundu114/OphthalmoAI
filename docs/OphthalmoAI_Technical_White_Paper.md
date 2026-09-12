# OphthalmoAI: A Calibrated Tri-Backbone Deep Vision Ensemble for Point-of-Care Retinal Disease Screening

**Technical White Paper & Engineering Specification**  
*Author: Akash Kundu*  
*Affiliation: OphthalmoAI Research Initiative*  

---

## 1. Abstract & Clinical Context
Ophthalmic screening of the posterior pole is critical for preventing irreversible blindness from conditions such as Diabetic Retinopathy, Glaucoma, and Age-related Macular Degeneration. However, automated screening tools frequently suffer from calibration drift, model overconfidence, and lack of visual explainability.

**OphthalmoAI** addresses these barriers through a robust, calibrated tri-backbone soft-voting ensemble paired with a dedicated explainability engine:
1. **Tri-Backbone Vision Architecture**: Combines DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M for complementary feature extraction across retinal microvasculature, optic disc geometry, and macular pigmentary changes.
2. **Platt Temperature Calibration**: Corrects neural network overconfidence via learned post-hoc temperature scaling ($T \in [1.06, 1.34]$), reducing Expected Calibration Error (ECE) to $0.0644$.
3. **Pixel-Aligned Explainability**: EfficientNet-B4 generates high-resolution Grad-CAM saliency heatmaps highlighting pathognomonic lesions.
4. **Empirical Performance**: On a held-out test split of 938 color fundus photographs, OphthalmoAI achieves **85.18% top-1 accuracy**, a **0.9805 Macro AUROC**, and a **0.8292 Macro F1 score**.

---

## 2. Target Pathology & Diagnostic Taxonomy

The model classifies color fundus imagery across six gold-standard categories:
- **Normal (Healthy Fundus)**: Uniform background, sharp foveal avascular zone, intact neuroretinal rim.
- **Diabetic Retinopathy (DR)**: Microaneurysms, dot/blot hemorrhages, hard exudates (ICD-10: E11.319).
- **Glaucoma**: Pathological cup-to-disc ratio expansion, neuroretinal notch formation (ICD-10: H40.9).
- **Cataract (Media Opacity)**: Lens clouding creating diffuse fundus haziness and vascular obscuration (ICD-10: H25.9).
- **Age-related Macular Degeneration (AMD)**: Soft/hard drusen, retinal pigment epithelium detachment (ICD-10: H35.30).
- **Hypertensive Retinopathy / Pathological Myopia**: Arteriolar attenuation, copper-wiring, posterior staphyloma (ICD-10: H35.00).

---

## 3. Mathematical Methodology

### 3.1 Temperature-Calibrated Soft-Voting Ensemble
Given an input fundus image $X$, each backbone $m \in \{1, 2, 3\}$ outputs raw logit vector $z_m(X) \in \mathbb{R}^6$. Logits are scaled by their model-specific calibration temperature $T_m > 0$ before computing softmax probabilities:
$$p_m(y = c \mid X) = \frac{\exp(z_{m, c} / T_m)}{\sum_{j=1}^6 \exp(z_{m, j} / T_m)}$$
The final ensemble prediction is obtained via soft-voting probability averaging:
$$P_{\text{ensemble}}(y = c \mid X) = \frac{1}{3} \sum_{m=1}^3 p_m(y = c \mid X)$$

### 3.2 Visual Interpretability via Grad-CAM
To guarantee clinician verifiability, the dedicated EfficientNet-B4 backbone computes gradient-weighted class activation mappings:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right), \quad \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{i, j}^k}$$
The activation map is upsampled to $384 \times 384$, colored via the viridis color spectrum, and embedded into clinical PDF reports.

---

## 4. Empirical Evaluation & Hardware Benchmarks

### 4.1 Diagnostic Performance ($n = 938$ Test Split)
- **Overall Accuracy**: **85.18%**
- **Macro AUROC**: **0.9805**
- **Macro F1 Score**: **0.8292**
- **Expected Calibration Error (ECE)**: **0.0644**

### 4.2 Hardware Telemetry on NVIDIA RTX 5060 Laptop GPU
- **Inference Throughput**: $< 35\text{ ms}$ per multi-model forward pass.
- **VRAM Utilization**: $\approx 1.8\text{ GB}$ total dedicated GPU memory for all 4 models.
- **Thermal Performance**: Under $78^\circ\text{C}$ across full test runs.
