# OphthalmoAI Known Limitations & Domain Boundary Analysis

**Status:** Technical & Clinical Boundary Specification  
**Reference Model:** OphthalmoAI v0.1.0-alpha

---

## 1. Domain Shift & Hardware Sensor Variations

1. **Camera Optical Variations:**
   - The model was trained predominantly on datasets acquired via desktop mydriatic fundus cameras (Canon CR-2, Topcon TRC-NW8, Zeiss Visucam).
   - Handheld smartphone lens attachments (e.g., Volk iNview, D-Eye) produce distinct vignetting, illumination hotspots, and chromatic aberrations that can degrade model confidence.
2. **Pigmentation & Fundus Background Heterogeneity:**
   - Retinal pigmentation varies significantly across ethnicities (e.g., blonde fundi in light-pigmented individuals vs. tigroid / highly melanated fundi in South Asian and African cohorts).
   - Background choroidal visibility may trigger false positive drusen alerts if not properly normalized.

---

## 2. Pathological & Optical Boundary Cases

1. **Optical Media Opacity (Dense Cataract / Corneal Scarring):**
   - Severe mature cataracts cause extensive light scattering and loss of high-frequency microvascular details. The model may correctly classify "Cataract" while failing to observe underlying subtle diabetic microaneurysms obscured by the media opacity.
2. **Co-Occurring Retinal Conditions:**
   - The multi-class softmax classifier assigns normalized probabilities across mutually exclusive categories. In patients with concurrent high-grade diabetic retinopathy and wet macular degeneration, the system may present elevated epistemic uncertainty ($u(X) > 0.15$) and must be reviewed by a human retina specialist.
3. **Peripheral Retinal Pathology:**
   - Standard $45^\circ$ fundus photography covers the posterior pole. Peripheral retinal tears, lattice degeneration, and peripheral capillary non-perfusion cannot be evaluated without ultra-widefield imaging ($200^\circ$ Optos), which is out of scope.

---

## 3. Explainability Caveat (Grad-CAM Interpretation)

- **Explanatory Aid Only:** Grad-CAM heatmaps highlight regions of the convolutional feature map that contributed positively to the network's class logit.
- **Not Localization Proof:** Heatmaps must **never** be interpreted as verified anatomical lesion segmentations or proof of morphological extent. Clinicians must independently inspect the un-overlayed raw photograph.
