# OphthalmoAI Intended Use & Clinical Scope

**Status:** Research & Educational Screening Prototype  
**Classification:** Non-Diagnostic Clinical Decision Support Aid  
**Target Users:** Ophthalmologists, Optometrists, Medical Researchers, Public Health Investigators

---

## 1. Intended Purpose

OphthalmoAI is designed to assist eye care professionals and biomedical researchers in analyzing digital color fundus photographs for signs of five common retinal disease patterns:
1. **Normal Posterior Pole:** Absence of observable microvascular abnormalities, drusen, or glaucomatous cupping.
2. **Diabetic Retinopathy (DR):** Presence of microaneurysms, intraretinal hemorrhages, hard exudates, or neovascularization.
3. **Glaucoma:** Structural optic nerve head alterations, increased cup-to-disc ratio (CDR $> 0.65$), or localized neuroretinal rim thinning.
4. **Cataract:** Optical media opacity impacting fundus illumination and obscuring retinal vascular landmarks.
5. **Age-Related Macular Degeneration (AMD):** Macular drusen, retinal pigment epithelium (RPE) hyperpigmentation, or choroidal neovascularization.

---

## 2. Operating Environment

- **Input Modality:** High-resolution digital color fundus photographs ($384\times384$ or higher resolution) centered on the posterior pole (macula and optic disc) captured with standard desktop fundus cameras ($45^\circ - 50^\circ$ Field of View).
- **Patient Population:** Adult individuals ($\ge 18$ years) presenting for scheduled screening or research observation.
- **Workflow Role:** Pre-screening triaging aid to highlight suspicious scans, quantify epistemic uncertainty, provide Grad-CAM explainability heatmaps, and prioritize urgent cases for human specialist evaluation.

---

## 3. Explicit Out-of-Scope Uses (Contraindications)

The following scenarios are strictly outside OphthalmoAI's validation domain and must **NOT** be attempted:
1. **Autonomous Clinical Diagnosis:** OphthalmoAI must never render final medical diagnoses or initiate pharmacologic / surgical therapy without direct physician oversight.
2. **Acute Emergency Trauma Triage:** Open globe injuries, chemical eye splashes, intraocular foreign bodies, and acute angle-closure crises must be directed immediately to an emergency department.
3. **Pediatric Retinopathy Screening:** Retinopathy of Prematurity (ROP) and pediatric retinoblastoma require specialized pediatric indirect ophthalmoscopy and are outside the model's training distribution.
4. **Non-Fundus Photography:** External eye selfies, anterior segment slit-lamp photos, corneal topography, optical coherence tomography (OCT), or ultrasound B-scans will be rejected by the domain validator.
