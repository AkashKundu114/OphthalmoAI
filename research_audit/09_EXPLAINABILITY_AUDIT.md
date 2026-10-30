# Phase 8 Audit Report: Explainability, Saliency Grounding & Clinical Attribution

**Document Reference:** `research_audit/09_EXPLAINABILITY_AUDIT.md`  
**Auditor:** Senior Clinical AI Safety, Ophthalmic Imaging, and Medical AI Auditor  
**Date of Audit:** October 30, 2026  
**Repository Targets:**  
- Engineering Repository: `D:\OphthalmoAI` (`backend/services/explanation_service.py`, `backend/biomarker_extractor.py`, `backend/vector_search.py`, `scripts/evaluate_extended_clinical_battery.py`)  
- Paper Repositories: `D:\OphthalmoAI\research\submissions\ieee\source\manuscript.tex`, `D:\OphthalmoAI-Paper\main.tex`  
**Audit Status:** COMPLETE  

---

## Executive Summary

Phase 8 evaluates the claims surrounding **Pixel-Aligned Saliency Grounding (PASG-GradCAM)**, anatomical biomarker energy ratios ($\eta_{\text{macula}}$, $\eta_{\text{disc}}$), landmark segmentation U-Nets, expert ophthalmologist audit fidelity, and **Content-Based Medical Image Retrieval (CBMIR)** using FAISS.

In the IEEE manuscript draft (`research/submissions/ieee/source/manuscript.tex`), Formulation 5 and Table VIII claim that OphthalmoAI achieves an **88.4% Pointing Game Accuracy**, an **Energy Ratio $\eta_{\text{macula}}$ of 62.8%**, and **$\eta_{\text{disc}}$ of 66.4%** through a lightweight U-Net trained on REFUGE/DRIVE and verified by an ophthalmologist (Dice $0.91 \pm 0.04$). Furthermore, it claims a penultimate 512-D EfficientNet-B4 FAISS index retrieving verified historical cases with multimodal outcomes in $<5\text{ ms}$.

Our forensic audit reveals that **none of these architectural claims exist in executable code**. The explainability subsystem relies on standard unconstrained Grad-CAM, randomly chosen spatial strings, an anterior-segment anatomical mismatch in `biomarker_extractor.py`, synthetic random vector generation in `vector_search.py`, and a 6-item hardcoded mock archive. In the revised `D:\OphthalmoAI-Paper\main.tex`, these ungrounded claims were largely excised, confirming they were non-reproducible assertions.

### Audit Verdict Summary Matrix

| Audit Target / Claim | Manuscript Claim | Codebase Reality | Evidence Label |
| :--- | :--- | :--- | :--- |
| **Service File Location** | `backend/services/gradcam_service.py` & `explainability_service.py` | Absent at specified paths; implemented in [`backend/services/explanation_service.py`](file:///d:/OphthalmoAI/backend/services/explanation_service.py#L1-L96) | **PARTIALLY VERIFIED** |
| **PASG-GradCAM Operator** | Pixel-aligned saliency constrained to anatomical masks | Standard vanilla Grad-CAM from `pytorch_grad_cam`; zero pixel alignment or mask gating | **CONTRADICTED** |
| **Energy Ratios $\eta_{\text{macula}}, \eta_{\text{disc}}$** | $\eta_{\text{macula}} = \frac{\iint_{\Omega_M} \tilde{L}}{\iint_{\Omega} \tilde{L}} \ge 0.40$, $\eta_{\text{disc}} \ge 0.40$ | **Zero implementation**. Neither metric is computed in any Python script; Table VIII is simulated | **CONTRADICTED / UNVERIFIED** |
| **Spatial Description Generation** | Saliency-grounded anatomical lesion localization | Uses `random.choice(quadrants)` from 5 hardcoded strings ([`explanation_service.py#L66-L95`](file:///d:/OphthalmoAI/backend/services/explanation_service.py#L66-L95)) | **INVALID / MOCKED** |
| **Anatomical Biomarker Extractor** | Quantitative retinal biomarker extraction | Detects iris/cornea/sclera via Hough circles on fundus photography ([`biomarker_extractor.py#L60-L117`](file:///d:/OphthalmoAI/backend/biomarker_extractor.py#L60-L117)) | **INVALID (ANATOMICAL MISMATCH)** |
| **Landmark Segmentation U-Net** | Lightweight U-Net trained on REFUGE & DRIVE | **No U-Net model exists** in `models/`; no segmentation scripts exist; no REFUGE/DRIVE weights | **CONTRADICTED** |
| **Ophthalmologist 10% Audit** | Expert ophthalmologist audit of 10% sample ($0.91 \pm 0.04$ Dice) | **Zero documentary evidence**, zero IRB, zero clinician names, zero segmentation ground truth | **UNVERIFIED / UNSUPPORTED** |
| **Lesion-Level Ground Truth Masks** | Diagnostic attribution grounded in microaneurysms, drusen | Datasets have only image-level categorical labels; **zero pixel lesion masks** | **CONTRADICTED** |
| **CBMIR Vector Search** | FAISS inverted file index on 512-D EfficientNet-B4 features | Pure NumPy dot product on 6 hardcoded mock cases; zero FAISS dependency ([`vector_search.py`](file:///d:/OphthalmoAI/backend/vector_search.py#L10-L194)) | **CONTRADICTED / SIMULATED** |
| **CBMIR Embeddings & Retrieval** | 512-D penultimate features from real multimodal cases | Embeddings are synthetic random Gaussians with static class offsets; queries blend class vectors | **CONTRADICTED / SIMULATED** |
| **Sub-5 ms Retrieval Latency** | Sub-5 ms case-based retrieval | Latency is $<1\text{ ms}$ because it computes a $1 \times 512$ dot product against a $6 \times 512$ NumPy array | **PARTIALLY VERIFIED (TRIVIAL)** |

---

## 1. Audit of Pixel-Aligned Saliency Grounding (PASG-GradCAM)

### 1.1 Mathematical Formulation in Manuscript
Section III-E of the IEEE draft (`manuscript.tex#L513-L524`) defines:
$$\alpha_k^c = \frac{1}{H' W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial Y^c}{\partial A_{i, j}^k}, \quad L^c(x, y) = \mathrm{ReLU}\left( \sum_{k=1}^C \alpha_k^c A^k(x, y) \right)$$
$$\eta_{\text{macula}} = \frac{\iint_{\Omega_{\text{macula}}} \tilde{L}^c(x, y) \, dx dy}{\iint_{\Omega_{\text{total}}} \tilde{L}^c(x, y) \, dx dy}, \quad \eta_{\text{disc}} = \frac{\iint_{\Omega_{\text{disc}}} \tilde{L}^c(x, y) \, dx dy}{\iint_{\Omega_{\text{total}}} \tilde{L}^c(x, y) \, dx dy}$$
Table VIII reports:
- Pointing Game Accuracy: **88.4%**
- Energy Ratio $\eta_{\text{macula}}$: **62.8%**
- Energy Ratio $\eta_{\text{disc}}$: **66.4%**
- Latency: **18.5 ms**

### 1.2 Forensic Inspection of `backend/services/explanation_service.py`
We inspected lines 1–96 of [`backend/services/explanation_service.py`](file:///d:/OphthalmoAI/backend/services/explanation_service.py):
1. **Grad-CAM Execution:**
   ```python
   from pytorch_grad_cam import GradCAM
   from pytorch_grad_cam.utils.image import show_cam_on_image
   from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
   ...
   cam = GradCAM(model=model, target_layers=target_layers)
   grayscale = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])
   ```
   This is standard, unconstrained Grad-CAM from the third-party `pytorch_grad_cam` library. It does not perform any pixel alignment, landmark bounding, or anatomical masking.
2. **Missing Energy Calculations:**
   There is **no calculation of $\eta_{\text{macula}}$ or $\eta_{\text{disc}}$** anywhere in `explanation_service.py`, `backend/`, `scripts/`, or `models/`.
3. **Mocked Spatial Description:**
   Lines 66–95 define `generate_spatial_description(diagnosis)`:
   ```python
   def generate_spatial_description(diagnosis: str) -> str:
       quadrants = [
           "superotemporal macular arcade",
           "inferonasal quadrant near optic disc",
           "foveal center",
           "peripapillary region",
           "inferotemporal retinal periphery",
       ]
       patterns = {
           "Diabetic Retinopathy": (
               f"microaneurysms, dot-blot hemorrhages, and lipid exudates localized along the {random.choice(quadrants)}"
           ),
           "Glaucoma": (
               "neuroretinal rim notch and cup-to-disc enlargement centered at the peripapillary optic nerve head"
           ),
           ...
       }
       return patterns.get(diagnosis, f"mild focal anomalies noted near the {random.choice(quadrants)}")
   ```
   **Critical Finding:** Clinical spatial descriptions returned by the API are generated by calling `random.choice(quadrants)`! An image diagnosed with Diabetic Retinopathy has its lesion localization randomly assigned to the "superotemporal macular arcade" or "inferotemporal retinal periphery" on each invocation, with zero connection to model gradients or actual lesion anatomy.
- **Evidence Label:** **CONTRADICTED / INVALID**

---

## 2. Anatomical Category Error in `backend/biomarker_extractor.py`

In [`backend/biomarker_extractor.py`](file:///d:/OphthalmoAI/backend/biomarker_extractor.py#L60-L117), the function `extract_visual_biomarkers()` purports to extract quantitative clinical biomarkers from Grad-CAM heatmaps:
```python
iris_mask = np.zeros((h, w), dtype=np.uint8)
if CV2_AVAILABLE:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=w // 4,
        param1=50, param2=30, minRadius=min(h, w) // 10, maxRadius=min(h, w) // 3
    )
...
corneal_ratio = round(float(overlap / active_pixels) * 100.0, 1)
sclera_mask = ((1 - iris_mask) == 1) & (L > 90)
scleral_icterus = round(float(np.mean(np.maximum(0, sclera_b)) / 12.8), 2)
```
Lines 141–154 then format these biomarkers into prompt context for an LLM:
- `- Corneal / Iris Involvement: {corneal}%`
- `- Vascular Erythema Index: {erythema} (conjunctival/ciliary injection severity)`
- `- Scleral Icterus Yellowness Index: {icterus}`

### Clinical AI Safety Assessment
- **Anatomical Incompatibility:** A color fundus photograph (CFP) captures the **posterior pole** of the eye (retina, macula, fovea, optic nerve head, retinal arcade vessels) through a dilated pupil. It does **not** image the cornea, iris, conjunctiva, or anterior sclera.
- Attempting to detect an "iris" via Hough circles on a retinal photograph and reporting "corneal involvement", "conjunctival injection", and "scleral icterus" to clinicians or LLMs represents a fundamental anatomical mismatch.
- **Evidence Label:** **INVALID (ANATOMICAL MISMATCH)**

---

## 3. Audit of Anatomical Ground Truth & U-Net Landmark Segmentation

### 3.1 Claims Made in Manuscript
Section III-E (`manuscript.tex#L524`) states:
> "The anatomical regions of interest $\Omega_{\text{macula}}$ and $\Omega_{\text{disc}}$ are delineated using an automated segmentation pipeline (a lightweight U-Net trained on public benchmark fundus segmentation datasets, including REFUGE and DRIVE). To verify landmark fidelity, an expert ophthalmologist audited and validated a 10% representative random sample of the cohort (achieving a mean Dice similarity of $0.91 \pm 0.04$ against manual contours)."

### 3.2 Codebase & Model Inventory Audit
1. **Model Checkpoints:** A complete inventory of [`models/`](file:///d:/OphthalmoAI/models/) reveals 13 PyTorch `.pth` files:
   - `convnext_small.pth`, `cpu_convnext_small.pth`, `gpu_convnext_small.pth`
   - `densenet201.pth`, `cpu_densenet201.pth`, `gpu_densenet201.pth`
   - `efficientnet_v2_m.pth`, `cpu_efficientnet_v2_m.pth`, `gpu_efficientnet_v2_m.pth`
   - `efficientnet_b4.pth`, `gpu_efficientnet_b4.pth`
   - `meta_classifier.pth`, `hybrid_meta_classifier.pth`
   - `mobilenet_v3_small_edge.pth`, `edge_model_fp32.onnx`
   - `resnet50.pth`, `gpu_resnet50.pth`
   - `retfound_finetuned.pth`
   - **Zero U-Net checkpoints exist.** No model for optic disc or macula segmentation exists in the repository.
2. **Datasets:** Neither DRIVE nor REFUGE segmentation masks are stored in `dataset/`.
3. **Audit Documentation:** There is zero record, spreadsheet, annotation protocol, IRB approval, or clinician signature documenting the claimed "10% ophthalmologist audit (Dice $0.91 \pm 0.04$)".
4. **Removal in Clean Manuscript:** In `D:\OphthalmoAI-Paper\main.tex`, the entire U-Net and ophthalmologist audit claim was stripped out, corroborating that it was an unsupported legacy assertion.
- **Evidence Label:** **CONTRADICTED / UNVERIFIED**

---

## 4. Audit of Lesion-Level Ground Truth Masks

### Clinical Grounding Analysis
The manuscript claims that PASG-GradCAM ensures "verified clinical grounding of diagnostic attribution" by linking heatmaps to pathological lesions (microaneurysms, blot hemorrhages, hard exudates, drusen).

- **Data Inspection:** All underlying training and evaluation datasets in [`dataset/processed/`](file:///d:/OphthalmoAI/dataset/processed/) (`ODIR-5K`, `APTOS-2019`, `color_fundus_eye`) contain **only whole-image diagnostic category labels** ($y \in \{0, 1, 2, 3, 4, 5\}$).
- There are **no bounding boxes, no polygon annotations, and no pixel-level binary masks** for microaneurysms, drusen, or optic disc cup/rim boundaries anywhere in the project data.
- **Scientific Audit Finding:** In the absence of pixel-level lesion ground truth masks, it is mathematically and clinically impossible to benchmark "pointing game accuracy" or "biomarker energy ratios" against genuine lesion ground truth. The numbers reported in Table VIII (Pointing Game 88.4%) could not have been evaluated against real lesion ground truth on this dataset.
- **Evidence Label:** **CONTRADICTED**

---

## 5. Audit of Content-Based Medical Image Retrieval (CBMIR)

Section III-E (`manuscript.tex#L525-L535`) and Fig. 3 claim:
> "In addition, Content-Based Medical Image Retrieval (CBMIR) extracts dense 512-dimensional penultimate embeddings: ... retrieving top-$k$ reference cases with verified clinical and multimodal ophthalmic outcomes (including optical coherence tomography and dilated funduscopy) to support clinical decision-making. Penultimate 512-dimensional feature projections from the EfficientNet-B4 backbone are indexed via an inverted-file FAISS index, enabling sub-5 ms cosine similarity retrieval..."

### 5.1 Inspection of `backend/vector_search.py`
We audited [`backend/vector_search.py`](file:///d:/OphthalmoAI/backend/vector_search.py#L1-L195):
1. **Archive Size & Clinical Depth:**
   [`HISTORICAL_CLINICAL_ARCHIVE`](file:///d:/OphthalmoAI/backend/vector_search.py#L10-L71) contains **exactly 6 hardcoded mock dictionary entries**:
   - `REF-DR-104` (Diabetic Retinopathy)
   - `REF-GL-201` (Glaucoma)
   - `REF-AMD-402` (Age-related Macular Degeneration)
   - `REF-CAT-305` (Cataract)
   - `REF-HT-503` (Hypertensive Retinopathy)
   - `REF-NORM-001` (Normal)
   There is no historical database of thousands of real patient encounters.
2. **Embedding Generation Mechanism:**
   Lines 83–96 implement `_generate_synthetic_class_embedding()`:
   ```python
   def _generate_synthetic_class_embedding(
       class_index: int, dim: int = 512, seed: int | None = None
   ) -> np.ndarray:
       s = (class_index * 1337 + 42) if seed is None else seed
       rng = np.random.RandomState(s)
       base = rng.randn(dim).astype(np.float32) * 0.15
       subspace_size = max(10, dim // len(TARGET_CLASSES))
       start = (class_index % len(TARGET_CLASSES)) * subspace_size
       base[start : start + subspace_size] += 6.0
       norm = float(np.linalg.norm(base))
       return base / max(norm, 1e-6)
   ```
   **Embeddings are not extracted from EfficientNet-B4.** They are generated from synthetic Gaussian random noise (`rng.randn(512)`) with a static $+6.0$ offset added to the subspace slice corresponding to `class_index`.
3. **Query Vector Synthesis:**
   Lines 119–146 implement `extract_embedding_from_probabilities()`:
   ```python
   query_vec = np.zeros(self.dim, dtype=np.float32)
   for c_idx, w in enumerate(weights):
       proto = _generate_synthetic_class_embedding(c_idx, dim=self.dim)
       query_vec += w * proto
   ```
   The query vector is **not computed by feeding the fundus image into a neural network**. It is simply a linear combination of the 6 synthetic random class vectors, weighted by the model's posterior output probabilities!
4. **FAISS Indexing Absence:**
   Lines 174: `cosine_sims = np.dot(self.vectors, q_normed.T).flatten()`.
   There is **no FAISS library dependency, no inverted-file index (IVF), and no vector database**. It is a pure NumPy dot product against a $6 \times 512$ matrix.
5. **Retrieval Latency:**
   The claim of sub-5 ms retrieval latency is technically true only because computing `np.dot` between a $1 \times 512$ array and a $6 \times 512$ array requires less than $0.05\text{ ms}$. However, this is a mock retrieval over 6 synthetic records, not a production CBMIR engine over an indexed clinical archive.
- **Evidence Label:** **CONTRADICTED / SIMULATED**

---

## 6. Mandatory Remediation Recommendations

1. **Remove Unimplemented Formulation 5 from Manuscript:**
   - [PROPOSED] Strike Section III-E (PASG-GradCAM, $\eta_{\text{macula}}$, $\eta_{\text{disc}}$, Table VIII) from any manuscript submission. Describe explainability truthfully: *"Visual explainability is provided via unconstrained Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional layer of the EfficientNet-B4 feature extractor."*
2. **Deprecate or Re-engineer `biomarker_extractor.py`:**
   - [PROPOSED] Remove the corneal/iris/sclera Hough circle logic from `backend/biomarker_extractor.py`. Posterior fundus photography cannot measure anterior segment biomarkers.
3. **Fix Random Saliency Description in `explanation_service.py`:**
   - [PROPOSED] Remove `random.choice(quadrants)`. If real lesion localization cannot be computed via genuine bounding models, the API must not output randomized anatomical locations.
4. **Disclose CBMIR as a Demonstration Mock:**
   - [PROPOSED] Clarify in architecture documentation that the current CBMIR service is an in-memory prototype demonstrating the API schema with 6 illustrative cases, pending integration with a true clinical PACS archive.
