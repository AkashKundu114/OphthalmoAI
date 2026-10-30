# Phase 11: Systematic Related Work, Prior Art & Novelty Assessment

**Audit Date:** October 30, 2026  
**Auditor:** Skeptical Research Supervisor, Senior Biomedical AI Researcher & IEEE Reviewer  
**Repository Context:** `D:\OphthalmoAI` (Engineering Codebase & Benchmarks) and `D:\OphthalmoAI-Paper` (Research Manuscript)  
**Manuscript Under Audit:** `research/manuscript.tex` (IEEE 2-column draft) & `main.tex` (Springer Nature revision)  
**Venue Target:** IEEE International Conference on Cognitive Computing and Engineering Evaluation (ICCECE 2027)  

---

## 1. Executive Summary & Review Posture

A primary reason biomedical artificial intelligence manuscripts face rejection at competitive IEEE conferences and top-tier journals is the conflation of **engineering integration** with **genuine algorithmic novelty**, alongside superficial literature reviews that fail to position the proposed work against landmark clinical benchmarks and modern foundation models.

This audit executes an uncompromising literature review and novelty analysis of OphthalmoAI across five foundational axes:
1. **Landmark Clinical AI & Commercial SaMD Systems:** Gulshan et al. (2016), Ting et al. (2017), Abràmoff et al. (IDx-DR / LumineticsCore, 2018), and Eyenuk EyeArt.
2. **Retinal Foundation Models:** RETFound (Zhou et al., 2023), FLAIR (Silva-Rodriguez et al., 2025), and EyeCLIP (Shi et al., 2025).
3. **Medical Conformal Prediction & Risk Control:** Vovk (Mondrian Conformal Prediction, 2003), Angelopoulos & Bates (2021), Angelopoulos et al. (Conformal Risk Control, 2023; Learn-then-Test, 2024), and Lu et al. (2021).
4. **Out-of-Distribution (OOD) Detection & Quality Gating:** MSP (Hendrycks & Gimpel, 2017), ODIN (Liang et al., 2018), Mahalanobis (Lee et al., 2018), and EyeQuality (Fu et al., 2019).
5. **Medical Explainability & Visual Grounding:** Grad-CAM (Selvaraju et al., 2017) and Explainability Sanity Checks (Adebayo et al., 2018).

---

## 2. Comprehensive State-of-the-Art Literature Survey

### 2.1 Landmark Clinical AI & Commercial Screening Systems
- **Gulshan et al. (JAMA 2016; 316(22):2402–2410; DOI: 10.1001/jama.2016.17216):**
  - *Contribution:* Applied deep convolutional neural networks (Inception-v3) to binary referable diabetic retinopathy (RDR; moderate or worse DR, or diabetic macular edema) across EyePACS ($n=9,963$) and Messidor-2 ($n=1,748$), reporting sensitivities of $97.5\%$ and $96.1\%$, and specificities of $93.4\%$ and $93.9\%$.
  - *Architectural & Clinical Scope:* Monolithic 2D classification for a single disease entity; high reliance on consensus adjudication (7–8 US board-certified ophthalmologists). Output was an uncalibrated point softmax probability without finite-sample error guarantees.
- **Ting et al. (JAMA 2017; 318(22):2211–2223; DOI: 10.1001/jama.2017.18152):**
  - *Contribution:* Developed the Singapore Integrated Diabetic Retinopathy Programme (SiDRP) multi-condition system across $494,661$ retinal images from multi-ethnic Asian cohorts, expanding beyond DR to include suspected glaucoma and age-related macular degeneration (AMD).
  - *Architectural & Clinical Scope:* Semi-independent binary classifiers evaluated sequentially; centralized server-based infrastructure; proprietary datasets without public distribution-free uncertainty control.
- **Abràmoff et al. / IDx-DR (npj Digital Medicine 2018; 1:39; DOI: 10.1038/s41746-018-0040-6):**
  - *Contribution:* Conducted the pivotal prospective multi-center clinical trial ($n=900$ diabetic patients in primary care) evaluating the autonomous IDx-DR (now Digital Diagnostics LumineticsCore) system, securing the first FDA De Novo clearance (DEN180001) for an autonomous diagnostic AI system in any medical specialty.
  - *Clinical Performance & Safety:* Reported sensitivity of $87.2\%$ ($95\%$ CI: $81.8\%-91.2\%$) and specificity of $90.7\%$ ($95\%$ CI: $88.3\%-92.7\%$) against stereoscopic 4-widefield fundus photography adjudicated by the Wisconsin Reading Center.
  - *Architectural Gating:* IDx-DR incorporates a deterministic image quality assessment (IQA) module that rejects ungradable scans before diagnostic inference, mandating re-imaging or clinical specialist referral.
- **Eyenuk EyeArt (Ophthalmology 2020; 127(8):1075–1085; DOI: 10.1016/j.ophtha.2020.02.008):**
  - *Contribution:* Prospective validation of the autonomous EyeArt system in primary care, demonstrating $95.5\%$ sensitivity and $86.5\%$ specificity for referable DR, securing FDA 510(k) clearance (K200667).
  - *Clinical Distinction:* Emphasizes high sensitivity to prevent missed referrals, but operates as a cloud-served or enterprise clinic appliance with proprietary closed-source algorithms.

### 2.2 Retinal Vision Foundation Models
- **RETFound (Zhou et al., Nature 2023; 622:156–163; DOI: 10.1038/s41586-023-06555-x):**
  - *Contribution:* First general-purpose retinal foundation model, built on a Vision Transformer (ViT-Large, $304\text{M}$ parameters) pre-trained via Masked Autoencoding (MAE, $75\%$ masking ratio) on $1.6\text{M}$ unlabeled fundus and OCT images.
  - *Strengths:* Exceptional label efficiency on downstream ocular classification, systemic disease prediction (myocardial infarction, Parkinson's disease, heart failure), and genetic stroke risk.
  - *Deployment Bottlenecks:* Requires $>24\text{ GB}$ VRAM for training and $\sim 2.3\text{ GB}$ peak VRAM for single-batch FP32 inference; compute-heavy architecture ill-suited for edge microcontrollers or disconnected rural clinics without dedicated GPU hardware.
- **FLAIR (Silva-Rodriguez et al., Nature Communications 2025; 16:1820; DOI: 10.1038/s41467-025-56891-2):**
  - *Contribution:* Vision-language foundation model pre-trained on retinal images paired with descriptive clinical text, enabling zero-shot disease classification, expert prompt conditioning, and text-guided retrieval.
  - *Contrast:* Excels at open-vocabulary exploration, but introduces vision-language hallucination risks and high parameter counts ($>400\text{M}$ multimodal parameters).
- **EyeCLIP (Shi et al., IEEE TMI 2025; DOI: 10.1109/TMI.2025.3526101):**
  - *Contribution:* Multi-modal contrastive pre-training framework leveraging structured ophthalmic diagnostic reports and anatomical knowledge graphs.
  - *Limitation:* Focuses on semantic representation learning in centralized academic clusters; does not provide formal distribution-free risk guarantees, edge ONNX runtimes, or deterministic biophysical optical guardrails.

### 2.3 Medical Conformal Prediction & Conformal Risk Control
- **Vovk (Mondrian Conformal Prediction 2003; Technical Report CSL-TR-03-001) & Vovk et al. (2005):**
  - *Foundational Theory:* Established that exchangeable non-conformity scores permit constructing finite-sample prediction sets $\mathcal{C}(X_{n+1})$ satisfying $\mathbb{P}(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - \alpha$. Mondrian conformal prediction partitions the calibration space into disjoint categories (e.g., disease classes or age groups) to guarantee group-conditional validity.
- **Angelopoulos & Bates (Foundations and Trends in Machine Learning 2021; 14(4):403–491; DOI: 10.1561/2200000085):**
  - *Contribution:* Comprehensive formulation of split conformal prediction, adaptive prediction sets (APS), and regularized adaptive prediction sets (RAPS), popularizing distribution-free uncertainty quantification across modern computer vision.
- **Angelopoulos et al. (Conformal Risk Control, JMLR 2023; 24(163):1–61):**
  - *Contribution:* Extended conformal prediction beyond misclassification loss ($0/1$ loss) to general bounded, monotonically non-increasing clinical loss functions $\mathcal{L}(\mathcal{C}(X), Y) \in [0, B]$, guaranteeing $\mathbb{E}[\mathcal{L}(\mathcal{C}(X_{n+1}), Y_{n+1})] \le \alpha$.
- **Angelopoulos et al. (Learn-then-Test, PNAS 2024; 121(24):e2311477121):**
  - *Contribution:* General framework for bounding multiple complex risk criteria simultaneously with family-wise error rate control.
- **Lu et al. (Medical Image Analysis 2021; 68:101892; DOI: 10.1016/j.media.2020.101892):**
  - *Contribution:* Applied ordinal conformal prediction to diabetic retinopathy grading, demonstrating that expanding prediction sets on borderline images reduces false-negative referrals. *Limitation:* Applied marginal error budgets that treat clean tabletop fundus scans and hazy cataract-degraded captures identically.

### 2.4 Out-of-Distribution (OOD) Detection in Medical Imaging
- **Maximum Softmax Probability (MSP, Hendrycks & Gimpel, ICLR 2017):**
  - *Mechanism:* Thresholds raw maximum softmax output $\max_c P(y=c \mid X) \ge \tau$.
  - *Failure Mode in Clinical AI:* Deep neural networks trained with cross-entropy loss are notoriously overconfident on non-medical imagery, frequently assigning $>90\%$ confidence to random noise, chest X-rays, or everyday scenery.
- **ODIN (Liang et al., ICLR 2018):**
  - *Mechanism:* Applies large temperature scaling ($T=1000$) combined with small adversarial input perturbations to amplify the gap between in-distribution and OOD samples.
  - *Limitation:* Requires backpropagation through the model for every inference call, increasing inference latency by $1.5\times-2.0\times$, rendering it impractical for low-power edge microcontrollers.
- **Mahalanobis Distance (Lee et al., NeurIPS 2018):**
  - *Mechanism:* Fits class-conditional Gaussian distributions on penultimate layer features and computes Mahalanobis distance.
  - *Failure Mode:* Ren et al. (NeurIPS 2021) proved that statistical latent-space detectors collapse under mild camera hardware or illumination shifts, misidentifying valid patient scans from a new clinic as OOD.
- **EyeQuality / Deep Fundus Quality Assessment (Fu et al., IEEE TMI 2019; 38(11):2690–2701):**
  - *Mechanism:* Multi-task deep neural network trained to classify fundus image quality into 'Good', 'Usable', and 'Reject'.
  - *Limitation:* Requires a heavy deep network solely for quality filtering and requires thousands of manually graded training images that are prone to inter-annotator disagreement.

### 2.5 Explainability & Visual Saliency Sanity Checks
- **Grad-CAM (Selvaraju et al., ICCV 2017; pp. 618–626):**
  - *Mechanism:* Computes the gradient of class score $Y^c$ with respect to penultimate feature activations $A^k$, averaging spatial gradients to compute feature weights $\alpha_k^c$ followed by ReLU.
- **Adebayo et al. (Sanity Checks for Saliency Maps, NeurIPS 2018; 31:9505–9515):**
  - *Landmark Finding:* Demonstrated that several widely used visual attribution methods (e.g., Guided Backpropagation, Guided Grad-CAM) are invariant to model parameter randomization and ground-truth data randomization, functioning effectively as edge detectors rather than faithful representations of model decision logic.
  - *Clinical Implication:* Cosmetic heatmaps in medical AI must be subjected to quantitative grounding benchmarks (e.g., Pointing Game, anatomical energy fraction against verified segmentation boundaries) rather than anecdotal visual presentation.

---

## 3. The Novelty & Contribution Matrix

The following matrix systematically dissects every architectural, algorithmic, and engineering component claimed in the OphthalmoAI research manuscript, classifying each into its true technical contribution tier:
1. **Established Technique:** Standard prior art adopted directly from literature without meaningful modification.
2. **Engineering Integration:** Assembly and optimization of existing techniques into a coherent, production-ready system.
3. **Methodological Modification:** Principled adaptation of an established mathematical framework to a domain-specific clinical constraint.
4. **Potentially Novel Contribution:** Unique conceptual, algorithmic, or empirical contribution with demonstrable scientific value.
5. **Unsupported Novelty Claim:** Aspirational or theoretical claim asserted in the manuscript but not supported by executable code, verified data, or valid experimental methodology.

| Proposed System Component | Prior Art / Baseline | OphthalmoAI Specific Formulation | Genuine Scientific Novelty vs. Engineering Glue | Methodological Gap or Forensic Flaw | Classification |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Tri-Backbone Ensemble with Learned Stacking** | Caruana et al. (2004), Wolpert (1992) stacking; standard soft voting. | Stacking MLP combining calibrated logits from DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M. | **Engineering Integration:** Selecting diverse architectures (Dense connections, $7\times 7$ Depthwise, Progressive MBConv) and tuning on clean splits provides empirical resilience, but stacking is standard ML practice. | Modest accuracy improvement over best single backbone (+1.29% over DenseNet-201). Ensemble increases parameter count to 120.5M. | **Engineering Integration** |
| **Platt Temperature Calibration (TC-MBE)** | Guo et al. (ICML 2017), Mehrtash et al. (IEEE TMI 2020). | Fits post-hoc temperature parameters ($T \in [1.17, 1.21]$) per backbone on validation split prior to logit fusion. | **Established Technique:** Straightforward parameter fitting of a single scalar per architecture minimizing negative log-likelihood. | No novel calibration algorithm developed; uses standard SciPy/PyTorch BFGS optimization. | **Established Technique** |
| **Deterministic Biophysical Optical Guardrail (OAC-DG)** | Statistical OOD: MSP (2017), ODIN (2018), Mahalanobis (2018); Fu et al. (2019). | Evaluates telecentric aperture geometry, pupil boundary ratios, spatial autocorrelation ($r \ge 0.35$), and hemoglobin/melanin absorption ($\rho_{\text{RB}} \ge 1.05$) in 1.8 ms on CPU before GPU execution. | **Potentially Novel Contribution:** Shifting OOD gating from brittle statistical latent feature spaces to deterministic optical and physiological camera physics invariants. Prevents compute waste and softmax hallucination on non-fundus inputs. | The 2,500 negative stress benchmark in `scripts/reproduce_evaluation.py` is hardcoded; raw negative stress datasets (CheXpert, ISIC) are missing from the repo. | **Methodological Modification / Potential Novelty** |
| **Admissibility-Weighted Conformal Risk Control (AW-CRC)** | Vovk Mondrian CP (2003); Angelopoulos et al. Conformal Risk Control (2023). | Dynamically weights non-conformity score by continuous optical quality: $s_{\text{AW}}(X, y) = (1 - P(y \mid X)) \cdot \Phi(X)^\gamma$, setting urgency-stratified quantiles ($\hat{q}_{\text{emerg}}, \hat{q}_{\text{rout}}$). | **Methodological Modification:** Bridging biophysical image admissibility directly into distribution-free conformal non-conformity scoring, restoring Grade C media coverage from 78.31% to 97.59%. | **Cardinality Flaw:** Produces 3.14% empty prediction sets ($\mathcal{C}(X) = \emptyset$, mean set size $0.9686 < 1.0$), leaving critical patients with zero diagnostic candidates without clear triage handling. | **Methodological Modification** |
| **Pixel-Aligned Saliency Grounding (PASG-GradCAM)** | Selvaraju et al. (2017) Grad-CAM; Adebayo et al. (2018) sanity checks. | Formulation 5 claims automated U-Net segmentation enforcing energy fractions ($\eta_{\text{macula}} \ge 0.40, \eta_{\text{disc}} \ge 0.40$) audited by an ophthalmologist (Dice 0.91). | **Unsupported Novelty Claim:** The repository contains NO U-Net segmentation model, NO pointing game code, and NO ophthalmologist audit data. In `explanation_service.py`, spatial descriptions are selected via `random.choice(quadrants)`. | Complete divergence between theoretical manuscript formulation and physical repository implementation. | **Unsupported Novelty Claim** |
| **Multimodal Clinical Synergy (Fundus + EHR Bio-Data)** | Huang et al. (2020), Xu et al. (2023) multimodal tabular-imaging fusion. | Claims combining fundus images with 12 EHR biometric variables (HbA1c, SBP, IOP, LogMAR) boosts accuracy from 85.18% to 87.42%. | **Invalid Methodology:** Tabular EHR data in `test_patient_clean.csv` was generated by `generate_patient_biodata.py` by sampling Gaussian distributions conditioned directly on ground-truth disease labels. | Circular label leakage artifact; evaluates synthetic generator, not genuine clinical synergy. Must be retracted. | **Unsupported Novelty Claim / Invalid** |
| **Contralateral Patient-Level Leakage Audit** | Standard epidemiological and ML best practices (Kaufman et al., 2012; Yagis et al., 2021). | Programmatic demonstration that random image splitting yielded 35.96% patient overlap and 97.23% accuracy, whereas strict patient partitioning yielded 89.77%. | **High-Value Methodological Negative Control:** Thoroughly documents and quantifies the magnitude of optimistic bias (+7.46% accuracy) caused by bilateral eye leakage in ophthalmic vision benchmarks. | Patient ID clustering for 16,140 `fundus_cfe_` images relied on a naive modulo index heuristic (`num // 2`) rather than verified hospital patient identifiers. | **Methodological Modification** |
| **Edge Deployment via FP16 ONNX & In-Browser WASM** | Standard ONNX Runtime, WebAssembly, and model quantization tooling. | Packaging MobileNet-V3-Small into FP32/FP16 ONNX for client-side zero-cloud in-browser execution (<50 ms). | **Engineering Integration:** Translating a trained vision model to ONNX/WASM for rural LMIC telehealth is high-quality software engineering, but represents standard deployment tooling. | The 89.77% accuracy was achieved by the 120.5M parameter tri-ensemble, NOT the 6.1 MB MobileNet-V3-Small edge model. Conflating the two architectures in the text is misleading. | **Engineering Integration** |
| **Multi-Tenant Row-Level Security & HIPAA Schema** | Standard PostgreSQL RLS policies; HIPAA security rule (45 CFR § 164.312). | Manuscript presents SQL snippet for PostgreSQL `CREATE POLICY tenant_isolation` using `current_setting('app.tenant')`. | **Aspirational / Proposed:** Standard enterprise relational database configuration. | Zero RLS policies exist in Alembic migrations or backend code; active local DB is SQLite, which cannot execute PostgreSQL RLS. | **Proposed / Unsupported Claim** |

---

## 4. Honest Delineation: Genuine Novelty vs. Engineering Glue

### 4.1 What is Truly Novel (or Methodologically Meaningful)
1. **The Biophysical Domain Guardrail (OAC-DG):**
   - *Assessment:* Shifting pre-inference out-of-distribution detection from opaque, brittle latent-space density estimators (which fail under camera sensor changes) to deterministic biophysical invariants (telecentric aperture geometry and hemoglobin/melanin chromophore absorption ratios $\rho_{\text{RB}} \ge 1.05$) is a **genuine methodological contribution**. Operating in $<2.0$ ms on host CPU before allocating GPU tensors is operationally valuable for frontline edge devices.
2. **Admissibility-Weighted Conformal Risk Control (AW-CRC):**
   - *Assessment:* Adapting non-conformity scores dynamically using continuous physical image quality ($\Phi(X)^\gamma$) to expand prediction sets on hazy Grade C media—guaranteeing 98.8% coverage on sight-threatening emergencies without manual threshold tuning—is a **sound methodological adaptation of conformal prediction to ophthalmic optical physics**.
3. **The Patient Leakage Forensic Analysis (97.23% vs. 89.77%):**
   - *Assessment:* Rigorously isolating the 319-patient overlap (35.96%) between naive training and test splits and proving that it artificially inflated accuracy from 89.77% to 97.23% provides an **exemplary methodological case study** on the dangers of image-level splitting in medical imaging.

### 4.2 What is Engineering Glue
1. **The Tri-Backbone Ensemble:**
   - Combining DenseNet-201, ConvNeXt-Small, and EfficientNet-V2-M with a 2-layer stacking MLP is standard machine learning engineering. It provides empirical stability, but contains no architectural or algorithmic innovation.
2. **Temperature Scaling (TC-MBE):**
   - Applying Platt temperature scaling to individual backbones is standard practice since Guo et al. (ICML 2017).
3. **ONNX Export and WebAssembly Pipeline:**
   - Converting a PyTorch checkpoint to ONNX and loading it via `onnxruntime-web` in a React application is modern full-stack web engineering, not machine learning research.
4. **FastAPI Middleware & Cryptographic Logging:**
   - Security headers, SlowAPI rate limiting, and SHA-256 audit logs are standard cybersecurity hygiene, not academic contributions.

### 4.3 What Must Be Retracted or Reframed
1. **PASG-GradCAM & Pointing Game Benchmark:**
   - Claiming an automated U-Net segmentation pipeline, an 88.4% Pointing Game benchmark, and a 10% ophthalmologist audit (Dice 0.91) while the codebase executes standard Grad-CAM and selects spatial quadrants via `random.choice()` is an **unsupported scientific claim**. It must be completely reframed as a proposed conceptual framework or removed until the code is committed.
2. **Multimodal EHR Synergy & Demographic Fairness:**
   - Because the 12 tabular EHR features (HbA1c, SBP, IOP, age, sex) were synthetically generated conditioned on true disease labels, all claims of multimodal diagnostic synergy (85.18% $\to$ 87.42%) and demographic equity (DIR $\ge 0.966$, $\Delta_{\text{EO}} = 0.022$) are **scientifically invalid**. They must be permanently removed from the primary manuscript.
3. **PostgreSQL Row-Level Security:**
   - The paper presents SQL code for database-level multi-tenancy that does not exist in the repository. Theoretical SQL snippets belong in system architecture documentation, not in an academic peer-reviewed screening manuscript.

---

## 5. Strategic Literature Positioning for Publication

To successfully navigate peer review at ICCECE 2027 or target Springer Nature venues, the authors must pivot their positioning narrative:

| Dimension | Flawed Manuscript Posture | Rigorous Scientific Posture |
| :--- | :--- | :--- |
| **System Identity** | "Autonomous clinical-grade enterprise SaMD platform." | "Investigational point-of-care retinal screening decision-support aid with safety-first rejection." |
| **Comparison vs. RETFound** | "OphthalmoAI decisively outperforms foundation models." | "OphthalmoAI demonstrates that a lightweight 120M ensemble optimized for edge execution matches foundation model accuracy while providing deterministic OOD safety and conformal risk control that cloud foundation models lack." |
| **External Specificity Collapse** | Suppressed or minimized in discussion. | Highlighted as a central clinical finding: camera hardware shifts cause severe false-positive over-referral (16.42% specificity on IDRiD), proving why selective prediction and uncertainty escalation are mandatory before real-world deployment. |
| **Explainability** | Asserted as a solved quantitative biomarker audit (Table 9). | Acknowledged as qualitative gradient-based visual localization (Grad-CAM), noting that true lesion-level segmentation grounding remains future work. |
| **Scope of Paper** | Sprawling 25-page compendium covering clinical trials, ML models, hardware silicon, enterprise databases, and regulatory roadmaps. | Focused, mathematically rigorous 8-page paper centered exclusively on: (1) Biophysical Guardrails, (2) Calibrated Ensembling, and (3) Admissibility-Weighted Conformal Risk Control. |
