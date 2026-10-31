# OphthalmoAI Paper Integration Queue

**Document ID:** `research_audit/PAPER_INTEGRATION_QUEUE.md`  
**Execution Date:** October 30, 2026  
**Auditor / Publication Specialist:** Senior Medical AI Researcher & Research Software Engineer  
**Status:** READY FOR MANUSCRIPT SYNCHRONIZATION  
**Target Repository:** `D:\OphthalmoAI-Paper`

---

## 1. Editorial Policy and Standing Publication Rules

1. **Academic Journal Status:**
   - Do NOT state or claim that the paper is published in IEEE, IEEE J-BHI, IEEE TMI, or any specific journal.
   - Strictly refer to formatting and standards as:
     - *"High-resolution scientific publication standards (300+ DPI)"*
     - *"Academic journal standards"*
     - *"Research manuscript & benchmark evaluation"*
     - *"Peer-reviewed biomedical informatics manuscript (in preparation)"*
2. **Non-Destructive Workflow:**
   - Keep all paper-related modifications in this integration queue. Do not destructively overwrite historical files in `D:\OphthalmoAI-Paper` until explicit integration review.
3. **Strict Truth in Data & Architecture:**
   - Completely excise synthetic EHR multimodal claims and synthetic demographic fairness claims.
   - Completely excise fictional U-Net anatomical segmentation and ungrounded energy ratios.
   - Completely excise 10% ophthalmologist audit claims that lack documentary IRB / clinician provenance.
   - Replace simulated performance tables with canonical GPU benchmark numbers from `research/results/canonical_eval_test_v2.json`.

---

## 2. Manuscript Change Queue

### Item 1: Abstract & Claims Remediation
- **Original Claim:** "OphthalmoAI integrates deep convolutional ensembles with clinical electronic health records (EHR) yielding a +2.24% multimodal diagnostic synergy and demographic fairness (DIR >= 0.966)."
- **Audited Defect:** Tabular EHR features were sampled from true disease labels (`scripts/generate_patient_biodata.py`), representing circular label leakage.
- **Queued Revision:** 
  - Restructure OphthalmoAI as a pure, high-precision **Retinal Computer Vision & Conformal Triage System**.
  - Frame synthetic EHR experiment honestly as an exploratory negative study or excise completely.
  - Report genuine vision-only ensemble performance on the sealed clean patient-level test set ($N = 1,389$): **Top-1 Accuracy 98.70% (Meta-Ensemble) / 98.56% (Uniform Soft Voting), Macro AUROC 0.9985, ECE 0.0389**.

### Item 2: Mathematical Reconciliation of Table 6 (Primary Benchmark Table)
- **Original Defect:** Reported Macro AUROC of 0.9914 was an arithmetic error (true mean was 0.9839); class sensitivities implied 1,871.4 cases, creating a 147.6 ghost case mismatch with the 2,019 reported total.
- **Queued Revision:** Replace Table 6 with the arithmetically verified canonical confusion matrix and metrics from `canonical_eval_test_v2.json`:
  - **Overall Accuracy:** 98.70% (1,371 / 1,389) [95% CI: 98.13% - 99.28%]
  - **Macro AUROC:** 0.9985
  - **Macro F1:** 0.9803
  - **ECE:** 0.0389
  - **Per-Class Sensitivity:**
    - Normal: 97.92% (376 / 384)
    - Diabetic Retinopathy: 100.00% (483 / 483)
    - Glaucoma: 100.00% (361 / 361)
    - Cataract: 96.97% (128 / 132)
    - AMD (ODIR): 79.31% (23 / 29)
  - **Total True Positives:** $376 + 483 + 361 + 128 + 23 = 1,371$. Exactly matches $1,389 - 18$ errors. Ghost cases = 0.

### Item 3: Conformal Risk Control & AW-CRC Formulation
- **Original Defect:** Evaluated with simulated Beta distributions; suffered from 3.14% empty prediction sets and top-diagnosis uniform thresholding.
- **Queued Revision:** 
  - Update methodology to candidate-specific urgency stratification with non-empty argmax fallback.
  - Report empirical AW-CRC results on $N = 1,389$ test cohort:
    - **Emergency Stratum Coverage ($\alpha = 0.01$):** 99.31% (Guarantee $\ge 99.0\%$)
    - **Routine Stratum Coverage ($\alpha = 0.05$):** 96.73% (Guarantee $\ge 95.0\%$)
    - **Overall Coverage:** 98.34%
    - **Mean Set Cardinality:** 1.01 classes / patient (99.0% singletons)
    - **Fallback Rate:** 0.58%

### Item 4: Explainability & Adebayo Sanity Checks (Section III-E & Table VIII)
- **Original Defect:** Claimed lightweight U-Net trained on REFUGE/DRIVE with Dice $0.91 \pm 0.04$ and ophthalmologist audit of 10% sample; code used vanilla Grad-CAM with `random.choice(quadrants)`.
- **Queued Revision:**
  - Excise U-Net segmentation, pointing game (88.4%), and unsubstantiated ophthalmologist audit claims.
  - Present standard Grad-CAM visual heatmaps accompanied by the **Adebayo et al. (NeurIPS 2018) Parameter Randomization Sanity Check**:
    - Randomizing final convolutional layer weights drops spatial Pearson correlation to $r = \mathbf{-0.0818 \pm 0.1581}$, proving that saliency maps are genuinely governed by learned neural weights.
  - Disclose clearly that saliency heatmaps indicate model activation regions rather than clinical lesion segmentations.

### Item 5: External Validation & Generalization Limitations (Section IV)
- **Original Defect:** Claimed "99.6% sensitivity across external cohorts" while omitting that specificity dropped to 16.42% (IDRiD) and 1.05% (RIM-ONE DL) under camera domain shift.
- **Queued Revision:**
  - Report both sensitivity and specificity transparently under disease-specific binary decision rules:
    - **IDRiD (DR screening):** Sensitivity 99.64% (278/279), Specificity 16.42% (22/134), AUROC 0.8005.
    - **RIM-ONE DL (Glaucoma screening):** Sensitivity 100.00% (200/200), Specificity 1.05% (3/285), AUROC 0.6175.
  - Highlight that AW-CRC conformal triage detects domain shift by triggering clinician human review on 100% of out-of-distribution cases (`human_review_trigger_rate: 1.0`), demonstrating the clinical safety value of selective classification.
  - Detail the foreground-masked Reinhard color constancy normalization.

### Item 6: Edge Deployment Architecture & Micro-Benchmarks
- **Original Defect:** Conflated heavy 124.3M parameter server ensemble with edge MobileNet inference latency.
- **Queued Revision:**
  - Clearly articulate the two-tier architectural design:
    - **Tier 1 (Client-Side Edge Triage):** MobileNetV3-Small ONNX runtime ($5.82\text{ MB}$ footprint, $1.57\text{ ms}$ CPU latency, $636.7\text{ QPS}$) for instant zero-latency pre-screening and bandwidth optimization.
    - **Tier 2 (Hospital / Cloud Server Ensemble):** 124.3M Tri-Backbone ConvNeXt + DenseNet + EfficientNet-V2 ensemble on GPU for definitive multi-class diagnostic evaluation, uncertainty quantification, and Grad-CAM generation.

### Item 7: Statistical Significance Section
- **Queued Addition:**
  - Incorporate McNemar exact test findings: Meta-Ensemble vs. Uniform Soft Voting ($p = 0.6875$), showing simple soft voting matches the meta-classifier without stacking overfitting risk.
  - Report 2,000-iteration paired bootstrap 95% confidence intervals for all architectures and metrics.
