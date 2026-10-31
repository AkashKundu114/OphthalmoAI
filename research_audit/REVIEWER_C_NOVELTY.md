# IEEE Conference Peer Review Report — Reviewer C (Novelty & Significance)

**Reviewer Identity:** Reviewer C (IEEE Reviewer / Novelty, Computer Vision & Biomedical AI Specialist)  
**Manuscript Title:** Quality-Aware Conformal Screening for Retinal Fundus Imaging Under Optical Domain Shift  
**Target Venue:** ICCECE 2027 (Double-Blind Review)  
**Recommendation:** **ACCEPT WITH MINOR REVISION**  
**Review Confidence:** 5/5 (Specialist in Foundation Models, Conformal Inference, and OOD Detection)  

---

## 1. Overall Assessment
The paper investigates selective classification and uncertainty quantification for multi-class retinal fundus screening. The system combines: (1) a CPU-bound optical plausibility gate; (2) a multi-backbone deep neural ensemble; (3) an Admissibility-Weighted Conformal Risk Control (AW-CRC) framework; and (4) an evaluation under camera domain shift.

While many of the underlying techniques (temperature scaling, split conformal prediction, convolutional vision backbones, Grad-CAM saliency sanity checks) are established in the literature, the authors' integration, rigorous ablation, and transparent disclosure of real-world negative findings create a compelling systems paper with high value for the medical AI community.

---

## 2. Key Strengths & Contributions
1. **Honest Delineation of Contributions:** Unlike many authors who rebrand known ensembling methods as "novel theoretical paradigms", the authors explicitly concede in Section V-A that their MLP stacking ensemble achieves no statistically significant improvement over simple uniform soft voting ($p = 0.6875$), and recommend soft voting as the non-parametric gold standard.
2. **Quality-Conditioned Non-Conformity Formulation:** The AW-CRC formulation $s_{\mathrm{AW}}(X, y) = (1 - \hat{P}(y \mid X)) \cdot \mathcal{S}(X)^\gamma$ successfully couples low-level optical acquisition spectroscopy with high-level statistical confidence sets.
3. **Reproducible Forensic Decontamination:** The authors provide a meticulous accounting of dataset quarantine (8,477 contaminated images purged), conflicting-label deduplication (776 clusters), and zero-ghost-case matrix reconciliation.
4. **Empirical Value of Negative Findings:** Proving that naive argmax specificity drops to $1.05\%$ on external cameras while conformal triage triggers 100% clinician escalation is a vital contribution that cautions against premature clinical deployment.

---

## 3. Critical Novelty Queries & Ablation Demands

### Query C.1: Theoretical Novelty vs. Applied Systems Integration
Is AW-CRC a new statistical theorem, or is it an applied parameterization of split conformal risk control?
- **Critique:** Bates et al. (2021) established the foundational distribution-free risk control theorem. AW-CRC modifies the non-conformity score function and applies stratum-specific quantiles.
- **Required Action:** The manuscript must ensure that the abstract and introduction do not use phrasing that implies a new distribution-free mathematical theorem was proven. It should be framed unambiguously as an **algorithmic and systems formulation** that dynamically couples optical spectroscopy to conformal non-conformity.

### Query C.2: Ablation of Optical Quality Weighting ($\gamma$) in AW-CRC
In Table V, comparing standard Uniform Split CRC (US-CRC) against proposed Admissibility-Weighted CRC (AW-CRC):
- US-CRC achieved: Emergency Coverage: $99.31\%$, Routine: $96.54\%$, Overall: $98.27\%$, Mean Set Size: $1.004$.
- AW-CRC achieved: Emergency Coverage: $99.31\%$, Routine: $96.73\%$, Overall: $98.34\%$, Mean Set Size: $1.010$.
- **Critique:** The overall coverage improvement is modest ($+0.07\%$ overall, $+0.19\%$ on routine). Why does optical weighting matter if the coverage gain is small?
- **Explanation Needed:** The authors must explain that in an in-domain clean test set where $98.7\%$ of images are pristine (Grade A), optical scores $\mathcal{S}(X)$ are uniformly high ($\approx 0.96$), so AW-CRC behaves similarly to US-CRC. The primary value of $\mathcal{S}(X)^\gamma$ manifests on degraded and borderline scans where optical scores drop toward $\tau_{\mathrm{guard}} = 0.50$, expanding prediction sets to trigger human review.

### Query C.3: Comparison with Retinal Foundation Models (e.g. RETFound)
Modern retinal imaging research has increasingly converged on self-supervised foundation models (such as RETFound, Zhou et al., *Nature* 2023).
- **Critique:** Why use a tri-backbone convolutional ensemble (EfficientNet, DenseNet, ConvNeXt; 124.3M parameters) instead of fine-tuning a foundation model like RETFound?
- **Clarification:** The authors should briefly contextualize that standard foundation models still require uncertainty quantification, conformal sets, and optical guardrails, and that convolutional architectures allow lightweight CPU-friendly deployment and deterministic gradient backpropagation without massive ViT inference overhead.

---

## 4. Minor Corrections
1. Fix citation key `\cite{shen2021eyequality}` to `\cite{fu2019eyequality}` (Fu et al., *IEEE TMI* 2019).
2. Ensure acronyms are defined at first appearance (e.g., CRC, RAPS, ECE).

---

## 5. Reviewer Recommendation
**Accept with Minor Revision.** The paper represents a refreshing, methodologically sound, and highly reproducible systems contribution for IEEE conference publication.
