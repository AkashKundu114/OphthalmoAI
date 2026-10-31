# OphthalmoAI Manuscript Revision Changelog

**Document ID:** `research_audit/PAPER_REVISION_LOG.md`  
**Execution Context:** Peer-reviewed biomedical informatics manuscript (in preparation) / ICCECE 2027  
**Audit Date:** October 31, 2026  
**Auditor:** Academic Editor & Research Integrity Panel  
**Status:** ALL REVISIONS VERIFIED & IMPLEMENTED  

---

## 1. Revision Overview

Following the three-reviewer adversarial peer review (Reviewers A, B, and C) and mathematical/statistical verification, targeted editorial, mathematical, and bibliographic corrections were applied directly to the authoritative manuscript `research/paper_revised.tex` and bibliography `research/references.bib`.

---

## 2. Itemized Change Log

### Revision 1: Citation Discrepancy Resolution (Reviewer A & C)
- **File:** `research/paper_revised.tex` (Line 70)
- **Target Text:**
  ```latex
  Image quality assessment (IQA) algorithms, such as EyeQuality \cite{shen2021eyequality}, typically employ deep neural networks...
  ```
- **Replacement Text:**
  ```latex
  Image quality assessment (IQA) algorithms, such as EyeQuality \cite{fu2019eyequality}, typically employ deep neural networks...
  ```
- **Rationale:** `shen2021eyequality` was an invalid BibTeX key causing a missing citation warning. In `references.bib`, the landmark EyeQuality paper is correctly cataloged as `fu2019eyequality` (Fu, Wang, Shen, Cui, Xu, Liu, Ling, *IEEE Transactions on Medical Imaging*, 2019).
- **Verification:** Verified zero missing citations in BibTeX compilation.

---

### Revision 2: Bibliography Deduplication (Reviewer A & Quality Control)
- **File:** `research/references.bib`
- **Action:** Stripped duplicate `@inproceedings` and `@book` entries for:
  1. `vovk2005algorithmic` (Vovk et al., *Algorithmic Learning in a Random World*)
  2. `huang2017densely` (Huang et al., DenseNet, *CVPR 2017*)
  3. `liu2022convnet` (Liu et al., ConvNeXt, *CVPR 2022*)
  4. `tan2021efficientnetv2` (Tan & Le, EfficientNetV2, *ICML 2021*)
- **Rationale:** Duplicate entries caused 4 warning/skip messages in `paper_revised.blg`.
- **Verification:** Total BibTeX entries reduced from 77 to 73 unique entries. `paper_revised.blg` now runs cleanly without collision errors.

---

### Revision 3: Temperature Scaling Order of Operations (Reviewer A Query A.2)
- **File:** `research/paper_revised.tex` (Lines 120-125)
- **Target Text:**
  ```latex
  \textbf{Ensemble Aggregation:} We benchmark two fusion strategies:
  \begin{enumerate}
      \item \textbf{Uniform Soft Voting:} $\hat{P}_{\mathrm{soft}}(k \mid X) = \frac{1}{M} \sum_{m=1}^M \hat{P}_m(k \mid X)$.
      \item \textbf{Learned Stacking (MLP):} A 2-layer perceptron (18 inputs $\to$ 32 hidden $\to$ 5 outputs) trained on out-of-fold calibration logits.
  \end{enumerate}
  ```
- **Replacement Text:**
  ```latex
  \textbf{Ensemble Aggregation:} We benchmark two fusion strategies:
  \begin{enumerate}
      \item \textbf{Uniform Soft Voting:} $\hat{P}_{\mathrm{soft}}(k \mid X) = \frac{1}{M} \sum_{m=1}^M \hat{P}_m(k \mid X)$, averaging the individually temperature-calibrated probabilities ($T_{\mathrm{eff}} = 1.121, T_{\mathrm{dense}} = 1.184, T_{\mathrm{conv}} = 1.153$).
      \item \textbf{Learned Stacking (MLP):} A 2-layer perceptron (18 inputs $\to$ 32 hidden $\to$ 5 outputs) trained on out-of-fold calibration logits with post-hoc temperature scaling ($T_{\mathrm{meta}} = 1.042$).
  \end{enumerate}
  ```
- **Rationale:** Clarifies the exact calibration sequence for both non-parametric soft-voting and learned stacking.

---

### Revision 4: Conformal Coverage Delineation (Reviewer A Query A.1)
- **File:** `research/paper_revised.tex` (Line 229)
- **Target Text:**
  ```latex
  The overall coverage was 98.34\% with a mean set size of 1.01 classes per patient and 98.99\% singleton sets. The non-empty argmax fallback intercepted 8 instances (0.58\%), safely preventing empty sets from being emitted to users.
  ```
- **Replacement Text:**
  ```latex
  The overall operational coverage was 98.34\% (raw conformal coverage without fallback: 97.91\%) with a mean set size of 1.01 classes per patient and 98.99\% singleton sets. The non-empty argmax fallback intercepted 8 instances (0.58\%), safely preventing null sets while flagging each intercepted scan for mandatory clinical examination.
  ```
- **Rationale:** Strictly separates theoretical raw conformal coverage ($97.91\%$) from operational coverage with non-empty fallback ($98.34\%$), resolving the mathematical objection that $Y \in \emptyset$ is False.

---

### Revision 5: AMD Power & Sensitivity Uncertainty Disclosure (Reviewer B Query B.1)
- **File:** `research/paper_revised.tex` (Line 279)
- **Target Text:**
  ```latex
  Consequently, AMD sensitivity (79.31\%) carries wider confidence intervals than DR or glaucoma.
  ```
- **Replacement Text:**
  ```latex
  Consequently, AMD sensitivity (79.31\%, 95\% Wilson CI [61.61\%, 90.15\%]) carries wider uncertainty bounds than DR (100.0\%) or glaucoma (100.0\%), making AMD conformal risk control validation provisional pending dedicated larger cohorts.
  ```
- **Rationale:** Formally discloses that while Diabetic Retinopathy and Glaucoma are definitively validated under $\ge 99.0\%$ emergency coverage, AMD sample size is low ($n=29$), requiring future dedicated clinical studies.

---

## 3. Summary of Manuscript Integrity Status

| Verification Criterion | Pre-Revision Status | Post-Revision Status | Verification Source |
| :--- | :---: | :---: | :--- |
| **BibTeX Citations** | 1 Missing Key (`shen2021eyequality`), 4 Duplicates | **Zero Missing Keys, Zero Duplicates** | `references.bib`, `paper_revised.tex` |
| **Mathematical Precision** | Conformal fallback conflated with raw coverage | **Raw vs. Operational Coverage Explicitly Separated** | Section III-D, Section V-E |
| **Statistical Disclosure** | Stacking superiority assumed by default | **McNemar Parity ($p = 0.6875$) Formally Highlighted** | Section V-A, Table II |
| **Clinical Boundaries** | AMD limitation noted casually | **Precise 95% Wilson CI $[61.61\%, 90.15\%]$ Documented** | Section VI (Limitations) |
| **Ghost Cases Discrepancy** | Zero ($1371 + 18 = 1389$) | **Zero ($1371 + 18 = 1389$) Verified** | Table III, `canonical_eval_test_v2.json` |
