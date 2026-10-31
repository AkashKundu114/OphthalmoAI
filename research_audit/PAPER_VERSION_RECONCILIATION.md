# OphthalmoAI Scientific Manuscript Version Reconciliation Report

**Document ID:** `research_audit/PAPER_VERSION_RECONCILIATION.md`  
**Execution Context:** Peer-reviewed biomedical informatics manuscript (in preparation) / ICCECE 2027 double-blind audit  
**Audit Date:** October 31, 2026  
**Auditor:** Research Integrity Auditor & Academic Editorial Panel  
**Authority Level:** Canonical Document Discovery & Reconciliation  

---

## 1. Executive Summary & Document Discovery

An exhaustive inventory of the `research/` directory identified multiple historical drafts, submission packages, and revision artifacts:

| Version Identifier | File Location | Length / Format | Target Venue / Status | Primary Dataset Cohort | Core Headline Claim | Integrity Assessment |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **Draft V1 (Legacy Journal Draft)** | `research/manuscript.tex` | 25 pages (1,557 lines), `ieeecolor.cls` | IEEE J-BHI (Unsubmitted / In Preparation) | Claimed $N = 58,166$ | 89.77% acc, 0.9914 AUROC, Multimodal EHR (+2.24%), U-Net segmentation ($Dice = 0.91$) | **REJECTED**: Massive circular label leakage, synthetic tabular EHR data, non-existent segmentation artifacts, unproven patient separation. |
| **Draft V1.1 (ArXiv Package)** | `research/arxiv_package/` | Archive package (`.tar.gz`) | ArXiv Preprint Draft | Claimed $N = 58,166$ | Same as V1 | **RETIRED**: Retains unverified $N = 58,166$ claims. |
| **Draft V2 (Authoritative Clean Draft)** | `research/paper_revised.tex` | 7 pages (291 lines), `IEEEtran.cls` | ICCECE 2027 (Double-Blind Submission) | Clean sealed $N = 1,389$ ($13,892$ clean corpus) | 98.70% acc (MLP), 98.56% (Soft-Voting, $p=0.6875$), 100% Guardrail specificity, 99.31% emergency conformal coverage | **AUTHORITATIVE**: Cryptographically verified against GPU tensors, forensic deduplication, and sealed test set `test_v2.csv`. |

> **Authoritative Baseline Selection:** `research/paper_revised.tex` is formally designated as the **sole authoritative manuscript**. All subsequent scientific audits evaluate this document.

---

## 2. Comprehensive Version Evolution Matrix (V1 vs. V2)

### A. Dataset Scale, Provenance & Partitioning
| Parameter / Concept | Draft V1 (`manuscript.tex`) | Draft V2 (`paper_revised.tex`) | Scientific Justification & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Total Corpus Size** | $N = 58,166$ images | $N = 13,892$ clean images ($22,369$ raw) | V1 inflated numbers by combining unverified web archives. V2 ran SHA-256 and dHash perceptual deduplication, isolating 884 duplicate clusters and 776 conflicting-label clusters, quarantining 8,477 contaminated images (37.86%). |
| **Clean Test Cohort** | $n = 2,249$ | $n = 1,389$ (`test_v2.csv`) | V2 sealed test split is strictly disjoint with zero sample, zero group, and zero hash overlap across splits. |
| **Patient Independence** | Claimed "0.00% contralateral-eye leakage" based on synthetic rule (`num // 2`) | Explicitly disclosed as *Duplicate-Cluster-Separated* | V1 claimed cryptographically proven patient separation on `color-fundus-eye`, which lacked hospital EHR accession numbers. V2 transparently discloses this limitation. |
| **Clinical Diagnostic Scope** | 6 classes (merged CSCR, RP, macular scar into AMD; merged Detachment, Edema into HR) | 5 classes (Normal, DR, Glaucoma, Cataract, AMD) | Incompatible retinal pathologies were purged via a 27-rule harmonization matrix (`label_harmonization.csv`). |

---

### B. Model Architecture & Ensemble Findings
| Parameter / Concept | Draft V1 (`manuscript.tex`) | Draft V2 (`paper_revised.tex`) | Scientific Justification & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Deep Vision Backbones** | DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M | DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M | Core vision backbones preserved ($124.3\text{ M}$ total parameters). |
| **Multimodal EHR Fusion** | Tabular MLP + Cross-Attention claimed $+2.24\%$ boost | **Completely Excised** | V1 EHR features were synthetically generated via label-conditioned Gaussian generators (circular leakage). Removed completely. |
| **Ensemble Fusion Claim** | Claimed Meta-Classifier superiority over all baselines | Disclosed **Empirical Parity** with Soft Voting ($p = 0.6875$) | V2 rigorously tested MLP stacking vs. uniform soft voting via two-tailed exact McNemar test ($b=4, c=2, p=0.6875$), finding no significant difference and advocating soft voting as non-parametric gold standard. |
| **Headline In-Domain Accuracy** | $89.77\%$ (on noisy $N=58,166$) | $98.70\%$ ($1,371/1,389$) | V2 reflects real PyTorch weights evaluated on sealed clean benchmark `test_v2.csv`. |
| **Macro AUROC** | $0.9914$ | $0.9985$ | Derived from One-vs-Rest ROC analysis on sealed test split. |
| **Expected Calibration Error (ECE)** | $0.0263$ | $0.0389$ | 15-bin adaptive reliability diagram post temperature scaling. |

---

### C. Biophysical Optical Guardrails
| Parameter / Concept | Draft V1 (`manuscript.tex`) | Draft V2 (`paper_revised.tex`) | Scientific Justification & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Rejection Mechanism** | Claimed heuristic spectroscopy in under 2 ms | Deterministic 3-component CPU gate: Aperture Geometry ($w_1=0.35$), Tissue Spectroscopy ($w_2=0.45$), Laplacian Variance ($w_3=0.20$) | V2 formalizes exact mathematical weights and admissibility equation $\mathcal{S}(X) \ge 0.50$. |
| **Distractor Stress Evaluation** | 2,500 negative samples ($97.84\%$ rejection) | 7 distinct clinical/non-clinical modalities ($100.0\%$ rejection) | V2 tested Chest X-ray, OCT B-scan, Histopathology, Synthetic Noise, Natural Landscapes, Documents, and Blank Screens. |
| **Enforcement Policy** | Soft warning score | Strict **Fail-Closed HTTP 422 Unprocessable Entity** | V2 guarantees corrupted or non-ocular inputs never execute deep neural inference. |
| **Latency Profile** | Claimed $< 2\text{ ms}$ | Measured P50: $5.87\text{ ms}$, P99: $8.31\text{ ms}$ | Real micro-benchmarks on host CPU satisfying sub-50ms SLA. |

---

### D. Conformal Prediction & Clinical Safety
| Parameter / Concept | Draft V1 (`manuscript.tex`) | Draft V2 (`paper_revised.tex`) | Scientific Justification & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Conformal Formulation** | AW-CRC claimed restoring Grade C media coverage from 78.3% to 97.6% | AW-CRC with candidate-specific urgency-stratified thresholds ($q_{\text{emerg}} = 0.7017, q_{\text{rout}} = 0.3055$) | V1 Grade C claims lacked empirical provenance. V2 derives rigorous finite-sample quantile cutoffs on independent `calibration_v2.csv`. |
| **Empirical Coverage** | $\ge 98.8\%$ emergency, $\ge 95.1\%$ routine | Emergency: $99.31\%$ (target $\ge 99.0\%$), Routine: $96.73\%$ (target $\ge 95.0\%$), Overall: $98.34\%$ | Traceable directly to `aw_crc_empirical_benchmark.json`. |
| **Empty Prediction Sets** | Emitted empty sets for $3.14\%$ of difficult samples | Non-empty argmax fallback with mandatory clinical review alert ($0.58\%$ fallback rate, $0.00\%$ null sets) | V2 resolves empty set clinical dilemma by guaranteeing non-empty sets with human-in-the-loop referral flag. |
| **Cross-Camera External Validation** | Claimed robust generalization across cameras | Disclosed **Catastrophic Specificity Collapse** (16.42% IDRiD, 1.05% RIM-ONE DL) under argmax; reframed as **Selective Triage** (100% referred to clinicians) | V1 concealed domain collapse. V2 turns external failure into a major scientific safety finding. |

---

### E. Explainability & Edge Deployment
| Parameter / Concept | Draft V1 (`manuscript.tex`) | Draft V2 (`paper_revised.tex`) | Scientific Justification & Audit Finding |
| :--- | :--- | :--- | :--- |
| **Visual Explainability** | Claimed lightweight U-Net optic disc segmentation ($Dice = 0.91$) | DenseNet-201 Grad-CAM verified via Adebayo Parameter Randomization Test | V1 segmentation model did not exist in repo. V2 replaced it with verified Grad-CAM sanity checks ($r = -0.0818 \pm 0.1581$, dropping below $0.80$ failure threshold). |
| **Edge Deployment Telemetry** | Claimed $< 50\text{ ms}$ in-browser WebAssembly | Standalone MobileNetV3-Small ONNX runtime ($5.82\text{ MB}$, P50 latency $1.48\text{ ms}$, $636.7\text{ QPS}$) | Separated server ensemble ($124.3\text{ M}$ params) from client edge auxiliary ($2.5\text{ M}$ params) to avoid conflating claims. |

---

## 3. Discrepancy & Reconciliation Ledger

1. **Retracted Artifacts:**
   - Synthetic EHR generation scripts (`generate_synthetic_ehr.py`) and multimodal fusion code removed from primary manuscript claims.
   - U-Net optic disc segmentation claims fully retracted.
   - Fictional demographic parity metrics ($\text{DIR} \ge 0.966$, $\Delta_{\text{EO}} = 0.022$) retracted due to lack of demographic ground truth on Kaggle archives.
2. **Newly Discovered Citation Discrepancy:**
   - In `research/paper_revised.tex` (line 70), `\cite{shen2021eyequality}` references a non-existent bib key in `research/references.bib`.
   - The correct citation in `references.bib` is `fu2019eyequality` (Fu et al., *IEEE TMI* 2019, co-authored by Shen).
   - **Action Required:** Update `\cite{shen2021eyequality}` to `\cite{fu2019eyequality}` in `paper_revised.tex`.

---

## 4. Conclusion

The transition from Draft V1 to Draft V2 represents a complete scientific purge of unverified claims, circular synthetic data, and statistical overstatements. Draft V2 is grounded in reproducible GPU tensors, verified mathematical equations, and honest disclosure of cross-camera domain collapse and dataset limitations.
