# IEEE Conference Peer Review Report — Reviewer A (Methodology & Systems)

**Reviewer Identity:** Reviewer A (IEEE Senior Reviewer / ML & Statistical Systems Specialist)  
**Manuscript Title:** Quality-Aware Conformal Screening for Retinal Fundus Imaging Under Optical Domain Shift  
**Target Venue:** ICCECE 2027 (Double-Blind Review)  
**Recommendation:** **ACCEPT WITH MINOR REVISION**  
**Review Confidence:** 5/5 (Expert in Deep Learning Ensembles, Conformal Prediction, and MLOps)  

---

## 1. Overall Impression & Summary of Work
The paper presents an engineering and scientific framework for multi-class retinal fundus screening under optical domain shift. The core contributions consist of: (1) a deterministic CPU optical plausibility gate based on aperture geometry and hemoglobin spectroscopy; (2) a calibrated tri-backbone deep neural ensemble evaluated against soft voting via McNemar testing; (3) Admissibility-Weighted Conformal Risk Control (AW-CRC) with candidate-specific urgency strata; and (4) an evaluation of cross-camera domain collapse on IDRiD and RIM-ONE DL.

The manuscript is refreshingly honest compared to typical medical AI submissions: it admits parity between stacking and soft voting ($p = 0.6875$), discloses catastrophic external specificity collapse under naive argmax (16.42% and 1.05%), and provides verifiable arithmetic consistency across its contingency matrices.

---

## 2. Strengths
1. **Flawless Arithmetic Reconciliation:** In Table III, the contingency matrix diagonal sum ($1,371$) plus off-diagonal errors ($18$) equals exactly $1,389$ test samples, with zero ghost cases.
2. **Methodological Rigor in Ensembling:** The authors conduct a two-tailed exact binomial McNemar test showing that the MLP stacking classifier ($98.70\%$) is statistically indistinguishable from uniform soft voting ($98.56\%$, $p = 0.6875$). Recommending soft voting as a non-parametric gold standard to prevent meta-overfitting is scientifically exemplary.
3. **Deterministic Fail-Closed Guardrail:** Intercepting non-fundus distractors on the CPU in $5.87\text{ ms}$ with 100.0% specificity across seven distinct modalities before allocating GPU memory is an excellent systems engineering choice.
4. **Reproducibility & Integrity:** All manifests, model weights, and calibration JSON files have published SHA-256 hashes and trace directly to Python execution paths.

---

## 3. Major Concerns & Technical Queries

### Query A.1: The Mathematical Impact of Empty-Set Fallback on Conformal Coverage
In Section III-D and Section V-E, the authors state that when the conformal prediction set is empty ($\mathcal{C}(X) = \emptyset$), the system injects the top-1 argmax prediction and flags the case with an urgent human review alert. The authors report that AW-CRC achieves $98.34\%$ overall coverage with a $0.58\%$ fallback rate ($8$ cases).
- **The Mathematical Issue:** Under standard conformal prediction theory, if $\mathcal{C}(X) = \emptyset$, the event $Y \in \mathcal{C}(X)$ is strictly False. The raw mathematical coverage of the conformal set prior to fallback is $97.91\%$ ($1,360 / 1,389$), not $98.34\%$. Injecting argmax is an engineering heuristic.
- **Required Action:** The manuscript must explicitly report both the raw mathematical conformal coverage ($97.91\%$) and the fallback-augmented operational coverage ($98.34\%$), explaining the exact distinction in the text of Section V-E.

### Query A.2: Temperature Scaling Order of Operations
In Section III-C, the manuscript describes Platt temperature scaling for each backbone. Is temperature scaling applied to each model's logits *before* computing the soft voting average, or is a single temperature applied to the aggregated ensemble logits?
- **Clarification Needed:** If soft voting averages probabilities from models with different temperature scales, does the calibration hold post-fusion? The manuscript should clarify this order of operations.

### Query A.3: Edge vs. Server Architectural Distinction
In Section III-A and Section V-H, the manuscript introduces an edge auxiliary runtime (MobileNetV3-Small ONNX, $5.82\text{ MB}$, $1.48\text{ ms}$ latency). 
- **Concern:** Is MobileNetV3-Small trained as a student distilled from the tri-backbone ensemble, or is it an independently trained lightweight baseline? The authors must ensure that accuracy claims for the 124.3M parameter server ensemble are never conflated with the edge model.

---

## 4. Minor Corrections & Polish
1. In Section II-B, line 70, the citation `\cite{shen2021eyequality}` triggers a BibTeX warning because the database entry is authored by Fu et al. (IEEE TMI 2019). Update the citation key to `\cite{fu2019eyequality}`.
2. In Table II, report the exact 95% bootstrap confidence intervals for all models to enable direct visual comparison alongside point estimates.
3. In Table V, add an explicit column showing the raw coverage without argmax fallback.

---

## 5. Reviewer Recommendation
**Accept with Minor Revision.** This is one of the most meticulously documented and statistically defensible submissions I have reviewed this cycle. Once the authors clarify the empty set fallback coverage and citation key, the paper will be ready for publication.
