# IEEE Conference Peer Review Report — Reviewer B (Clinical AI Safety & Ophthalmology)

**Reviewer Identity:** Reviewer B (Senior Clinical Ophthalmologist & Medical AI Validation Researcher)  
**Manuscript Title:** Quality-Aware Conformal Screening for Retinal Fundus Imaging Under Optical Domain Shift  
**Target Venue:** ICCECE 2027 (Double-Blind Review)  
**Recommendation:** **ACCEPT WITH MINOR REVISION**  
**Review Confidence:** 5/5 (Specialist in Retinal Disease Screening, Tele-Ophthalmology, and Medical Device Safety)  

---

## 1. Overall Clinical Assessment
Automated deep learning systems for retinal photography often suffer from high optimism in academic literature but fail dramatically when brought into real-world eye clinics. This manuscript stands out because the authors proactively confront the two most dangerous failure modes in clinical ophthalmic AI:
1. **Unmonitored silent misclassification under camera domain shifts** (e.g. moving between Kowa and Nidek cameras).
2. **Symmetric loss functions that penalize non-sight-threatening errors equally with sight-threatening irreversible blindness**.

The introduction of Admissibility-Weighted Conformal Risk Control (AW-CRC) with candidate-specific urgency cutoffs ($\alpha_{\mathrm{emerg}} = 0.01$ for DR, Glaucoma, AMD vs. $\alpha_{\mathrm{rout}} = 0.05$ for Cataract, Normal) aligns directly with clinical triage principles.

---

## 2. Clinical Strengths
1. **Prioritization of Vision-Threatening Pathologies:** Enforcing a $\ge 99.0\%$ coverage constraint on emergency conditions ensures that diabetic macular edema, proliferative retinopathy, and glaucomatous optic cupping are rarely omitted from the differential prediction set. In the clean test set, empirical emergency coverage reached $99.31\%$.
2. **Transparent Domain Shift Disclosure:** In Table VI, the authors disclose that naive argmax classification yields a catastrophic specificity collapse down to $16.42\%$ on IDRiD and $1.05\%$ on RIM-ONE DL. Instead of hiding this under misleading sensitivity-only headlines, the authors show that AW-CRC selective classification safely routes 100.0% of these shifted scans to human ophthalmologists.
3. **Rigorous Clinical Decontamination:** Purging conflicting-label duplicates and excluding clinically incompatible diseases (such as mapping central serous chorioretinopathy or retinal detachment into AMD) demonstrates commendable clinical integrity.
4. **Deterministic Optical Gate:** Halting non-ocular inputs and severe blur on the CPU before deep inference prevents false reassurance in primary care settings.

---

## 3. Clinical Objections & Clarifications Required

### Clinical Query B.1: Low AMD Prevalence and Diagnostic Power
In the clean sealed test set ($N = 1,389$), there are only $n = 29$ verified AMD cases ($2.09\%$ prevalence). The model achieves $79.31\%$ sensitivity on AMD ($23 / 29$), missing 6 cases under argmax.
- **Clinical Implication:** While DR ($483/483$) and Glaucoma ($361/361$) achieved $100\%$ sensitivity, an AMD sensitivity of $79.31\%$ with a wide 95% Wilson CI of $[61.61\%, 90.15\%]$ means the system is not yet validated for standalone AMD screening.
- **Required Action:** The authors must emphasize in the Abstract, Discussion, and Limitations that OphthalmoAI's emergency guarantees are clinically validated for DR and Glaucoma, but AMD screening remains provisional pending larger cohorts.

### Clinical Query B.2: Impact of 100% Referral Rate on Clinical Workload
When evaluating external camera datasets (IDRiD and RIM-ONE DL), AW-CRC escalates $100.0\%$ of images to human review.
- **Clinical Dilemma:** While a $100\%$ referral rate is medically safe (preventing unmonitored false negatives), in a deployed clinic it results in an unmanageable clinical bottleneck where the AI provides zero workload reduction.
- **Required Action:** The Discussion must explicitly state that while selective triage provides safe fail-closed behavior, practical deployment across new camera hardware requires local camera-specific color harmonization or site-level conformal re-calibration.

### Clinical Query B.3: Optical Guardrail Behavior on Dense Cataract Eyes
Does the optical gate ($\tau_{\mathrm{guard}} = 0.50$) reject eyes with dense mature cataract due to poor contrast?
- **Clarification:** In the paper, Cataract achieves $96.97\%$ sensitivity in Tier-1. The authors should briefly clarify whether eyes with advanced cataract pass the Tier-0 gate and whether the cataract class is intended for surgical referral.

---

## 4. Minor Clinical Recommendations
1. Ensure the disclaimer that OphthalmoAI is an "experimental research prototype, not a certified medical device" is prominently retained.
2. In Table III, annotate the 6 AMD misclassifications (5 classified as Normal, 1 as Cataract) with clinical commentary regarding early non-exudative drusen contrast.

---

## 5. Reviewer Recommendation
**Accept with Minor Revision.** The authors have addressed the most critical clinical safety boundaries with rare intellectual honesty. With the minor clarifications outlined above, this paper will serve as an exemplary model for safe translational medical AI.
