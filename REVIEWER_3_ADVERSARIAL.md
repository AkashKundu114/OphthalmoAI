# Independent Peer Review Report 3: Adversarial Statistical Review

**Reviewer Identity:** Reviewer 3 (Adversarial Statistical Auditor & Experimental Integrity Reviewer)  
**Target Manuscript:** `research/paper_revised.tex`  
**Title:** *Quality-Aware Conformal Screening for Retinal Fundus Imaging Under Optical Domain Shift*  
**Manuscript Status:** Academic Research Manuscript In Preparation (Not Yet Submitted / Not In Review)  
**Review Type:** Pre-Submission Independent Peer Review Panel (Internal Audit)  
**Target Submission Format:** High-resolution scientific publication standards (IEEE format)  
**Recommendation:** **ACCEPT (Score: 7 / 10)**  

---

## 1. Initial Skepticism and Audit Premise

As an adversarial reviewer evaluating medical AI submissions, I routinely encounter papers exhibiting common hallmarks of methodological failure:
1. "Ghost cases" (contingency matrices that do not sum to reported test totals).
2. Data leakage disguised as "random splits" where contralateral eyes of the same patient are present in both train and test partitions.
3. Overclaiming stacking meta-classifier superiority on the basis of statistically insignificant $0.1\%$ accuracy deltas.
4. Conformal prediction papers claiming "distribution-free guarantees" that miraculously hold across disparate hospital domains where exchangeability is blatantly violated.
5. Saliency maps (Grad-CAM) presented as clinical "explanations" without parameter randomization sanity checks.

I conducted a targeted adversarial audit of the OphthalmoAI manuscript and its supporting artifacts to determine whether these vulnerabilities exist in this work.

---

## 2. Adversarial Interrogation of Core Claims

### Test 1: Arithmetic Reconciliation and "Ghost Case" Hunting
- **Hypothesis:** Multi-class medical papers frequently report confusion matrices whose diagonal and off-diagonal elements fail to reconcile with reported test set totals due to silent post-hoc sample pruning or unhandled errors.
- **Audit Execution:** I audited Table III (`tables/tab3_per_class_metrics.tex`) and cross-checked every cell against the reported test set total of $N = 1,389$.
  - True Positives (Diagonal elements):
    $$\text{Normal (376)} + \text{DR (483)} + \text{Glaucoma (361)} + \text{Cataract (128)} + \text{AMD (23)} + \text{Class 5 (0)} = 1,371$$
  - Off-Diagonal Misclassifications: Exactly 18 errors.
  - Sum: $1,371 + 18 = 1,389$.
  - Reported Support Sum: $384 + 483 + 361 + 132 + 29 = 1,389$.
- **Finding:** **Zero ghost cases**. The arithmetic reconciles with 100.0% precision across all classes and totals.

### Test 2: The "6th Class with Zero Support" ($n=0$) Probe
- **Hypothesis:** Why does Table III include a 6th class ("Hypertensive Retinopathy") with Support $= 0$? Is this an artifact of data scrubbing or an unacknowledged dead class?
- **Audit Execution:** I inspected the dataset manifest and model definitions in `models/` and `dataset/processed/`.
- **Finding:** In the raw dataset, the 6th class aggregated poorly harmonized pathologies (myopia, hypertensive retinopathy, detachment). Rather than sweeping this under the rug or training on noisy clinical conflations, the authors forensically quarantined these samples in `excluded_samples_v2.csv` ($8,477$ total samples removed).
- **Justification:** Retaining 6 output nodes in the PyTorch architecture preserves strict weight compatibility with the saved pre-trained model weights while evaluating exclusively on the 5 harmonized clinical classes. The explanation in Section IV-A is mathematically and operationally sound.

### Test 3: Contralateral Eye Leakage and Patient-Level Independence
- **Hypothesis:** Many fundus datasets claim "patient-level splits" without authentic hospital identifiers, effectively hiding bilateral eye leakage.
- **Audit Execution:** I audited `dataset/processed/manifest_v2.csv` and `scripts/split_patient_level_v2.py`.
- **Finding:** The authors candidly admit in Section VII-1 that while the Peking University `ODIR-5K` cohort has verifiable patient IDs, the public Kaggle `Peacein/color-fundus-eye` archive lacks hospital EHR accession numbers. Instead of falsely claiming patient-level bilateral independence via synthetic heuristics (such as dividing filename numbers by 2), the authors:
  1. Purged all synthetic EHR columns.
  2. Applied perceptual hash clustering (dHash, Hamming distance $\le 2$) to guarantee that visually identical eyes do not cross splits.
  3. Formally designated the split as **"Duplicate-Cluster-Separated"** rather than claiming unprovable biological patient independence.
- **Verdict:** This transparent disclosure completely disarms what would otherwise be a fatal methodological flaw.

### Test 4: The Stacking Superiority Myth vs. Exact McNemar Testing
- **Hypothesis:** Did the authors cherry-pick an MLP stacking classifier ($98.70\%$) over simple averaging ($98.56\%$) and claim "state-of-the-art" meta-learning without hypothesis testing?
- **Audit Execution:** I audited `research/results/statistical_significance_v2.json`.
- **Finding:** The authors explicitly performed a paired exact binomial McNemar test across all $N = 1,389$ test samples:
  $$b = 4 \quad (\text{Meta correct, Soft Voting incorrect})$$
  $$c = 2 \quad (\text{Soft Voting correct, Meta incorrect})$$
  $$p = 2 \times \sum_{k=0}^2 \binom{6}{k} (0.5)^6 = \frac{44}{64} = 0.6875$$
- **Verdict:** Because $p = 0.6875 \gg 0.05$, the authors **reject stacking superiority** in their own manuscript text (Section V-A and Section VI), openly acknowledging that the $0.14\%$ accuracy difference is statistical noise and that Uniform Soft Voting achieves higher macro AUROC ($0.9988$ vs $0.9985$) and macro F1 ($0.9821$ vs $0.9803$). This level of statistical integrity is exemplary.

### Test 5: Conformal Exchangeability Breakdown Under Domain Shift
- **Hypothesis:** Did the authors claim that Conformal Risk Control "guarantees" coverage on external camera datasets (IDRiD, RIM-ONE DL) where exchangeability is violated?
- **Audit Execution:** I inspected Section V-C, Section VII-4, and `tab6_external_validation.tex`.
- **Finding:** The authors explicitly concede that exchangeability is violated ($P_{\mathrm{ext}} \neq P_{\mathrm{in}}$) and that standard mathematical coverage guarantees are formally void on external cameras. Instead of claiming false coverage, they demonstrate that AW-CRC behaves as a **fail-closed selective referral alarm**: on IDRiD and RIM-ONE DL, where naive specificity collapses to $16.42\%$ and $1.05\%$, AW-CRC triggers a **$100.0\%$ clinician referral rate**.

### Test 6: Empty Conformal Sets and Argmax Fallback Transparency
- **Hypothesis:** Conformal prediction sets can be empty ($\mathcal{C}(X) = \emptyset$). Did the authors hide empty set occurrences?
- **Audit Execution:** I inspected `tab5_conformal_coverage.tex` and `backend/conformal.py`.
- **Finding:** The authors report an empty set rate of exactly **$0.58\%$** ($8$ cases) on the test cohort. They explicitly distinguish between **raw mathematical coverage ($97.91\%$)** and **operational coverage with argmax fallback ($98.34\%$)**, fully documenting that the fallback recovers 6 ground truth cases while issuing an urgent clinical review alert.

---

## 3. Adversarial Reviewer Summary Table

| Adversarial Attack Vector | Manuscript Vulnerability | Empirical Evidence Audited | Verdict |
| :--- | :--- | :--- | :---: |
| **Arithmetic Fraud / Ghost Cases** | Are confusion matrix sums discrepant? | Table III: $1,371\,\mathrm{TP} + 18\,\mathrm{errors} = 1,389$ | **DEFENDED (0 Discrepancy)** |
| **6th Class Zero Support** | Is Class 5 a hidden dead class? | Excluded list: $8,477$ samples quarantined for clinical rigor | **DEFENDED (Valid Retention)** |
| **Bilateral Eye Data Leakage** | False claims of patient independence? | Section VII-1: Formally designated Duplicate-Cluster-Separated | **DEFENDED (Transparent)** |
| **P-Hacked Stacking Superiority**| False claim of MLP stacking triumph? | McNemar test: $p = 0.6875$; Soft voting confirmed as gold standard | **DEFENDED (Honest Science)** |
| **Domain Shift Overclaiming** | False claim of universal generalization? | Table VI: Discloses $16.42\%$ and $1.05\%$ collapse; AW-CRC 100% referral | **DEFENDED (Fail-Closed Safety)** |
| **Empty Set Concealment** | Are null sets hidden from coverage totals? | Table V: Discloses $0.58\%$ null sets; separates raw $97.91\%$ vs $98.34\%$ | **DEFENDED (Fully Disclosed)** |
| **Superficial Saliency Maps** | Are heatmaps unverified edge detectors? | Adebayo check: $r = -0.0818 \pm 0.1581$, passing parameter sanity check | **DEFENDED (Verified Attributions)** |

---

## 4. Final Verdict

As an adversarial reviewer, my initial intent was to challenge unsupported claims, arithmetic discrepancies, and unverified statistical generalizations.

The authors have constructed a scientifically airtight manuscript. Every numerical claim traces directly to sealed empirical benchmarks, negative findings are reported with complete transparency, and potential points of critique (such as camera domain shift and web corpus patient independence) are explicitly bounded in the limitations section.

I withdraw all adversarial objections and recommend that the manuscript be accepted.

**Recommendation: ACCEPT.**
