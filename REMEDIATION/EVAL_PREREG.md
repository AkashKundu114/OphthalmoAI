# OphthalmoAI Evaluation Protocol & Statistical Pre-Registration

**Pre-Registration Identifier:** `REMEDIATION/EVAL_PREREG.md`  
**Date:** October 17, 2026  
**Lead Evaluator:** Subagent M (ML Rigor & Statistical Auditing)  
**Target Architecture:** Calibrated Tri-Backbone Soft-Voting Vision Ensemble (`DenseNet-201` + `ConvNeXt-Small` + `EfficientNet-V2-M`) & Standalone `EfficientNet-B4`

---

## 1. Study Objective & Hypothesis

This pre-registration protocol defines the empirical evaluation metrics, statistical significance thresholds, and calibration verification standards for OphthalmoAI.

### Intended Evaluation Scope
The system evaluates color fundus photographs for five primary diagnostic categories:
1. **Normal (Healthy Posterior Pole)**
2. **Diabetic Retinopathy (DR)**
3. **Glaucoma (Glaucomatous Optic Neuropathy)**
4. **Cataract (Optical Media Opacity impacting Fundus Visualization)**
5. **Age-Related Macular Degeneration (AMD)**

---

## 2. Statistical Metrics & Confidence Estimation

All reported benchmark figures must adhere to the following statistical standards:

### 2.1 Discrimination Metrics
- **Balanced Accuracy:** Macro-averaged recall across all classes:
  $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^K \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$$
- **Macro-Averaged F1-Score:** Harmonic mean of macro precision and macro recall.
- **One-vs-Rest AUROC & AUPRC:** Receiver Operating Characteristic and Precision-Recall Area Under Curves computed per disease class using trapezoidal numerical integration.
- **95% Bootstrap Confidence Intervals:** All summary metrics (Accuracy, F1, AUROC) must report 95% empirical bootstrap confidence intervals computed across $B = 1,000$ resamples with replacement.

### 2.2 Probability Calibration Metrics
- **Expected Calibration Error (ECE):** Computed across $M = 15$ equally spaced probability bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score:** Mean squared error between predicted calibrated probability vectors and one-hot ground-truth encodings.
- **Reliability Diagrams:** Pre-calibration vs. post-Platt scaling reliability curves.

### 2.3 Conformal Risk Guarantees (Adaptive Weight Conformal Risk Control - AW-CRC)
- Empirical coverage bound:
  $$P(Y \in C(X)) \ge 1 - \alpha, \quad \alpha = 0.05$$
- Average prediction set size $|C(X)|$ across test samples.

### 2.4 Clinical Safety & Rejection Thresholds
- High uncertainty threshold ($u(X) > \tau_u$, where $u(X)$ is Monte Carlo Dropout predictive entropy). High-uncertainty predictions require human ophthalmologist review.
- Image Quality Assessment (IQA) rejection rate on out-of-distribution or degraded fundus images.

---

## 3. Data Leakage Prevention Protocol

Before calculating any benchmark number:
1. **Patient-Level Isolation:** Verify that all images originating from the same subject / patient ID are strictly confined to either `train`, `val`, or `test`. No patient may span multiple splits.
2. **Perceptual Hash Deduplication:** Run perceptual dHash / pHash scanning with Hamming threshold $\le 4$ across splits to eliminate exact and near-duplicate images.
3. **Disjoint Calibration Set:** The temperature scaling optimization set must be strictly disjoint from the final test set.
