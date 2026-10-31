# OphthalmoAI Mathematical & Theoretical Audit Report

**Document ID:** `research_audit/MATHEMATICAL_REVIEW.md`  
**Execution Context:** Peer-reviewed biomedical informatics manuscript (in preparation) / ICCECE 2027  
**Audit Date:** October 31, 2026  
**Auditor:** Conformal Prediction Specialist, Statistical Reviewer, Senior Deep Learning Engineer  
**Status:** COMPLETE & AUDITED  

---

## 1. Mathematical Notation & Formulation Inventory

The manuscript formalizes retinal screening as a multi-class selective classification problem under optical quality constraints. We audit the consistency of mathematical symbols across `research/paper_revised.tex`:

| Symbol | Mathematical Definition | Space / Domain | Audit Finding |
| :--- | :--- | :---: | :--- |
| $X$ | Raw RGB fundus image | $\mathbb{R}^{H \times W \times 3}$ | Standard tensor definition ($384 \times 384 \times 3$). |
| $Y$ | Ground truth diagnostic class | $\mathcal{Y} = \{1, 2, \dots, K\}$, $K=5$ | 5 classes: Normal, DR, Glaucoma, Cataract, AMD. |
| $\mathcal{S}(X)$ | Biophysical optical admissibility score | $[0.0, 1.0] \subset \mathbb{R}$ | Convex combination of 3 components ($\sum w_i = 1.0$). |
| $\tau_{\mathrm{guard}}$ | Pre-inference guardrail admission cutoff | $\tau_{\mathrm{guard}} = 0.50$ | Fail-closed threshold below which inference is aborted. |
| $\mathbf{z}_m(X)$ | Raw unnormalized logit vector of model $m$ | $\mathbb{R}^K$ | Models: EfficientNet-V2-M, DenseNet-201, ConvNeXt-Small. |
| $T_m$ | Platt temperature scaling parameter | $\mathbb{R}_{>0}$ | Calibrated on `calibration_v2.csv` via L-BFGS. |
| $\hat{P}(k \mid X)$ | Calibrated posterior probability for class $k$ | $\Delta^{K-1} = \{p \in [0, 1]^K : \sum p_k = 1\}$ | Derived via temperature-scaled softmax or soft voting. |
| $s_{\mathrm{AW}}(X, y)$ | Admissibility-weighted non-conformity score | $[0.0, 1.0] \subset \mathbb{R}$ | Modulated by $\mathcal{S}(X)^\gamma$, where $\gamma = 1.0$. |
| $q_{\mathrm{strata}(k)}$ | Calibrated conformal quantile threshold | $[0.0, 1.0] \subset \mathbb{R}$ | Candidate-specific cutoffs: $q_{\mathrm{emerg}}, q_{\mathrm{rout}}$. |
| $\mathcal{C}(X)$ | Conformal prediction confidence set | $\mathcal{P}(\mathcal{Y}) \setminus \{\emptyset\}$ | Set of admitted diagnostic hypotheses. |

---

## 2. Equation-by-Equation Audit

### Equation (1): Optical Admissibility Score
$$\mathcal{S}(X) = w_1 \mathcal{A}_{\mathrm{geom}}(X) + w_2 \mathcal{C}_{\mathrm{spec}}(X) + w_3 \mathcal{V}_{\mathrm{lap}}(X)$$
- **Manuscript Statement:** Weights $w_1 = 0.35, w_2 = 0.45, w_3 = 0.20$ ($\sum w_i = 1.0$).
- **Verification:** Verified against `backend/validators.py`.
- **Mathematical Soundness:** Because each constituent metric is normalized to $[0, 1]$, $\mathcal{S}(X) \in [0, 1]$ is guaranteed.
- **Audit Status:** **VALID & CONSISTENT**.

### Equation (2): Biophysical Spectroscopy Boundaries
$$1.25 \le \frac{\mu_R}{\mu_G} \le 2.50 \quad \text{and} \quad \frac{\mu_B}{\mu_R} < 0.45$$
- **Manuscript Statement:** $\mu_R, \mu_G, \mu_B$ represent mean channel intensities computed strictly over the valid retinal aperture mask ($I_{\mathrm{gray}} > 15$).
- **Verification:** Verified in `backend/validators.py` lines 80-105.
- **Biophysical Basis:** Human retinal tissue is densely vascularized with oxygenated and deoxygenated hemoglobin, which exhibits peak absorbance in the green-blue spectrum ($500-580\text{ nm}$) and high reflectance in the red spectrum ($>600\text{ nm}$). Non-fundus distractors (e.g., chest X-rays, histopathology, screenshots) severely violate these ratios.
- **Audit Status:** **VALID & BIOPHYSICALLY SOUND**.

### Logit Temperature Scaling
$$\hat{P}_m(y = k \mid X) = \frac{\exp(z_{m, k} / T_m)}{\sum_{j=1}^K \exp(z_{m, j} / T_m)}$$
- **Manuscript Statement:** $T_m > 0$ is optimized via L-BFGS on independent calibration split (`calibration_v2.csv`) to minimize Negative Log-Likelihood (NLL).
- **Verification:** Verified in `scripts/calibrate_conformal.py` and stored in `models/calibration_temperature.json`:
  - $T_{\mathrm{effnet}} = 1.1214$
  - $T_{\mathrm{dense}} = 1.1843$
  - $T_{\mathrm{convnext}} = 1.1528$
- **Mathematical Soundness:** Platt temperature scaling is strictly rank-preserving (does not alter top-1 argmax predictions) while contracting overconfident logit distributions.
- **Audit Status:** **VALID**.

### Equation (3): AW-CRC Non-Conformity Scoring
$$s_{\mathrm{AW}}(X, y) = (1 - \hat{P}(y \mid X)) \cdot \mathcal{S}(X)^\gamma$$
- **Manuscript Statement:** Optical score weighting parameter $\gamma = 1.0$.
- **Verification:** Implemented in `backend/conformal.py` and `models/aw_crc_calibration.json`.
- **Mathematical Mechanics:**
  1. For high-quality scans ($\mathcal{S}(X) \approx 1.0$), $s_{\mathrm{AW}}(X, y) \approx 1 - \hat{P}(y \mid X)$, recovering standard split-conformal non-conformity.
  2. For optically degraded scans ($\mathcal{S}(X) \to \tau_{\mathrm{guard}} = 0.50$), $s_{\mathrm{AW}}(X, y)$ is scaled down by a factor of $\mathcal{S}(X)$, meaning even classes with lower softmax probabilities satisfy the inclusion condition $s_{\mathrm{AW}} \le q$.
  3. Consequently, optical ambiguity automatically inflates set cardinality $|\mathcal{C}(X)|$, naturally triggering the multi-class referral policy ($|\mathcal{C}(X)| \ge 2$) for degraded scans.
- **Audit Status:** **VALID & INGENIOUS**.

### Quantile Index Calculation
$$\beta = \frac{\lceil(n+1)(1-\alpha)\rceil}{n}$$
- **Manuscript Statement:** Applied separately for emergency ($\alpha_{\mathrm{emerg}} = 0.01$) and routine ($\alpha_{\mathrm{rout}} = 0.05$) strata.
- **Exact Stratum Sample Counts in `calibration_v2.csv` ($N = 1,389$):**
  - Emergency Stratum ($n_{\mathrm{emerg}} = 869$):
    $$\lceil (869 + 1) \times (1 - 0.01) \rceil = \lceil 870 \times 0.99 \rceil = \lceil 861.3 \rceil = 862$$
    $$\beta_{\mathrm{emerg}} = \frac{862}{869} \approx 0.991945$$
    Empirical calibrated quantile cutoff: $q_{\mathrm{emerg}} = \mathbf{0.701732}$.
  - Routine Stratum ($n_{\mathrm{rout}} = 520$):
    $$\lceil (520 + 1) \times (1 - 0.05) \rceil = \lceil 521 \times 0.95 \rceil = \lceil 494.95 \rceil = 495$$
    $$\beta_{\mathrm{rout}} = \frac{495}{520} \approx 0.951923$$
    Empirical calibrated quantile cutoff: $q_{\mathrm{rout}} = \mathbf{0.305541}$.
- **Verification:** Calibrated values in `models/aw_crc_calibration.json` match to 6 decimal places.
- **Audit Status:** **EXACT MATHEMATICAL TRUTH**.

### Equation (4): Urgency-Stratified Prediction Set Rule
$$\mathcal{C}(X) = \left\{k \in \mathcal{Y} : (1 - \hat{P}(k \mid X)) \cdot \mathcal{S}(X)^\gamma \le q_{\mathrm{strata}(k)}\right\}$$
- **Manuscript Statement:** Evaluates each candidate disease against its own stratum threshold.
- **Audit Status:** **VALID**.

---

## 3. Theoretical Soundness & Boundary Condition Audit

### A. The Exchangeability Assumption & External Domain Shift
Standard conformal prediction theory guarantees:
$$\mathbb{P}_{(X, Y) \sim \mathcal{D}}\left(Y \in \mathcal{C}(X)\right) \ge 1 - \alpha$$
**Critical Assumption:** The calibration samples $(X_1, Y_1), \dots, (X_n, Y_n)$ and the test sample $(X_{n+1}, Y_{n+1})$ must be **exchangeable** (drawn identically and independently from the same distribution $\mathcal{D}$).

- **Audit Finding:**
  1. On in-domain clean test data (`test_v2.csv`), exchangeability holds. Empirical coverage ($99.31\%$ emergency, $96.73\%$ routine) strictly satisfies the theoretical guarantees ($\ge 99.0\%$ and $\ge 95.0\%$).
  2. On external camera cohorts (IDRiD, RIM-ONE DL), exchangeability is **severely violated** due to hardware optical shifts (illumination, sensor response, field of view).
  3. Consequently, mathematical coverage guarantees **do not apply** to external cameras without local re-calibration.
- **Manuscript Compliance:** Section III-D and Section V-B in `paper_revised.tex` explicitly clarify this limitation and frame AW-CRC on external cameras as a **selective triage detector** rather than a coverage guarantee.

### B. The Empty Set Conundrum & Non-Empty Fallback Mechanics
Under standard non-conformity thresholding, if for all classes $k \in \mathcal{Y}$:
$$(1 - \hat{P}(k \mid X)) \cdot \mathcal{S}(X)^\gamma > q_{\mathrm{strata}(k)}$$
the formal conformal set is empty: $\mathcal{C}(X) = \emptyset$.

- **Theoretical Consequence:** When $\mathcal{C}(X) = \emptyset$, the coverage condition $Y \in \mathcal{C}(X)$ is strictly **False** (coverage = 0%).
- **Empirical Frequency on Sealed Test Set ($N = 1,389$):**
  - Standard US-CRC produces 5 empty sets ($0.36\%$).
  - AW-CRC produces 8 empty sets ($0.58\%$).
- **Raw Conformal Coverage vs. Fallback Coverage:**
  - True raw conformal coverage ($Y \in \mathcal{C}(X)$ with $\emptyset$ treated as failure): **$97.91\%$**.
  - Fallback-augmented coverage (injecting top-1 argmax): **$98.34\%$**.
- **Audit Recommendation:** The manuscript must explicitly report both numbers (true coverage $97.91\%$ and fallback-augmented coverage $98.34\%$), explaining that empty sets trigger an immediate mandatory clinician review flag.

---

## 4. Conclusion

All mathematical formulations in `research/paper_revised.tex` are derived correctly, code-reconciled, and grounded in reproducible calibration files. The manuscript's treatment of exchangeability violation and empty set fallback is scientifically defensible.
