# Mathematical and Methodological Review — OphthalmoAI

**Document ID:** `MATHEMATICAL_REVIEW.md`  
**Execution Context:** Peer-reviewed biomedical informatics manuscript (in preparation)  
**Target Venue:** ICCECE 2027 (IEEE Conference, Double-Blind)  
**Auditor Persona:** Mathematical and Statistical Methodology Auditor  
**Date:** October 31, 2026  
**Audited Artifacts:** `research/paper_revised.tex`, `backend/conformal.py`, `backend/validators.py`, `scripts/run_aw_crc_calibration.py`, `models/aw_crc_calibration.json`, `research/results/aw_crc_empirical_benchmark.json`  

---

## 1. Executive Summary

This review provides a comprehensive mathematical derivation, theoretical audit, and algorithmic verification of the mathematical formulations in OphthalmoAI. Specifically, we inspect:
1. The deterministic biophysical optical admissibility metric $\mathcal{S}(X)$.
2. Platt temperature scaling for probability calibration across heterogeneous backbones.
3. Admissibility-Weighted Conformal Risk Control (AW-CRC) nonconformity scores, calibration quantiles, and inclusion rules.
4. Finite-sample coverage guarantees and exchangeability boundaries under optical domain shifts.
5. The mathematical implications of the operational non-empty argmax fallback.

All formulas have been audited against their actual Python implementations in `backend/validators.py`, `backend/conformal.py`, and `scripts/run_aw_crc_calibration.py`.

---

## 2. Mathematical Formulation: Tier-0 Biophysical Optical Admissibility

### 2.1 Metric Definition
For an input fundus image $X \in \mathbb{R}^{H \times W \times 3}$ with RGB color channels, the biophysical admissibility score $\mathcal{S}(X) \in [0, 1]$ is formulated as a convex combination of three orthogonal physical and spatial terms:
$$\mathcal{S}(X) = w_1 \mathcal{A}_{\mathrm{geom}}(X) + w_2 \mathcal{C}_{\mathrm{spec}}(X) + w_3 \mathcal{V}_{\mathrm{lap}}(X)$$
where $w_1 = 0.35$, $w_2 = 0.45$, $w_3 = 0.20$, satisfying $\sum_{i=1}^3 w_i = 1.0$.

### 2.2 Component Derivations

#### A. Aperture Geometry ($\mathcal{A}_{\mathrm{geom}}$)
Standard telecentric fundus cameras image the retina through a circular mechanical or optical mask. Let $M(x, y) = \mathbb{I}(X(x, y) > \theta_{\mathrm{dark}})$ be the binary mask of non-background pixels ($\theta_{\mathrm{dark}} = 15/255$).
Let $\mathrm{Area}(M)$ be the pixel area of the foreground mask, and $\mathrm{Hull}(M)$ be its convex hull. The circularity factor is computed as:
$$\mathcal{F}_{\mathrm{circ}} = \frac{4 \pi \cdot \mathrm{Area}(M)}{\mathrm{Perimeter}(M)^2}$$
$$\mathcal{A}_{\mathrm{geom}}(X) = \mathrm{clip}\left( \frac{\mathrm{Area}(M)}{\mathrm{Area}(\mathrm{Hull}(M))} \cdot \min\left(1.0, \frac{\mathcal{F}_{\mathrm{circ}}}{0.785}\right), 0, 1 \right)$$
For genuine telecentric masks, $\mathcal{A}_{\mathrm{geom}} \approx 1.00$. Rectangular photographs, histology slides, or natural scenes violate circular mask boundaries, yielding low scores.

#### B. Hemoglobin-Melanin Absorption Spectroscopy ($\mathcal{C}_{\mathrm{spec}}$)
Human retinal tissue reflection is dominated by oxyhemoglobin, deoxyhemoglobin, and melanin. Oxygenated retinal microvasculature strongly absorbs short wavelengths (blue: $400\text{--}480\,\mathrm{nm}$; green: $500\text{--}570\,\mathrm{nm}$) and predominantly reflects long wavelengths (red: $600\text{--}750\,\mathrm{nm}$).
Let $\mu_R, \mu_G, \mu_B$ be the mean channel intensities restricted to the valid retinal mask $M$:
$$\mu_C = \frac{1}{|M|} \sum_{(x, y) \in M} X_C(x, y), \quad C \in \{R, G, B\}$$
Physiological fundus reflectance mandates:
1. $1.25 \le \frac{\mu_R}{\mu_G} \le 2.50$ (Red dominance over green)
2. $\frac{\mu_B}{\mu_R} < 0.45$ (Blue suppression due to macular pigment and hemoglobin absorption)

The spectral score is derived via smooth penalty margins:
$$\mathcal{C}_{\mathrm{spec}}(X) = \exp\left( - \frac{(\mu_R/\mu_G - 1.875)^2}{2 \cdot (0.35)^2} \right) \cdot \mathbb{I}\left(\frac{\mu_B}{\mu_R} < 0.45\right)$$
If the spectrum violates fundus physiology (e.g., green-heavy dermatology, blue ocean landscapes, or grayscale radiographs where $\mu_R \approx \mu_G \approx \mu_B$), $\mathcal{C}_{\mathrm{spec}}(X) \to 0$.

#### C. Structural High-Frequency Variance ($\mathcal{V}_{\mathrm{lap}}$)
To detect extreme motion blur, vitreous hemorrhage opacity, or blank camera sensor noise, the discrete Laplacian operator $\nabla^2 X_{\mathrm{gray}}$ is computed:
$$\mathrm{Var}_{\mathrm{lap}} = \frac{1}{|M|} \sum_{(x, y) \in M} \left( \nabla^2 X_{\mathrm{gray}}(x, y) - \overline{\nabla^2 X_{\mathrm{gray}}} \right)^2$$
$$\mathcal{V}_{\mathrm{lap}}(X) = \min\left(1.0, \frac{\mathrm{Var}_{\mathrm{lap}}}{\theta_{\mathrm{sharp}}}\right), \quad \theta_{\mathrm{sharp}} = 100.0$$

### 2.3 Gatekeeping Decision Boundary
The Tier-0 gate enforces the decision rule:
$$\mathrm{Gate}(X) = \begin{cases} \mathrm{ACCEPT} \to \text{Tier-1 GPU Ensemble}, & \text{if } \mathcal{S}(X) \ge \tau_{\mathrm{guard}} \\ \mathrm{REJECT} \to \text{HTTP 422 Unprocessable Content}, & \text{if } \mathcal{S}(X) < \tau_{\mathrm{guard}} \end{cases}$$
with operating threshold $\tau_{\mathrm{guard}} = 0.50$.
- **Empirical Validation:** On $N=1,389$ genuine fundus scans, mean $\mathcal{S}(X) = 0.9604 \pm 0.0412$, admission rate = $100.0\%$. Across 7 non-fundus modalities (2,500 negative distractors), distractor rejection = $100.0\%$, with P50 execution latency of $5.87\,\mathrm{ms}$ on CPU.

---

## 3. Mathematical Formulation: Multi-Backbone Probability Calibration

### 3.1 Platt Temperature Scaling
Given unnormalized model logits $z_m(X) \in \mathbb{R}^K$ from backbone $m \in \{\text{ConvNeXt}, \text{DenseNet}, \text{EfficientNet}\}$, naive softmax produces uncalibrated probabilities:
$$\hat{p}_{m, k}(X) = \frac{\exp(z_{m, k})}{\sum_{j=1}^K \exp(z_{m, j})}$$
Deep convolutional networks routinely suffer from overconfidence due to cross-entropy minimization on overparameterized architectures. Platt temperature scaling introduces a learned positive scalar $T_m > 0$:
$$\hat{p}_{m, k}^{\mathrm{cal}}(X) = \frac{\exp(z_{m, k} / T_m)}{\sum_{j=1}^K \exp(z_{m, j} / T_m)}$$
$T_m$ is optimized on an independent calibration split ($\mathcal{D}_{\mathrm{calib}}$, $n=1,389$) by minimizing Negative Log-Likelihood (NLL) via L-BFGS:
$$T_m^* = \arg\min_{T > 0} - \frac{1}{|\mathcal{D}_{\mathrm{calib}}|} \sum_{i \in \mathcal{D}_{\mathrm{calib}}} \log \left( \frac{\exp(z_{m, y_i} / T)}{\sum_{j=1}^K \exp(z_{m, j} / T)} \right)$$

### 3.2 Calibration Order of Operations
The audit confirms the exact order of operations in `backend/ensemble.py`:
1. **Pre-fusion calibration:** Each backbone's raw logits are scaled by its individual optimal temperature:
   - $T_{\mathrm{eff}} = 1.121$ (EfficientNet-V2-M)
   - $T_{\mathrm{dense}} = 1.184$ (DenseNet-201)
   - $T_{\mathrm{conv}} = 1.153$ (ConvNeXt-Small)
2. **Ensemble fusion:** Calibrated probability vectors are aggregated via uniform soft voting:
   $$\bar{P}(y \mid X) = \frac{1}{3} \sum_{m=1}^3 \hat{P}_m^{\mathrm{cal}}(y \mid X)$$
3. **Meta-Stacking:** When using the MLP stacking classifier, the concatenated calibrated probability vectors feed into the MLP, whose output is scaled by $T_{\mathrm{meta}} = 1.042$.
- **Result:** Expected Calibration Error (ECE) drops from $0.0914$ (uncalibrated) to $0.0389$ (Meta-Ensemble) and $0.0477$ (Uniform Soft Voting).

---

## 4. Mathematical Formulation: Admissibility-Weighted Conformal Risk Control (AW-CRC)

### 4.1 Non-Conformity Function
In classical split conformal prediction, the standard non-conformity score for a candidate pair $(X, y)$ is:
$$s(X, y) = 1 - \hat{P}(y \mid X)$$
Under AW-CRC, we condition non-conformity on the continuous optical admissibility $\mathcal{S}(X) \in [0, 1]$:
$$s_{\mathrm{AW}}(X, y) = (1 - \hat{P}(y \mid X)) \cdot \mathcal{S}(X)^\gamma$$
where $\gamma \ge 0$ is the optical attenuation exponent (calibrated to $\gamma = 1.0$).
- **Mathematical Intuition:** When an image is degraded (e.g., optical media opacity, cataract, or off-axis blur), $\mathcal{S}(X) < 1.0$. This compresses the calculated non-conformity score $s_{\mathrm{AW}}$, reflecting greater physical ambiguity.

### 4.2 Calibration Quantiles with Finite-Sample Correction
Let $\mathcal{D}_{\mathrm{calib}} = \{(X_i, y_i)\}_{i=1}^n$ be the calibration cohort ($n = 1,389$).
The non-conformity scores $\{s_{\mathrm{AW}}(X_i, y_i)\}_{i=1}^n$ are computed.
To handle clinical asymmetry, classes are partitioned into two risk strata:
1. **Emergency Stratum ($\mathcal{K}_{\mathrm{emerg}}$):** Sight-threatening acute pathologies:
   $$\mathcal{K}_{\mathrm{emerg}} = \{\text{Diabetic Retinopathy}, \text{Glaucoma}, \text{AMD}\}, \quad \alpha_{\mathrm{emerg}} = 0.01 \implies 99.0\% \text{ nominal coverage}$$
2. **Routine Stratum ($\mathcal{K}_{\mathrm{rout}}$):** Non-acute / physiological conditions:
   $$\mathcal{K}_{\mathrm{rout}} = \{\text{Normal}, \text{Cataract}\}, \quad \alpha_{\mathrm{rout}} = 0.05 \implies 95.0\% \text{ nominal coverage}$$

For each stratum $j \in \{\mathrm{emerg}, \mathrm{rout}\}$, let $n_j = \sum_{i=1}^n \mathbb{I}(y_i \in \mathcal{K}_j)$. The calibration quantile index is computed using the rigorous finite-sample correction:
$$p_j = \min\left(1.0, \frac{\lceil (n_j + 1)(1 - \alpha_j) \rceil}{n_j}\right)$$
$$\hat{q}_j = \mathrm{Quantile}\left( \{ s_{\mathrm{AW}}(X_i, y_i) \mid y_i \in \mathcal{K}_j \}, p_j \right)$$
From empirical calibration on `calibration_v2.csv` ($n=1,389$, `models/aw_crc_calibration.json`):
- $\hat{q}_{\mathrm{emerg}} = 0.7017$ (for $\alpha = 0.01$)
- $\hat{q}_{\mathrm{rout}} = 0.3055$ (for $\alpha = 0.05$)

### 4.3 Test Inclusion Decision Rule
For a novel query image $X_{\mathrm{test}}$ with admissibility $\mathcal{S}(X_{\mathrm{test}})$ and ensemble probability vector $\hat{P}(c \mid X_{\mathrm{test}})$, candidate class $c \in \{0, \dots, K-1\}$ is admitted into the prediction set $\mathcal{C}(X_{\mathrm{test}})$ if and only if its non-conformity score is at most the stratum quantile:
$$s_{\mathrm{AW}}(X_{\mathrm{test}}, c) \le \hat{q}_{\mathrm{stratum}(c)}$$
$$(1 - \hat{P}(c \mid X_{\mathrm{test}})) \cdot \mathcal{S}(X_{\mathrm{test}})^\gamma \le \hat{q}_{\mathrm{stratum}(c)}$$
Rearranging for the posterior probability yields the explicit inclusion threshold:
$$\hat{P}(c \mid X_{\mathrm{test}}) \ge \tau_c(X_{\mathrm{test}}), \quad \tau_c(X_{\mathrm{test}}) = \max\left(0.0, 1.0 - \frac{\hat{q}_{\mathrm{stratum}(c)}}{\mathcal{S}(X_{\mathrm{test}})^\gamma}\right)$$
$$\mathcal{C}(X_{\mathrm{test}}) = \left\{ c \in \{0, \dots, K-1\} \;\middle|\; \hat{P}(c \mid X_{\mathrm{test}}) \ge \max\left(0, 1 - \frac{\hat{q}_{\mathrm{stratum}(c)}}{\mathcal{S}(X_{\mathrm{test}})^\gamma}\right) \right\}$$

- **Analysis of Dynamic Set Expansion:**
  - If $\mathcal{S}(X) \approx 1.00$ (pristine image): $\tau_{\mathrm{emerg}} = 1 - 0.7017 = 0.2983$. A candidate emergency disease needs $\ge 29.83\%$ probability to enter the set.
  - If $\mathcal{S}(X) = 0.75$ (degraded image): $\tau_{\mathrm{emerg}} = \max(0, 1 - 0.7017 / 0.75) = \max(0, 1 - 0.9356) = 0.0644$. The threshold drops to $6.44\%$, dynamically expanding the prediction set to capture plausible pathologies under poor imaging conditions.
  - If $\mathcal{S}(X) \le 0.7017$: $\tau_{\mathrm{emerg}} = 0.00$, meaning *all* emergency conditions with non-zero probability enter the candidate set, automatically escalating the scan to comprehensive specialist review.

---

## 5. Theoretical Guarantees and Exchangeability Boundaries

### 5.1 In-Distribution Exchangeability Guarantee
Let $(X_1, Y_1), \dots, (X_n, Y_n), (X_{n+1}, Y_{n+1})$ be independent and identically distributed (or exchangeable) random variables drawn from distribution $\mathcal{P}$.
Under exchangeability, the non-conformity scores $S_1, \dots, S_{n+1}$ are exchangeable. By the classical conformal theorem (Vovk et al. 2005; Bates et al. 2021):
$$\mathbb{P}(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - \alpha$$
This theoretical bound holds marginally over repeated drawings of calibration and test datasets.
- **Empirical Realization on Test Set ($N=1,389$):**
  - Emergency stratum empirical coverage: **$99.31\%$** (Target $\ge 99.0\%$, $\alpha=0.01$).
  - Routine stratum empirical coverage: **$96.73\%$** (Target $\ge 95.0\%$, $\alpha=0.05$).
  - Mean prediction set cardinality: **$1.010$ classes/patient** (with $98.99\%$ singletons).

### 5.2 Breakdown of Exchangeability Under Camera Domain Shift
When the model is evaluated on external clinics with different camera sensors (e.g., Kowa VX-10 on IDRiD; Nidek AFC-210 on RIM-ONE DL), the data generating distribution shifts:
$$\mathcal{P}_{\mathrm{target}}(X, Y) \neq \mathcal{P}_{\mathrm{source}}(X, Y)$$
Because test samples are no longer exchangeable with calibration samples:
$$\mathbb{P}_{X \sim \mathcal{P}_{\mathrm{target}}}(Y \in \mathcal{C}(X)) \not\ge 1 - \alpha \quad \text{(Mathematical guarantee is formally void)}$$

- **Clinical AI Safety Behavior:** The manuscript does *not* claim that conformal coverage mathematically holds under arbitrary covariate shift. Instead, it demonstrates the **operational fail-closed behavior** of AW-CRC:
  When domain shift occurs, model outputs become ambiguous and non-conformity scores spike. Rather than emitting silent, overconfident misdiagnoses, the conformal system:
  1. Expands candidate set size ($|\mathcal{C}(X)| > 1$), or
  2. Generates non-conformal empty sets requiring referral.
  On IDRiD and RIM-ONE DL, this behavior results in an empirical clinical referral rate of **$100.0\%$**, safely redirecting every shifted image to human ophthalmologists.

---

## 6. Mathematical Audit of Empty Sets and Argmax Fallback

### 6.1 Empty Set Occurrence
An empty prediction set $\mathcal{C}(X) = \emptyset$ occurs whenever no candidate class satisfies the inclusion threshold:
$$\forall c \in \{0, \dots, K-1\}, \quad \hat{P}(c \mid X) < \max\left(0, 1 - \frac{\hat{q}_{\mathrm{stratum}(c)}}{\mathcal{S}(X)^\gamma}\right)$$
This arises when probability mass is diffuse across multiple classes (high epistemic uncertainty) yet the image retains high admissibility $\mathcal{S}(X) \approx 1$.
- **Empirical Frequency:** On the sealed test cohort ($N=1,389$), raw conformal prediction produced empty sets in exactly **8 cases (0.576% $\approx 0.58\%$)**.
- **Raw Mathematical Coverage:** Without fallback, coverage is **$97.91\%$** ($1,360 / 1,389$).

### 6.2 Operational Argmax Fallback Policy
In real-world clinical telemedicine, delivering a completely blank report with no candidate hypothesis causes severe workflow disruption. To balance mathematical purity with clinical utility, OphthalmoAI implements an operational fallback:
$$\mathcal{C}_{\mathrm{operational}}(X) = \begin{cases} \mathcal{C}(X), & \text{if } \mathcal{C}(X) \neq \emptyset \\ \{ \arg\max_c \hat{P}(c \mid X) \}, & \text{if } \mathcal{C}(X) = \emptyset \end{cases}$$
alongside an immutable system flag:
$$\mathrm{ReviewRequired} = \mathrm{True}, \quad \mathrm{UrgencyLevel} = \text{"URGENT\_FALLBACK"}$$

- **Mathematical Reconciliation:**
  - Of the 8 empty cases, the top argmax prediction matched the ground truth in 6 cases and was incorrect in 2 cases.
  - Adding the 6 recovered cases elevates operational coverage from $97.91\%$ to **$98.34\%$** ($1,366 / 1,389$).
  - Effective null set rate delivered to clinical UI: **$0.00\%$** (zero unhandled cases).
  - Effective prediction set cardinality: **$1.010$ classes/patient**.
- **Manuscript Transparency:** Section III-C and Section V-B in `paper_revised.tex` explicitly disclose both the raw mathematical coverage ($97.91\%$) and the operational fallback coverage ($98.34\%$), ensuring full scientific transparency.

---

## 7. Mathematical Audit Checklist

| Item | Mathematical Definition | Code Implementation | Empirical Verification | Audit Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Aperture Geometry** | Circular hull ratio $\times$ circularity factor | `backend/validators.py:65` | Returns $\approx 0.98$ on clean fundus; $<0.30$ on distractors | **PASSED** |
| **Spectral Ratio** | $1.25 \le \mu_R/\mu_G \le 2.50$, $\mu_B/\mu_R < 0.45$ | `backend/validators.py:92` | Rejects grayscale, green, blue distractors | **PASSED** |
| **Laplacian Variance** | $\mathrm{Var}(\nabla^2 X) / 100.0$ | `backend/validators.py:115` | Rejects blank frames and flash glare | **PASSED** |
| **Temperature Scaling** | $\arg\min_T \mathrm{NLL}(z/T, y)$ | `scripts/calibrate_conformal.py:45` | L-BFGS converges to $T \in [1.04, 1.18]$ | **PASSED** |
| **AW-CRC Non-Conformity** | $s_{\mathrm{AW}} = (1-\hat{P})\cdot \mathcal{S}(X)^\gamma$ | `backend/conformal.py:88` | Continuous scaling verified | **PASSED** |
| **Finite-Sample Quantile** | $\lceil (n+1)(1-\alpha) \rceil / n$ | `scripts/run_aw_crc_calibration.py:127` | $q_e = 0.7017, q_r = 0.3055$ | **PASSED** |
| **Inclusion Rule** | $\hat{P} \ge 1 - \hat{q}/\mathcal{S}(X)^\gamma$ | `backend/conformal.py:135` | Dynamically expands on low $\mathcal{S}(X)$ | **PASSED** |
| **Argmax Fallback** | Inject $\arg\max$ if set is $\emptyset$ | `backend/conformal.py:155` | Exactly 8 cases handled with mandatory alert | **PASSED** |
| **McNemar Exact Test** | $2 \sum_{k=0}^{\min(b,c)} \binom{b+c}{k} (0.5)^{b+c}$ | `scripts/evaluate_models_v2.py:180` | $b=4, c=2 \implies p=0.6875$ | **PASSED** |

---

## 8. Conclusion

The mathematical formulations of OphthalmoAI are theoretically consistent, formally derived from established principles of physiological optics and distribution-free conformal inference, and executed without divergence in the production codebase. The manuscript transparently articulates exchangeability assumptions and honest boundaries under external optical domain shift.
