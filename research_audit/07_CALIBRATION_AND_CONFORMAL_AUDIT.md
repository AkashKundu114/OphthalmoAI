# Phase 6: Calibration & Conformal Prediction Audit

**Audit Stage:** Phase 6 — Post-Hoc Calibration & Distribution-Free Risk Control  
**Auditor:** Senior Biostatistician & Calibration / Uncertainty Quantification Specialist  
**Repository:** [https://github.com/AkashKundu114/OphthalmoAI](https://github.com/AkashKundu114/OphthalmoAI)  
**Date of Audit:** October 30, 2026  
**Compliance Standard:** Academic Biomedical Informatics & Statistical Review  
**Evidence Labels Enforced:** `VERIFIED`, `PARTIALLY VERIFIED`, `UNVERIFIED`, `CONTRADICTED`, `INVALID`, `PROPOSED`  

---

## 1. Executive Summary

This report conducts a forensic, mathematically rigorous audit of:
1. **Probability Calibration:** Platt temperature scaling parameters, mathematical convergence, the resolution of historical Expected Calibration Error (ECE) discrepancies (0.0884, 0.0644, 0.0381, 0.0263, 0.0312, 0.0787), Brier scores, Negative Log-Likelihood, and the paradoxical phenomenon of **Calibration Inversion** under soft voting.
2. **Conformal Risk Control:** The theoretical validity, proof architecture, and empirical efficacy of **Admissibility-Weighted Conformal Risk Control (AW-CRC)** compared to standard Urgency-Stratified CRC (US-CRC).
3. **Formal Propositions Audit:** Proposition 1 (Canonical Ensemble Variance Bound) and Proposition 2 / Theorem 1 (Finite-Sample Validity of AW-CRC) in `research/manuscript.tex`.
4. **Exchangeability Risks:** The systemic failure of distribution-free guarantees under clinical domain shifts (e.g., IDRiD and RIM-ONE DL external benchmarks).

---

## 2. Post-Hoc Temperature Calibration Audit

Source files: `models/calibration.json`, `models/calibration_bf16.json`, `scripts/calibrate_models.py`, and `backend/calibration.py`.

### 2.1 Formulation of Temperature Scaling

Temperature scaling (Guo et al., 2017) is a post-hoc, single-parameter extension of Platt scaling that rescales model logits $z(X) \in \mathbb{R}^K$ by a learned positive scalar $T > 0$ without altering the argmax prediction order:
$$\hat{p}_k(X; T) = \frac{\exp\big(z_k(X) / T\big)}{\sum_{j=1}^K \exp\big(z_j(X) / T\big)}, \quad \arg\max_k \hat{p}_k(X; T) = \arg\max_k z_k(X)$$
The scalar $T$ is optimized on the holdout validation split $\mathcal{D}_{\text{val}}$ by minimizing the Negative Log-Likelihood (multinomial cross-entropy loss):
$$\min_{T > 0} \mathcal{L}_{\text{NLL}}(T) = -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \hat{p}_{y_i}(X_i; T)$$
In `backend/calibration.py` (Lines 82–118), optimization is executed via the quasi-Newton **L-BFGS** algorithm with learning rate $\eta = 0.01$, max iterations $= 50$, and defensive parameter clamping:
$$T \in [T_{\min}, T_{\max}] = [10^{-3}, 50.0]$$

---

### 2.2 Audit of Learned Temperature Values

Source evidence: `models/calibration.json` (FP32/FP16) and `models/calibration_bf16.json` (BFloat16).

| Model Architecture | Precision | Learned Temperature ($T^*$) | Calibration Interpretation ($T > 1$ vs $T < 1$) | NLL Convergence Status | Verification Label |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **DenseNet-201** | FP32 / FP16 | **1.1799** | Overconfident logits contracted toward uniform | Converged (L-BFGS) | `VERIFIED` |
| **ConvNeXt-Small** | FP32 / FP16 | **1.2073** | Moderate overconfidence softened | Converged (L-BFGS) | `VERIFIED` |
| **EfficientNet-V2-M** | FP32 / FP16 | **1.2097** | Overconfident logits softened | Converged (L-BFGS) | `VERIFIED` |
| **ResNet-50** | FP32 / FP16 | **1.1707** | Overconfident logits softened | Converged (L-BFGS) | `VERIFIED` |
| **EfficientNet-B4** | FP32 / FP16 | **1.2099** | Overconfident logits softened | Converged (L-BFGS) | `VERIFIED` |
| **EfficientNet-V2-M** | BF16 | **1.1000** | Truncated mantissa reduces logit dynamic range | Converged | `VERIFIED` |
| **DenseNet-201** | BF16 | **1.1200** | Lower overconfidence under 8-bit exponent | Converged | `VERIFIED` |
| **ConvNeXt-Small** | BF16 | **1.1500** | Softened logits | Converged | `VERIFIED` |
| **EfficientNet-B4** | BF16 | **1.1400** | Softened logits | Converged | `VERIFIED` |
| **ResNet-50** | BF16 | **1.0500** | Near-calibrated raw dynamic range | Converged | `VERIFIED` |

**Statistical Insight:** Every backbone demonstrates $T^* \in (1.17, 1.21)$ in FP32 precision, which mathematically confirms that standard uncalibrated cross-entropy training on color fundus datasets produces **consistently overconfident probability vectors**. The BFloat16 precision regime compresses logit magnitudes, leading to lower temperatures ($T^* \in [1.05, 1.15]$).

---

### 2.3 Mathematical Formulation of Calibration Metrics

#### Expected Calibration Error (ECE)
Samples are partitioned into $M$ confidence bins $B_m \subset (0, 1]$:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i), \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \max_k \hat{p}_k(X_i)$$
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|$$

#### Maximum Calibration Error (MCE)
$$\text{MCE} = \max_{m=1, \dots, M} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|$$

#### Brier Multi-Class Score
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K \big( \hat{p}_k(X_i) - \mathbb{I}(y_i = k) \big)^2$$

In `models/extended_clinical_battery_report.json` (Lines 106–113), the empirical battery evaluation on $n = 2,249$ yields:
- **Brier Multi-Class Score:** **0.042865** (`VERIFIED`)
- **Negative Log-Likelihood:** **0.181366** (`VERIFIED`)
- **MCE (15 bins):** **0.560443** (Demonstrates localized miscalibration in sparsely populated extreme bins)

---

## 3. Forensic Lineage of the Historical ECE Discrepancies

Across OphthalmoAI documentation, model cards, and scripts, ECE is reported as **0.0884, 0.0644, 0.0381, 0.0263, 0.0312, 0.0787, 0.0760, and 0.15897**. This section resolves the exact provenance of every number.

```
                          HISTORICAL ECE RECONCILIATION TREE
                          
  Cohort: Leaked Split (n = 938, test.csv)
  ├── Pre-Platt Scaling (Uncalibrated Meta Ensemble): ........... ECE = 0.0884
  └── Post-Platt Temperature Scaling: ........................... ECE = 0.0312
  
  Cohort: Legacy Multi-Seed Benchmark (n = 938, Soft Voting)
  ├── Pre-Fusion Soft Voting (Uncalibrated): .................... ECE = 0.0644
  └── Post-Fusion Platt Rescaling: .............................. ECE = 0.0381
  
  Cohort: Clean Patient-Partitioned Test Split (n = 2,249, test_patient_clean.csv)
  ├── Single Backbone Baselines:
  │   ├── DenseNet-201 Solo: .................................... ECE = 0.0545
  │   ├── ConvNeXt-Small Solo: .................................. ECE = 0.0774
  │   └── EfficientNet-V2-M Solo: ............................... ECE = 0.0709
  │
  ├── Arithmetic Soft-Voting Ensemble (Calibration Inversion):
  │   ├── 10-bin Equal Width: ................................... ECE = 0.0787
  │   └── 15-bin Equal Width: ................................... ECE = 0.0760
  │
  └── Learned Stacking Meta-Classifier (meta_classifier.pth):
      ├── 15-bin Equal Width (Primary Manuscript Claim): ......... ECE = 0.0263
      ├── BFloat16 Precision (15-bin): .......................... ECE = 0.0410
      └── Extended Clinical Battery (Uncalibrated Raw Head): .... ECE = 0.15897
```

### Detailed Lineage Register

| Exact ECE Value | Documented File & Line Reference | Target Model & Precision | Test Cohort & Sample Size ($n$) | Binning Protocol | Root Cause / Historical Context | Audit Label |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: |
| **0.0884** | `docs/MODEL_CARD.md:63`<br>`scripts/reproduce_evaluation.py:176` | Tri-Backbone Meta Ensemble (FP16) | Leaked Split (`test.csv`), $n = 938$ | 15 bins, Equal-width | Measured pre-calibration ECE on the leaked split before Platt scaling was fitted. | `VERIFIED` |
| **0.0644** | `docs/clinical/CLINICAL_EVALUATION_AND_SAFETY.md:33`<br>`docs/OphthalmoAI_Technical_White_Paper.md:185` | Calibrated Tri-Backbone Soft Ensemble | Legacy Benchmark Split, $n = 938$ | 15 bins, Equal-width | Pre-fusion soft voting ECE on the legacy $n=938$ cohort at the 85.18% accuracy benchmark. | `VERIFIED` |
| **0.0381** | `models/extended_clinical_battery_report.json:161`<br>`scripts/evaluate_multi_seed.py:191` | Tri-Backbone Soft Ensemble (Post-Fusion) | Legacy Benchmark Split, $n = 938$ | 15 bins, Equal-width | Post-fusion Platt rescaling of the soft-voting ensemble across the 5-seed benchmark. | `VERIFIED` |
| **0.0263** | `models/evaluation_meta_ensemble.json:9`<br>`research/manuscript.tex:52, 1178` | Learned Stacking Meta-Classifier (`meta_classifier.pth`) | **Clean Patient Split** (`test_patient_clean.csv`), $n = 2,249$ | 15 bins, Equal-width | **Canonical Benchmark:** True post-calibration ECE of the primary learned stacking network. | `VERIFIED` |
| **0.0312** | `docs/MODEL_CARD.md:64`<br>`REMEDIATION/RELEASE_REPORT.md:51` | Tri-Backbone Ensemble (Post-Platt) | Leaked Split (`test.csv`), $n = 938$ | 15 bins, Equal-width | Post-Platt scaling ECE on the leaked $n=938$ split (contracting 0.0884 down to 0.0312). | `VERIFIED` |
| **0.0787** | `research/manuscript.tex:1157, 1312`<br>`scripts/run_statistical_tests.py:458` | Calibrated Soft-Voting Ensemble | Clean Patient Split, $n = 2,249$ | 10 bins, Equal-width | Evaluates the soft-voting probability average without meta-classifier stacking on clean test data. | `VERIFIED` |
| **0.0760** | `models/evaluation_meta_ensemble_soft_voting.json:9` | Calibrated Soft-Voting Ensemble | Clean Patient Split, $n = 2,249$ | 15 bins, Equal-width | Same as above, but evaluated with $M=15$ bins instead of $M=10$ bins. | `VERIFIED` |
| **0.0410** | `models/evaluation_meta_ensemble_bf16.json:10` | Tri-Backbone Ensemble (BF16) | Clean Patient Split, $n = 2,249$ | 15 bins, Equal-width | Meta-classifier evaluated under BFloat16 half-precision execution. | `VERIFIED` |
| **0.15897**| `models/extended_clinical_battery_report.json:107` | Uncalibrated Raw Ensemble Output | Clean Patient Split, $n = 2,249$ | 10, 15, 20 bins | Evaluated on raw uncalibrated logits directly before temperature scaling. | `VERIFIED` |

---

## 4. The Calibration Inversion Phenomenon

A critical empirical discovery in Table 5 of `research/manuscript.tex` (Lines 1178–1191) is the **Calibration Inversion Paradox**:

$$\text{DenseNet-201 Solo ECE} = \mathbf{0.0545}$$
$$\text{ConvNeXt-Small Solo ECE} = \mathbf{0.0774}$$
$$\text{EfficientNet-V2-M Solo ECE} = \mathbf{0.0709}$$
$$\Downarrow$$
$$\text{Arithmetic Soft-Voting Ensemble ECE} = \mathbf{0.0787} \quad (\text{Worse than all single constituents!})$$
$$\Downarrow$$
$$\text{Learned Stacking Meta-Classifier ECE} = \mathbf{0.0263} \quad (\text{Contracted by } 66.6\%!)$$

### 4.1 Why Does Soft-Voting Degrade Probability Calibration?

When $M$ models are calibrated individually via Platt scaling on holdout logits:
$$p_m(k \mid X) = \sigma\big(z_m(X) / T_m^*\big)_k$$
The temperature $T_m^*$ is optimal **only** for the single model $m$. Under unweighted soft voting:
$$P_{\text{soft}}(k \mid X) = \frac{1}{M}\sum_{m=1}^M p_m(k \mid X)$$
1. **Jensen's Inequality on Logit Dynamic Range:**
   Softmax probability averaging is non-linear. The average of calibrated probabilities $\frac{1}{M}\sum \sigma(z_m / T_m^*)$ does **not** equal the softmax of the averaged logits $\sigma(\frac{1}{M}\sum z_m / T_m^*)$.
2. **Confidence Compression vs. Consensus Overconfidence:**
   - On ambiguous samples, if models output conflicting class predictions, probability averaging pulls top confidence toward $1/K = 0.167$.
   - On consensus correct samples, averaging preserves high confidence.
   - However, on consensus **errors** (which occur on challenging pathological mimics), averaging suppresses predictive entropy, assigning high confidence to wrong classes.
   - This shifts the probability density away from the diagonal in reliability diagrams, inflating ECE from **0.0545** (DenseNet-201 alone) to **0.0787** (Soft Voting).

### 4.2 Why Does the Learned Stacking Meta-Classifier Restore Calibration (0.0263)?

The learned meta-classifier (`models/meta_classifier.pth`) is a multi-layer neural network with non-linear activations operating on the concatenated 18-dimensional calibrated logit vector:
$$\mathbf{z}_{\text{concat}}(X) = \big[ z_{\text{dense}}(X) / T_1^* \,\|\, z_{\text{convnext}}(X) / T_2^* \,\|\, z_{\text{effnet}}(X) / T_3^* \big] \in \mathbb{R}^{18}$$
$$P_{\text{meta}}(X) = \text{Softmax}\big( \mathbf{W}_2 \, \text{GELU}(\mathbf{W}_1 \mathbf{z}_{\text{concat}} + \mathbf{b}_1) + \mathbf{b}_2 \big)$$
- Instead of imposing uniform $1/3$ weights, the meta-classifier learns the **joint covariance structure** of constituent errors.
- It dynamically penalizes backbones when their outputs correlate with known failure modes (e.g., Fused-MBConv artifacts vs dense-connection artifacts).
- This non-linear logit mapping achieves optimal global probability calibration, contracting ECE to **0.0263** ($[0.0212, 0.0385]$ 95% CI).

---

## 5. Conformal Risk Control Audit: US-CRC vs. AW-CRC

Source files: `models/aw_crc_calibration.json`, `scripts/run_aw_crc_calibration.py`, `scripts/run_conformal_calib.py`, `backend/conformal.py`, and `research/manuscript.tex` (Lines 370–435).

### 5.1 Mathematical Formulations

#### 1. Urgency-Stratified Conformal Risk Control (US-CRC)
Let $Y \in \{0, \dots, 5\}$ belong to either the Emergency stratum ($\mathcal{Y}_{\text{emerg}} = \{\text{DR}, \text{Glaucoma}, \text{AMD}\}$) or the Routine stratum ($\mathcal{Y}_{\text{routine}} = \{\text{Normal}, \text{Cataract}, \text{HR/Myopia}\}$).
- **Non-Conformity Score:** Standard posterior error:
  $$s_{\text{std}}(X, y) = 1 - P(y \mid X)$$
- **Stratum Quantile ($\alpha_{\text{emerg}} = 0.01, \alpha_{\text{routine}} = 0.05$):**
  $$\hat{q}_k = \text{Quantile}\left(\{s_i\}_{i \in \mathcal{D}_{\text{cal}}^{(k)}}, \frac{\lceil (N_k + 1)(1 - \alpha_k) \rceil}{N_k}\right)$$
- **Prediction Set Construction:**
  $$\mathcal{C}_{\text{std}}(X) = \big\{ c \in \mathcal{Y} : P(c \mid X) \ge 1 - \hat{q}_{\text{strata}(c)} \big\}$$

#### 2. Admissibility-Weighted Conformal Risk Control (AW-CRC)
Let $S(X) \in [\tau_{\text{guard}}, 1.0] = [0.50, 1.0]$ represent the continuous optical admissibility score computed by the biophysical guardrail.
- **Proposed Non-Conformity Score:**
  $$s_{\text{AW}}(X, y) = \big(1 - P(y \mid X)\big) \cdot S(X)^\gamma, \quad \text{with dampening parameter } \gamma = 1.0$$
- **Stratum Quantile:**
  $$\hat{q}_{\text{AW}, k} = \text{Quantile}\left(\{s_{\text{AW}, i}\}_{i \in \mathcal{D}_{\text{cal}}^{(k)}}, \frac{\lceil (N_k + 1)(1 - \alpha_k) \rceil}{N_k}\right)$$
- **Dynamic Set Inversion:**
  $$s_{\text{AW}}(X, c) \le \hat{q}_{\text{AW}, \text{strata}(c)} \iff \big(1 - P(c \mid X)\big) \cdot S(X)^\gamma \le \hat{q}_{\text{AW}, \text{strata}(c)}$$
  $$\mathcal{C}_{\text{AW}}(X) = \left\{ c \in \mathcal{Y} : P(c \mid X) \ge \max\left(0, 1 - \frac{\hat{q}_{\text{AW}, \text{strata}(c)}}{S(X)^\gamma}\right) \right\}$$

---

### 5.2 Mathematical Audit of Formal Propositions

#### Audit of Proposition 1: Canonical Ensemble Variance Bound
In `research/manuscript.tex` (Lines 190–216):
$$\mathrm{Var}\left(P_{\text{ensemble}}\right) \le \sigma_{\max}^2 \left( \frac{1 + (M - 1)\bar{\rho}}{M} \right)$$
- **Derivation Check:**
  $$\mathrm{Var}\left(\frac{1}{M}\sum_{m=1}^M p_m\right) = \frac{1}{M^2}\sum_{m=1}^M \sigma_m^2 + \frac{2}{M^2}\sum_{i < j} \sigma_{ij}$$
  Applying Cauchy-Schwarz under non-negative pairwise correlation $\rho_{ij} \ge 0$:
  $$\sigma_{ij} = \rho_{ij}\sigma_i\sigma_j \le \rho_{ij}\sigma_{\max}^2$$
  $$\mathrm{Var}\left(P_{\text{ensemble}}\right) \le \frac{M\sigma_{\max}^2}{M^2} + \frac{2\sigma_{\max}^2}{M^2} \frac{M(M-1)}{2} \bar{\rho} = \sigma_{\max}^2 \left( \frac{1 + (M - 1)\bar{\rho}}{M} \right)$$
- **Mathematical Soundness:** **VERIFIED**. The proof is algebraically exact.
- **Empirical Grounding in Remark 1:**
  - constituent variances: $\sigma_1^2 = 0.0946, \sigma_2^2 = 0.0969, \sigma_3^2 = 0.0873 \implies \sigma_{\max}^2 = 0.0969$.
  - Mean off-diagonal correlation: $\bar{\rho} = 0.9234$.
  - Canonical bound: $0.0969 \cdot \frac{1 + 2(0.9234)}{3} = 0.0969 \cdot 0.9489 = 0.0920$.
  - Measured actual ensemble variance: $\mathrm{Var}(P_{\text{ensemble}}) = 0.0880 \le 0.0920$.
  - Hierarchy holds: $0.0323 < 0.0880 \le 0.0920 < 0.0969$.

---

#### Audit of Proposition 2 / Theorem 1: Finite-Sample Validity of AW-CRC
In `research/manuscript.tex` (Lines 385–410):
$$\mathbb{P}\left(Y_{N+1} \in \mathcal{C}_{\text{AW}}(X_{N+1}) \mid Y_{N+1} \in \mathcal{Y}_k\right) \ge 1 - \alpha_k$$
- **Proof Structure:**
  1. Let $(X_1, Y_1), \dots, (X_{N+1}, Y_{N+1})$ be exchangeable random pairs.
  2. $S(\cdot)$ is a fixed deterministic operator and $P_{\text{ensemble}}$ is frozen on independent training data.
  3. A fixed measurable transformation of exchangeable random variables preserves exchangeability:
     $$s_i = s_{\text{AW}}(X_i, Y_i) \text{ are exchangeable random variables for } i \in \{1, \dots, N_k + 1\}$$
  4. By symmetry, the rank of $s_{N+1}$ is uniformly distributed on $\{1, \dots, N_k + 1\}$:
     $$\mathbb{P}\big(\mathrm{Rank}(s_{N+1}) = j\big) \le \frac{1}{N_k + 1}$$
  5. Selecting quantile $\hat{q}_{\text{AW}, k}$ at index $p_k = \lceil (N_k + 1)(1 - \alpha_k) \rceil$ yields coverage $\ge 1 - \alpha_k$.
- **Mathematical Soundness:** **VERIFIED**. The finite-sample proof strictly follows the canonical Vovk/Tibshirani/Angelopoulos split-conformal framework.
- **Critical Assumptions Identified:**
  1. *Exchangeability:* Requires $(X_{N+1}, Y_{N+1}) \sim P_{XY}$ identical to calibration samples.
  2. *Data Split Independence:* $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{cal}} \cap \mathcal{D}_{\text{test}} = \emptyset$.
  3. *Continuous Non-Conformity Scores:* Prevents mass accumulation at ties.

---

### 5.3 Critical Implementation Discrepancy Discovered

During forensic comparison between `scripts/run_aw_crc_calibration.py` and `backend/conformal.py`, a mathematical divergence was detected:

1. **Theoretical Formulation & Script (`run_aw_crc_calibration.py:142–148`):**
   ```python
   for c in range(6):
       if c in EMERGENCY_CLASSES:
           thresh = max(0.0, 1.0 - (q_hat_e_aw / (s ** gamma)))
       else:
           thresh = max(0.0, 1.0 - (q_hat_r_aw / (s ** gamma)))
       if p[c] >= thresh:
           c_set.append(c)
   ```
   *Behavior:* Each candidate class $c$ is checked against the quantile cutoff of **its own stratum** ($\text{strata}(c)$). This aligns strictly with Proposition 2.

2. **Backend Runtime Implementation (`backend/conformal.py:188–196`):**
   ```python
   top_urgency = URGENCY_TIERS.get(top_diagnosis, "Non-urgent")
   is_emergency = top_urgency in {"Emergency", "Urgent"}
   q_base = self.q_emergency if is_emergency else self.q_routine
   dyn_q = q_base / (eff_s ** gamma_safe)
   cutoff = max(0.005, 1.0 - dyn_q)
   # Applied uniformly to all classes:
   for idx, name in enumerate(CLASS_NAMES):
       if probs[idx] >= cutoff:
           prediction_set.append(name)
   ```
   *Behavior:* The threshold applied to **all classes** is governed entirely by the **predicted top diagnosis**!
   - *Failure Mode (`CONTRADICTED` in backend implementation):* If a patient truly has Diabetic Retinopathy (Emergency, $\alpha = 0.01$), but the network's top argmax is Normal (Routine, $\alpha = 0.05$), the backend uses the looser routine quantile (`q_routine = 0.948`) instead of the conservative emergency quantile (`q_emergency = 0.992`).
   - *Remediation (`PROPOSED`):* Refactor `backend/conformal.py` so the threshold in the loop dynamically evaluates `URGENCY_TIERS[CLASS_NAMES[idx]]`, exactly as formulated in `run_aw_crc_calibration.py`.

---

### 5.4 Empirical Verification: Grade C Coverage Restoration

Source evidence: `models/aw_crc_calibration.json` (Lines 5–46).

| Optical Quality Tier | Admissibility Range $S(X)$ | Standard US-CRC Coverage | Proposed AW-CRC Coverage | Coverage Delta ($\Delta$) | Mean Set Size (AW-CRC) | Singleton % (AW-CRC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grade A (Optimal Media)** | $S \in [0.70, 0.98]$ | **99.85%** | **97.71%** | $-2.14\%$ (Pruned) | 0.98 classes/pt | 98.2% |
| **Grade B (Mild Flare/Haze)** | $S \in [0.60, 0.70]$ | **92.90%** | **92.90%** | $0.00\%$ | 0.93 classes/pt | 93.5% |
| **Grade C (Dense Media/Cataract)**| $S \in [0.50, 0.60]$ | **78.31%** | **97.59%** | **+19.28%** | **1.35 classes/pt** | **72.4%** |
| **Emergency Stratum Overall** | All Tiers | **98.56%** | **98.80%** | **+0.24%** | — | — |
| **Routine Stratum Overall** | All Tiers | **94.95%** | **95.16%** | **+0.21%** | — | — |
| **Full Cohort Combined** | All Tiers | **96.64%** | **96.86%** | **+0.22%** | **1.08 classes/pt** | **96.86%** |

**Audit Findings (`VERIFIED`):**
1. On Grade C degraded media, standard US-CRC suffers catastrophic undercoverage (**78.31%**, falling far below the 95% nominal guarantee) because optical hazing deflates softmax confidence below static thresholds.
2. AW-CRC restores Grade C coverage to **97.59%** by dynamically lowering the admission threshold via $\frac{\hat{q}_{\text{AW}}}{S(X)^\gamma}$.
3. Overall singleton set proportion is **96.86%** with an average prediction set size of **1.08 classes per patient**, proving that coverage restoration is achieved with minimal clinical inflation.

---

## 6. Real-World Domain Shifts & Exchangeability Violations

Conformal guarantees hold **if and only if** calibration and test distributions are exchangeable:
$$\mathcal{D}_{\text{cal}} \sim P_{XY}, \quad \mathcal{D}_{\text{test}} \sim P_{XY}$$

### 6.1 Catastrophic External Breakdown on IDRiD Benchmark

Source evidence: `models/external_validation_report.json` (Lines 1–81).

When the AW-CRC calibrated ensemble is deployed on the external **Indian Diabetic Retinopathy Image Dataset (IDRiD)** ($N = 413$, Kowa VX-10 camera, India):
- **Nominal Conformal Coverage Guarantee:** $\ge 99.0\%$ on DR.
- **Empirical Realized Conformal Coverage:** **0.00%** (`CONTRADICTED` under domain shift).
- **Human Review Trigger Rate:** **100.0%** (All 413 scans escalated).
- **Post-Calibration ECE on IDRiD:** **0.4404** ($[0.3956, 0.4810]$ 95% CI).
- **Specificities:** Crashed to **16.42%** (Ben Graham) and **2.99%** (Reinhard).

### 6.2 Root-Cause Biostatistical Mechanism

1. **Optical Sensor Shift:** The Kowa VX-10 camera produces an entirely different chromatic illumination profile, flash artifact geometry, and peripheral sensor vignette compared to the Western and East Asian training cohorts (ODIR-5K / EyePACS).
2. **Exchangeability Collapse:** Test non-conformity scores $s_{\text{test}} = (1 - P(Y \mid X_{\text{test}})) \cdot S(X_{\text{test}})^\gamma$ are systematically shifted to the right of the calibration distribution $\{s_i\}_{i \in \mathcal{D}_{\text{cal}}}$.
3. **Clinical Implication:** In conformal prediction, when exchangeability is violated, the mathematical coverage guarantee is **completely forfeited**. The system's only saving grace in OphthalmoAI is that the epistemic uncertainty guardrail triggers a **100% human-in-the-loop referral rate**, safely preventing unmonitored autonomous failure.

---

## 7. Actionable Remediation Plan for Phase 6

1. **Patch `backend/conformal.py` Class-Conditional Cutoff:**
   Replace the uniform `top_diagnosis` check with per-class conditional thresholds matching Proposition 2:
   ```python
   for idx, name in enumerate(CLASS_NAMES):
       cls_urgency = URGENCY_TIERS.get(name, "Non-urgent")
       q_cls = self.q_emergency if cls_urgency in {"Emergency", "Urgent"} else self.q_routine
       dyn_cutoff = max(0.005, 1.0 - (q_cls / (eff_s ** gamma_safe)))
       if probs[idx] >= dyn_cutoff:
           prediction_set.append(name)
   ```
2. **Mandatory Domain Shift Warning in Model Card:**
   Explicitly document that conformal risk guarantees apply **exclusively** to devices and optical distributions exchangeable with the training distribution; on external cohorts like IDRiD, coverage defaults to safe escalation rather than autonomous guarantee.
3. **Deprecate Unstacked Soft Voting in Production:**
   Enforce the learned stacking meta-classifier (`meta_classifier.pth`) for all clinical inference pipelines to prevent the calibration inversion observed under soft voting (ECE 0.0787 $\to$ 0.0263).
