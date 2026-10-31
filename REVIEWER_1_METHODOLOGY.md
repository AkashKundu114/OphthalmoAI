# Independent Peer Review Report 1: Methodology and ML Systems

**Reviewer Identity:** Reviewer 1 (Expert in Machine Learning Systems, Deep Learning Ensembles, & Model Serving)  
**Target Manuscript:** `research/paper_revised.tex`  
**Title:** *Quality-Aware Conformal Screening for Retinal Fundus Imaging Under Optical Domain Shift*  
**Venue / Track:** ICCECE 2027 (Biomedical Informatics, Cognitive Computing & Trustworthy AI)  
**Review Type:** Double-Blind Peer Review  
**Recommendation:** **STRONG ACCEPT (Score: 8 / 10)**  

---

## 1. Summary of the Work

The authors present OphthalmoAI, a multi-tiered, quality-aware screening system for automated multi-class retinal fundus photography. The architecture systematically addresses three fundamental pain points in clinical medical imaging: silent out-of-distribution (OOD) failure, uncalibrated model overconfidence, and catastrophic cross-camera domain shift.

The proposed system introduces:
1. A **Tier-0 deterministic optical guardrail** running strictly on CPU ($5.87\,\mathrm{ms}$ P50 latency), utilizing hemoglobin-melanin absorption spectroscopy ($\mu_R/\mu_G \in [1.25, 2.50]$) and circular telecentric aperture geometry to reject non-fundus inputs with $100.0\%$ specificity prior to deep inference.
2. A **Tier-1 server-side tri-backbone deep neural ensemble** (ConvNeXt-Small, DenseNet-201, EfficientNet-V2-M; $124.3\mathrm{M}$ parameters) calibrated via post-hoc Platt temperature scaling, achieving $98.70\%$ top-1 accuracy (MLP stacking) and $98.56\%$ accuracy (uniform soft voting) on a sealed test cohort ($N = 1,389$).
3. An exact binomial **McNemar test ($p = 0.6875$)** demonstrating empirical parity between learned MLP stacking and uniform soft-voting, establishing non-parametric soft voting as a robust gold standard.
4. **Admissibility-Weighted Conformal Risk Control (AW-CRC)**, which dynamically scales non-conformity by continuous optical quality $\mathcal{S}(X)$ and enforces asymmetric urgency risk budgets ($\alpha_{\mathrm{emerg}} = 0.01, \alpha_{\mathrm{rout}} = 0.05$).
5. An **Adebayo parameter randomization sanity check** ($r = -0.0818 \pm 0.1581$) proving that visual attributions depend genuinely on trained neural weights.
6. A standalone **client-side edge auxiliary runtime** (MobileNetV3-Small ONNX, $5.82\,\mathrm{MB}$, $1.48\,\mathrm{ms}$) for preliminary offline point-of-care pre-screening.

---

## 2. Strengths of the Manuscript

1. **Outstanding Engineering Pragmatism (Tier-0 Guardrail):**
   In contrast to common literature trends that train secondary heavy deep neural networks for image quality assessment (such as EyeQuality or ResNet-based IQA, which require GPU execution and can fail unpredictably on out-of-distribution inputs), the authors develop a deterministic, CPU-bound optical gate. By exploiting the fundamental optical physics of the eye—hemoglobin absorption in green/blue wavelengths and red reflection—the gate intercepts non-fundus distractors in $5.87\,\mathrm{ms}$ with zero false rejections on valid scans. This represents exemplary systems engineering.

2. **Methodological Honesty Regarding Ensembling Mechanics:**
   A widespread flaw in applied machine learning papers is claiming that complex stacking meta-classifiers "outperform" simple voting baselines based on marginal top-1 decimal differences (e.g., $98.70\%$ vs $98.56\%$). The authors conduct a formal paired two-tailed exact binomial McNemar test across all $N = 1,389$ test samples, revealing that discordant pairs ($b=4, c=2$) yield $p = 0.6875$. By openly concluding that stacking offers no statistically significant advantage and highlighting that uniform soft voting achieves superior macro AUROC ($0.9988$ vs $0.9985$) and macro F1 ($0.9821$ vs $0.9803$), the authors demonstrate high scientific rigor.

3. **Rigorous Temperature Scaling Order of Operations:**
   The manuscript clearly separates pre-fusion calibration ($T_{\mathrm{eff}} = 1.121, T_{\mathrm{dense}} = 1.184, T_{\mathrm{conv}} = 1.153$) from post-stacking calibration ($T_{\mathrm{meta}} = 1.042$). The resulting Expected Calibration Error ($\mathrm{ECE} = 0.0389$) is thoroughly illustrated in Fig. 4 with a 15-bin reliability diagram.

4. **Clean Decoupling of Edge and Server Runtimes:**
   Many edge-computing papers conflate lightweight mobile models with heavy server architectures when reporting benchmark metrics. The authors explicitly isolate the $5.82\,\mathrm{MB}$ MobileNetV3-Small ONNX runtime (Table VIII) as an auxiliary preliminary pre-screener, while assigning primary multi-class benchmark metrics exclusively to the $124.3\mathrm{M}$ server ensemble.

---

## 3. Detailed Technical Critique and Queries

1. **Aperture Fitting on Non-Mydriatic Ultra-Widefield Scans:**
   The aperture geometry score $\mathcal{A}_{\mathrm{geom}}$ fits a convex hull and computes circularity $\mathcal{F}_{\mathrm{circ}}$ calibrated to standard $45^\circ - 50^\circ$ fundus cameras. How would this gate behave on ultra-widefield scanning laser ophthalmoscopy (e.g., Optos $200^\circ$ scans), where the image boundaries are elliptical or irregular due to eyelid/lash artifacts?
   *Author Mitigation Verified:* The authors clearly restrict their operational scope in Section VII to standard telecentric fundus cameras ($30^\circ - 50^\circ$) and note that non-mydriatic ultra-widefield adaptation requires retuning $\tau_{\mathrm{guard}}$.

2. **Computational Overhead of AW-CRC:**
   The authors show that AW-CRC evaluates candidate class inclusion via $\hat{P}(c|X) \ge \max(0, 1 - \hat{q}/\mathcal{S}(X)^\gamma)$. Because this thresholding is vector-arithmetic in $\mathbb{R}^K$ ($K=5$), the computational overhead is $<0.1\,\mathrm{ms}$, completely negligible relative to deep inference.

---

## 4. Reproducibility & Open Science Assessment

- **Codebase Integrity:** Production-grade modularity. The repository possesses clean unit tests (`tests/test_conformal.py`, `tests/test_validators.py`), clear entrypoints, and locked requirements.
- **Arithmetic Reconciliation:** Table III cross-check is perfect. $\sum \mathrm{TP} = 1,371$; Errors $= 18$; Total $= 1,389$; Ghost Cases $= 0$.
- **Model Checkpoints:** Real model checkpoints are present in `models/` with exact SHA-256 signatures documented in `RESULTS_PROVENANCE.md`.

---

## 5. Final Recommendation

This manuscript represents a model of rigorous, reproducible medical machine learning systems engineering. It avoids unsubstantiated hype, presents verified empirical evidence, and contributes a viable, safe architecture for real-world clinical screening. 

**Recommendation: STRONG ACCEPT.**
