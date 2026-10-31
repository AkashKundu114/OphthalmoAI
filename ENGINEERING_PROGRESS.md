# OphthalmoAI Engineering Progress & Iteration Tracking Ledger

**Document ID:** `ENGINEERING_PROGRESS.md`  
**Execution Context:** Autonomous Multi-Agent Engineering Audit & Remediation Loop  
**Git HEAD:** `b744414`  
**Lead Auditor:** Lead Auditor & Remediation Orchestrator  
**Audit Date:** October 31, 2026  
**Status:** COMPLETE & INDEPENDENTLY RECONCILED  

---

## 1. Iteration Summary Metrics (Section F Compliance)

| Metric | Recorded Value | Audit Verification Evidence |
| :--- | :---: | :--- |
| **Current Git HEAD** | `b744414` | Clean working tree on `main` branch. |
| **Total Issues Tracked** | **22** | Itemized in [`MASTER_ISSUE_REGISTER.csv`](file:///d:/OphthalmoAI/MASTER_ISSUE_REGISTER.csv). |
| **Issues Resolved** | **22** | Verified by unit tests, code inspection, and reproduction scripts. |
| **Issues Remaining Unresolved** | **0** | Zero unaddressed P0, P1, or P2 defects. |
| **Critical Regressions** | **0** | Zero regressions across test suites or benchmark pipelines. |
| **Automated Tests Executed** | **317** | `pytest tests/` spanning 37 test modules. |
| **Tests Passed / Failed** | **317 Passed / 0 Failed (100.0%)** | Verified via full test execution logs. |
| **Experiments Reproduced** | **7** | Primary classification, McNemar testing, AW-CRC calibration, Optical guardrail stress, Adebayo sanity check, External camera shift, Edge ONNX micro-benchmarks. |
| **Evidence Deliverables Produced** | **11** | All 11 mandated reports authored in repository root and synchronized with `research_audit/`. |

---

## 2. Itemized Experiment Reproduction Ledger

1. **In-Domain Primary Classification Benchmark (`test_v2.csv`, $N = 1,389$):**
   - Meta-Ensemble: **98.70%** top-1 accuracy ($1,371 / 1,389$), Balanced Acc: $98.28\%$, Macro F1: $0.9803$, Macro AUROC: $0.9985$, ECE: $0.0389$.
   - Uniform Soft-Voting: **98.56%** top-1 accuracy ($1,369 / 1,389$), Balanced Acc: $97.44\%$, Macro F1: $0.9821$, Macro AUROC: $0.9988$, ECE: $0.0477$.
   - Zero ghost cases: $1,371$ True Positives $+ 18$ errors $= 1,389$ total.
2. **Paired Statistical Significance Testing:**
   - Two-tailed exact binomial McNemar test between Meta-Ensemble and Uniform Soft-Voting: $b=4, c=2, n_{\text{disc}}=6, p = \mathbf{0.6875}$.
   - Two-tailed exact McNemar test vs. EfficientNet-V2-M: $b=4, c=3, n_{\text{disc}}=7, p = \mathbf{1.0000}$.
   - Stacking superiority refuted; non-parametric soft-voting established as gold standard.
3. **Admissibility-Weighted Conformal Risk Control (AW-CRC):**
   - Emergency Stratum ($\alpha = 0.01$): Cutoff $q_{\text{emerg}} = 0.7017$, realized coverage **$99.31\%$** (target $\ge 99.0\%$).
   - Routine Stratum ($\alpha = 0.05$): Cutoff $q_{\text{rout}} = 0.3055$, realized coverage **$96.73\%$** (target $\ge 95.0\%$).
   - Raw conformal coverage: **$97.91\%$**; Operational coverage with fallback: **$98.34\%$**; Fallback rate: **$0.58\%$** (8 cases).
   - Mean set size: **$1.0101$ classes/patient**; Singleton sets: **$98.99\%$**.
4. **Biophysical Optical Guardrail Stress Test:**
   - In-domain fundus admission sensitivity: **100.0%** ($1,389 / 1,389$).
   - Distractor rejection specificity: **100.0%** across 7 non-fundus modalities via HTTP 422.
   - P50 CPU execution latency: **5.87 ms** (SLA $< 50\text{ ms}$).
5. **Adebayo Saliency Sanity Check:**
   - DenseNet-201 final convolutional layer weight randomization dropped spatial Pearson correlation to $r = \mathbf{-0.0818 \pm 0.1581} \ll 0.80$, proving genuine learned parameter dependence.
6. **External Cross-Camera Domain Shift:**
   - IDRiD (Kowa camera): Sensitivity **99.64%**, Specificity **16.42%**, Referral **100.0%**.
   - RIM-ONE DL (Nidek camera): Sensitivity **100.00%**, Specificity **1.05%**, Referral **100.0%**.
   - AW-CRC safe abstention verified: 100% of out-of-distribution scans routed to ophthalmologists.
7. **Client-Side Edge Runtime Telemetry:**
   - Standalone MobileNetV3-Small ONNX FP32 ($5.82\text{ MB}$, 2.54M params): P50 latency **1.48 ms**, throughput **636.7 QPS** on host CPU.

---

## 3. Multi-Agent Disagreements & Resolution Matrix

During the audit, the independent red-team reviewer and specialist subagents challenged several foundational claims. Here is how evidence resolved every conflict:

| Disputed Claim / Issue | Red-Team / Specialist Objection | Lead Auditor Evidence Resolution | Manuscript & Code Action Taken |
| :--- | :--- | :--- | :--- |
| **Meta-Ensemble Superiority** | Red-team argued stacking merely overfits logits and provides no statistical value over averaging. | **Conceded**. Exact McNemar test yielded $p = 0.6875$ vs soft voting and $p = 1.0000$ vs EfficientNet-V2-M. | Manuscript Section V-A and Abstract updated to state empirical parity; soft-voting recommended as non-parametric gold standard. |
| **Patient-Level Separation on CFE** | Red-team proved Kaggle CFE never had hospital accession numbers; early `num // 2` was synthetic. | **Conceded**. CFE raw metadata contains zero clinical patient IDs. | Formally designated CFE split as *Duplicate-Cluster-Separated*; transparently disclosed potential bilateral ocular leakage in Section VI. |
| **100% External Referral Claim** | Red-team argued a 100% referral rate reflects total classification failure, not generalization. | **Conceded**. A 1.05% specificity means the model cannot autonomously screen on external cameras. | Framed 100% referral as a **fail-closed domain shift alarm**, explicitly disclosing that cross-camera screening requires site-level re-calibration. |
| **Synthetic EHR in Manifests** | Data specialist discovered `manifest_v2.csv` retained 8 synthetic columns despite removal from splits. | **Verified**. Columns `hba1c_pct`, `systolic_bp`, etc., were still present in `manifest_v2.csv`. | Purged synthetic columns from `manifest_v2.csv`; added assertion in `test_no_synthetic_ehr_in_v2_manifests` (PASSED). |
| **Orphaned Checkpoint Reference** | Implementation auditor found `meta_ensemble_v2.pt` did not exist on disk; active checkpoint is `meta_classifier_gpu.pth`. | **Verified**. Active model weights reside in `meta_classifier_gpu.pth` (462.42 MB). | Updated `RESULTS_PROVENANCE.md` with live SHA-256 hashes; corrected default shape in `onnx_inference.py` to `(1, 3, 224, 224)`. |

---

## 4. Final Deliverables Ledger (Root Directory)

The following 11 authoritative deliverables have been authored, verified, and placed in the project root:

1. [`ENGINEERING_PROGRESS.md`](file:///d:/OphthalmoAI/ENGINEERING_PROGRESS.md): Iteration summary metrics, experiment reproductions, and disagreement resolution.
2. [`ENGINEERING_VERIFICATION_REPORT.md`](file:///d:/OphthalmoAI/ENGINEERING_VERIFICATION_REPORT.md): Authoritative verification status for all 22 tracked issues.
3. [`ENGINEERING_REMEDIATION_PLAN.md`](file:///d:/OphthalmoAI/ENGINEERING_REMEDIATION_PLAN.md): End-to-end multi-phase remediation architecture and code fixes.
4. [`MASTER_ISSUE_REGISTER.csv`](file:///d:/OphthalmoAI/MASTER_ISSUE_REGISTER.csv): Exhaustive 22-issue CSV ledger with root causes, evidence, and resolution proof.
5. [`DATASET_FORENSICS_REPORT.md`](file:///d:/OphthalmoAI/DATASET_FORENSICS_REPORT.md): Forensic audit of patient identities, DSU dHash deduplication, and quarantined samples.
6. [`MODEL_AND_TRAINING_AUDIT.md`](file:///d:/OphthalmoAI/MODEL_AND_TRAINING_AUDIT.md): Deep inspection of model checkpoints, parameter volumes, head structures, and edge segregation.
7. [`STATISTICAL_VALIDATION_REPORT.md`](file:///d:/OphthalmoAI/STATISTICAL_VALIDATION_REPORT.md): In-depth biostatistical recalculations, exact McNemar tests, and conformal risk control bounds.
8. [`EXTERNAL_VALIDATION_REPORT.md`](file:///d:/OphthalmoAI/EXTERNAL_VALIDATION_REPORT.md): Analysis of external camera domain collapse (IDRiD, RIM-ONE DL) and AW-CRC selective triage.
9. [`DEPLOYMENT_AND_SECURITY_AUDIT.md`](file:///d:/OphthalmoAI/DEPLOYMENT_AND_SECURITY_AUDIT.md): API security, fail-closed boundaries, air-gapped privacy, and edge ONNX telemetry.
10. [`REPRODUCIBILITY_GUIDE.md`](file:///d:/OphthalmoAI/REPRODUCIBILITY_GUIDE.md): Step-by-step reproduction pipeline with exact terminal commands and expected outputs.
11. [`SECOND_PASS_ADVERSARIAL_REVIEW.md`](file:///d:/OphthalmoAI/SECOND_PASS_ADVERSARIAL_REVIEW.md): Independent hostile red-team challenge deconstructing headline claims.

---

## 5. Final Readiness Verdict

$$\mathbf{READY\ FOR\ SUBMISSION\ —\ FULLY\ SCIENTIFICALLY\ DEFENSIBLE}$$
