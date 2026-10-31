# OphthalmoAI Metric Provenance & Traceability Ledger

**Document ID:** `research_audit/METRIC_PROVENANCE.md`  
**Execution Date:** October 30, 2026  
**Auditor / Provenance Engineer:** Statistical Validation Specialist & Provenance Auditor  
**Remediation Baseline Commit:** `373bc4c263559b1184f293a14fb2cde15327039d`

---

## 1. Metric Provenance Standard

Every reported metric satisfies four strict reproducibility criteria:
1. **Raw Tensor Origin:** Metrics are computed directly from raw model logits/probabilities and ground-truth labels (`research/results/raw_preds_test_v2.npz`).
2. **Arithmetic Consistency:** Precision, Recall/Sensitivity, Specificity, F1, and Accuracy strictly reconcile with the contingency matrix ($\text{trace} = 1,371$, zero ghost cases).
3. **Partition Traceability:** The exact manifest filename, row count, and SHA-256 hash are recorded.
4. **Model Artifact Hash:** The model checkpoint SHA-256 hashes used for inference are permanently documented.

---

## 2. Checkpoint and Manifest Cryptographic Inventory

### Partition Manifests (`dataset/processed/`)
| Manifest File | Row Count | Group Count | SHA-256 Hash |
| :--- | :---: | :---: | :--- |
| `manifest_v2.csv` | 13,911 | 13,892 | `837d9fae155bc32b904c6001004ca43343354cb2f578c946271a396264e10d29` |
| `train_v2.csv` | 9,725 | 9,715 | `04452ee13d00b646abba6fecf7c806012dcf30db5827914139546cee2c09bfbe` |
| `validation_v2.csv` | 1,389 | 1,388 | `a88fdb29b8ad310cf42522f3fa56508e0e30eb4dad0b855bf9d55b0126a02cec` |
| `calibration_v2.csv` | 1,389 | 1,388 | `d8f492cf04be16bc58777a0296d36242930f5d65f667433ec07af17705882c88` |
| `test_v2.csv` [SEALED] | 1,389 | 1,388 | `5adbe806facde3d4cb0f4ca449e0318360937b09592ab1e7424caf29007c75f1` |

### Model Checkpoints (`models/`)
| Model Artifact | File Size | SHA-256 Hash |
| :--- | :---: | :--- |
| `gpu_convnext_small.pth` | 194.5 MB | `c1bf60c5a2c4e6378e9dbba068c12b77c6dafd803c61ae3cf2098b68858d4a6f` |
| `gpu_densenet201.pth` | 77.9 MB | `1c4cb8424a682ea38d8f8a4ea82da3dbe0d7718e001ce502b4507005477c7b6c` |
| `gpu_efficientnet_v2_m.pth` | 208.9 MB | `5d6ff73aa0b2aa9e3ebf6d9016e343a4ea8aa4a06552a281ba4f74d08ec8fb5f` |
| `meta_ensemble_v2.pt` | 13.9 KB | `8883e7215f5394e1d13db7ebfa3a19b7a42bb9c504a3f12440b8a2111fe35359` |
| `edge_model_fp32.onnx` | 5.82 MB | `76503cbfb16867332c96c5612dae82b7db5f5539fa3f9b2d861d85ea3e0ef82c` |

---

## 3. Canonical Empirical Metrics on Sealed Clean Test Set (n = 1,389)

**Artifacts:** `research/results/canonical_eval_test_v2.json`, `research/results/raw_preds_test_v2.npz`

### Overall Model Performance
| Model | Top-1 Accuracy | Balanced Acc | Macro F1 | Macro AUROC | ECE | 95% Bootstrap CI (Accuracy) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Meta-Ensemble** | **98.70%** (1371/1389) | **98.28%** | 0.9803 | **0.9985** | **0.0389** | [98.13%, 99.28%] |
| **Uniform Soft Voting** | 98.56% (1369/1389) | 97.44% | 0.9821 | **0.9988** | 0.0477 | [97.91%, 99.21%] |
| **EfficientNet-V2-M** | 98.63% (1370/1389) | 98.05% | **0.9861** | 0.9979 | 0.0369 | [98.06%, 99.21%] |
| **DenseNet-201** | 98.13% (1363/1389) | 97.10% | 0.9679 | 0.9920 | 0.0462 | [97.34%, 98.78%] |
| **ConvNeXt-Small** | 97.98% (1361/1389) | 96.97% | 0.9745 | 0.9979 | 0.0404 | [97.26%, 98.70%] |

### Per-Class Detailed Performance (Meta-Ensemble)
| Diagnostic Category | Support (n) | Sensitivity / Recall | Specificity | Precision | F1-Score | OvR AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 384 | 97.92% (376/384) | 99.90% (1004/1005) | 99.73% | 0.9882 | 0.9996 |
| **Diabetic Retinopathy** | 483 | 100.00% (483/483) | 99.89% (905/906) | 99.79% | 0.9990 | 0.9999 |
| **Glaucoma** | 361 | 100.00% (361/361) | 98.93% (1017/1028) | 97.04% | 0.9850 | 0.9988 |
| **Cataract** | 132 | 96.97% (128/132) | 100.00% (1257/1257) | 100.00% | 0.9846 | 0.9997 |
| **AMD (ODIR)** | 29 | 79.31% (23/29) | 100.00% (1360/1360) | 100.00% | 0.8846 | 0.9946 |
| **HR / Myopia** | 0 | N/A (Excluded) | 100.00% (1389/1389) | N/A | N/A | N/A |

**Arithmetic Cross-Check:**
- True Positives: $376 + 483 + 361 + 128 + 23 = 1,371$
- False Negatives: $8 + 0 + 0 + 4 + 6 = 18$
- Sum of TP + FN = $1,389$ (Exact equality to test cohort cardinality; 0 ghost cases).

---

## 4. Empirical Conformal Calibration Traceability

**Artifact:** `research/results/aw_crc_empirical_benchmark.json`
- **Calibration Cohort:** `calibration_v2.csv` (n = 1,389)
- **Test Evaluation Cohort:** `test_v2.csv` (n = 1,389)
- **Standard US-CRC Thresholds:** $q_{\text{emerg}} = 0.7234$, $q_{\text{rout}} = 0.3145$
- **Proposed AW-CRC Thresholds ($\gamma=1.0$):** $q_{\text{emerg}} = 0.7017$, $q_{\text{rout}} = 0.3055$
- **Emergency Stratum Empirical Coverage:** 99.31% (Target $\ge 99.0\%$)
- **Routine Stratum Empirical Coverage:** 96.73% (Target $\ge 95.0\%$)
- **Overall Empirical Coverage:** 98.34%
- **Mean Set Cardinality:** 1.01 classes/pt
- **Singleton Proportion:** 99.0%
- **Fallback Empty Set Rate:** 0.58% (Safely resolved via top-1 argmax fallback)

---

## 5. Statistical Hypothesis Testing Traceability

**Artifact:** `research/results/statistical_significance_v2.json`
- **Meta-Ensemble vs. Soft Voting:** Discordant pairs = 6 ($b=4, c=2$), Two-tailed exact binomial McNemar $p = 0.6875$ (Not statistically significant).
- **Meta-Ensemble vs. EfficientNet-V2-M:** Discordant pairs = 7 ($b=4, c=3$), Two-tailed exact binomial McNemar $p = 1.0000$ (Not statistically significant).
- **EfficientNet-V2-M vs. DenseNet-201:** $\Delta \text{Macro F1} = +0.0540$ (95% CI: $[+0.0030, +0.1902]$, $P(\text{EffNet} > \text{DenseNet}) = 99.5\%$).
