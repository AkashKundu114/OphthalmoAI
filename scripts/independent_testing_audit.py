#!/usr/bin/env python3
"""
OphthalmoAI: Independent Testing & Reproducibility Auditor Verification Script
==============================================================================
Executes:
1. Exact metric recalculation on sealed test cohort (test_v2.csv, n=1,389)
   - Overall Accuracy, Balanced Accuracy
   - Per-class Precision, Recall, F1
   - Macro F1, Weighted F1
   - One-vs-Rest Macro AUROC & AUPRC
   - ECE (15 bins) and Multi-class Brier Score
   - Full 6x6 Confusion Matrix
2. Conformal coverage evaluation with and without non-empty argmax fallback:
   - Standard US-CRC and AW-CRC
   - True conformal coverage vs Fallback coverage
   - Empty set frequency and coverage delta
3. Paired statistical significance testing (McNemar's exact two-tailed binomial test)
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.fundus_validator import validate_fundus_image

PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
RESEARCH_RES_DIR = ROOT_DIR / "research" / "results"

TARGET_CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]
NUM_CLASSES = len(TARGET_CLASSES)
EMERGENCY_CLASSES = {1, 2, 4} # DR, Glaucoma, AMD
ROUTINE_CLASSES = {0, 3, 5}   # Normal, Cataract, HR/Myopia

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def compute_multiclass_brier(probs: np.ndarray, labels: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
    # 1/N * sum_{i} sum_{k} (p_{ik} - y_{ik})^2
    one_hot = np.zeros_like(probs)
    for i, l in enumerate(labels):
        if 0 <= l < num_classes:
            one_hot[i, l] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def mcnemar_exact_test(y_true, y_pred1, y_pred2):
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    a = int(np.sum(correct1 & correct2))
    d = int(np.sum(~correct1 & ~correct2))
    n_discordant = b + c
    if n_discordant == 0:
        p_val = 1.0
    else:
        p_val = stats.binomtest(min(b, c), n_discordant, p=0.5, alternative="two-sided").pvalue
    return {
        "a_both_correct": a,
        "b_m1_only": b,
        "c_m2_only": c,
        "d_both_incorrect": d,
        "discordant_pairs": n_discordant,
        "p_value": float(p_val),
        "significant_p05": bool(p_val < 0.05)
    }

def main():
    test_csv = PROCESSED_DIR / "test_v2.csv"
    test_npz_path = RESEARCH_RES_DIR / "raw_preds_test_v2.npz"
    calib_csv = PROCESSED_DIR / "calibration_v2.csv"
    calib_npz_path = RESEARCH_RES_DIR / "raw_preds_calibration_v2.npz"

    test_df = pd.read_csv(test_csv)
    test_data = np.load(test_npz_path)
    calib_df = pd.read_csv(calib_csv)
    calib_data = np.load(calib_npz_path)

    y_test = test_data["y_true"]
    assert len(y_test) == 1389, f"Expected 1389 test cases, got {len(y_test)}"

    models_to_audit = {
        "Meta-Ensemble": test_data["meta_probs"],
        "Soft-Voting": test_data["soft_probs"],
        "ConvNeXt-Small": test_data["convnext_probs"],
        "DenseNet-201": test_data["densenet_probs"],
        "EfficientNet-V2-M": test_data["effnet_probs"],
    }

    print("=" * 80)
    print("TASK 2: INDEPENDENT METRIC RECALCULATION ON TEST_V2.CSV (N=1,389)")
    print("=" * 80)

    audit_metrics = {}

    present_classes = sorted(list(set(y_test))) # [0, 1, 2, 3, 4]
    
    for name, probs in models_to_audit.items():
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, labels=present_classes, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, preds, labels=present_classes, average="weighted", zero_division=0)
        
        # 6x6 confusion matrix
        cm_6x6 = confusion_matrix(y_test, preds, labels=list(range(NUM_CLASSES)))

        # Per-class metrics
        per_class = {}
        aurocs = []
        auprcs = []
        for c in range(NUM_CLASSES):
            c_name = TARGET_CLASSES[c]
            support = int(np.sum(y_test == c))
            tp = int(cm_6x6[c, c])
            fp = int(np.sum(cm_6x6[:, c]) - tp)
            fn = int(np.sum(cm_6x6[c, :]) - tp)
            tn = int(len(y_test) - (tp + fp + fn))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            # One-vs-Rest AUROC and AUPRC
            y_bin = (y_test == c).astype(int)
            if len(np.unique(y_bin)) == 2:
                auc = float(roc_auc_score(y_bin, probs[:, c]))
                prc = float(average_precision_score(y_bin, probs[:, c]))
                aurocs.append(auc)
                auprcs.append(prc)
            else:
                # Class 5 is absent in test_v2
                auc = None
                prc = None

            per_class[c_name] = {
                "support": support,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": float(round(prec, 4)),
                "recall": float(round(rec, 4)),
                "f1": float(round(f1, 4)),
                "auroc": float(round(auc, 4)) if auc is not None else "N/A (support=0)",
                "auprc": float(round(prc, 4)) if prc is not None else "N/A (support=0)",
            }

        macro_auroc = float(np.mean(aurocs))
        macro_auprc = float(np.mean(auprcs))
        ece = compute_ece(probs, y_test, n_bins=15)
        brier = compute_multiclass_brier(probs, y_test, num_classes=NUM_CLASSES)

        audit_metrics[name] = {
            "accuracy": float(round(acc, 4)),
            "balanced_accuracy": float(round(bal_acc, 4)),
            "macro_f1": float(round(macro_f1, 4)),
            "weighted_f1": float(round(weighted_f1, 4)),
            "macro_auroc": float(round(macro_auroc, 4)),
            "macro_auprc": float(round(macro_auprc, 4)),
            "ece_15bins": float(round(ece, 4)),
            "brier_score": float(round(brier, 4)),
            "confusion_matrix_6x6": cm_6x6.tolist(),
            "per_class": per_class
        }

        print(f"\nModel: {name}")
        print(f"  Accuracy:          {acc*100:.2f}% ({np.trace(cm_6x6)} / {len(y_test)})")
        print(f"  Balanced Accuracy: {bal_acc*100:.2f}%")
        print(f"  Macro F1:          {macro_f1:.4f}")
        print(f"  Weighted F1:       {weighted_f1:.4f}")
        print(f"  Macro AUROC (OvR): {macro_auroc:.4f}")
        print(f"  Macro AUPRC (OvR): {macro_auprc:.4f}")
        print(f"  ECE (15 bins):     {ece:.4f}")
        print(f"  Brier Score:       {brier:.4f}")
        print(f"  Confusion Matrix 6x6:")
        for row in cm_6x6:
            print(f"    {row}")

    print("\n" + "=" * 80)
    print("TASK 3: CONFORMAL COVERAGE & FALLBACK AUDIT ON TEST_V2.CSV (N=1,389)")
    print("=" * 80)

    # Compute optical admissibility S(X)
    print("Computing optical admissibility scores S(X) on calibration and test sets...")
    def get_optical_scores(df):
        scores = []
        for _, r in df.iterrows():
            img_path = PROCESSED_DIR / r["relative_path"]
            try:
                with Image.open(img_path) as img:
                    _, s, _, _ = validate_fundus_image(img)
            except Exception:
                s = 0.50
            s = max(0.50, min(1.0, float(s)))
            scores.append(s)
        return np.array(scores)

    val_s = get_optical_scores(calib_df)
    test_s = get_optical_scores(test_df)

    y_val = calib_data["y_true"]
    val_probs_soft = calib_data["soft_probs"]
    val_probs_meta = calib_data["meta_probs"]
    test_probs_soft = test_data["soft_probs"]
    test_probs_meta = test_data["meta_probs"]

    alpha_emerg = 0.01
    alpha_rout = 0.05

    conformal_audit_results = {}

    for ens_name, (v_p, t_p) in [("Uniform Soft-Voting", (val_probs_soft, test_probs_soft)),
                                  ("Meta-Ensemble", (val_probs_meta, test_probs_meta))]:
        # Calibrate Standard US-CRC
        scores_emerg_std = []
        scores_rout_std = []
        for i in range(len(y_val)):
            sc = 1.0 - v_p[i, y_val[i]]
            if y_val[i] in EMERGENCY_CLASSES:
                scores_emerg_std.append(sc)
            else:
                scores_rout_std.append(sc)

        q_level_e = min(1.0, math.ceil((len(scores_emerg_std) + 1) * (1.0 - alpha_emerg)) / len(scores_emerg_std))
        q_level_r = min(1.0, math.ceil((len(scores_rout_std) + 1) * (1.0 - alpha_rout)) / len(scores_rout_std))
        q_e_std = float(np.quantile(scores_emerg_std, q_level_e))
        q_r_std = float(np.quantile(scores_rout_std, q_level_r))

        # Calibrate AW-CRC (gamma=1.0)
        scores_emerg_aw = []
        scores_rout_aw = []
        for i in range(len(y_val)):
            sc = (1.0 - v_p[i, y_val[i]]) * val_s[i]
            if y_val[i] in EMERGENCY_CLASSES:
                scores_emerg_aw.append(sc)
            else:
                scores_rout_aw.append(sc)

        q_e_aw = float(np.quantile(scores_emerg_aw, q_level_e))
        q_r_aw = float(np.quantile(scores_rout_aw, q_level_r))

        for mode, (q_e, q_r) in [("Standard US-CRC", (q_e_std, q_r_std)), ("AW-CRC", (q_e_aw, q_r_aw))]:
            key = f"{ens_name} - {mode}"
            empty_count = 0
            covered_true = []
            covered_fallback = []
            set_sizes_true = []
            set_sizes_fallback = []

            for i in range(len(y_test)):
                y = y_test[i]
                p = t_p[i]
                s = test_s[i]

                c_set = []
                for c in range(NUM_CLASSES):
                    q = q_e if c in EMERGENCY_CLASSES else q_r
                    if mode == "Standard US-CRC":
                        thresh = 1.0 - q
                    else:
                        thresh = max(0.0, 1.0 - (q / s))
                    if p[c] >= thresh:
                        c_set.append(c)

                is_empty = (len(c_set) == 0)
                if is_empty:
                    empty_count += 1
                    # In true conformal prediction, empty set contains 0 elements: y in empty_set is FALSE
                    covered_true.append(0.0)
                    set_sizes_true.append(0)
                    
                    # With fallback, argmax is inserted
                    argmax_c = int(np.argmax(p))
                    covered_fallback.append(1.0 if argmax_c == y else 0.0)
                    set_sizes_fallback.append(1)
                else:
                    covered_true.append(1.0 if y in c_set else 0.0)
                    covered_fallback.append(1.0 if y in c_set else 0.0)
                    set_sizes_true.append(len(c_set))
                    set_sizes_fallback.append(len(c_set))

            empty_freq = empty_count / len(y_test)
            true_cov = float(np.mean(covered_true))
            fb_cov = float(np.mean(covered_fallback))
            delta_cov = fb_cov - true_cov

            conformal_audit_results[key] = {
                "q_emergency": float(round(q_e, 4)),
                "q_routine": float(round(q_r, 4)),
                "empty_set_count": empty_count,
                "empty_set_frequency": float(round(empty_freq, 4)),
                "empty_set_pct": float(round(empty_freq * 100, 2)),
                "true_conformal_coverage": float(round(true_cov, 4)),
                "true_conformal_coverage_pct": float(round(true_cov * 100, 2)),
                "fallback_coverage": float(round(fb_cov, 4)),
                "fallback_coverage_pct": float(round(fb_cov * 100, 2)),
                "delta_coverage_pct": float(round(delta_cov * 100, 2)),
                "mean_set_size_true": float(round(np.mean(set_sizes_true), 4)),
                "mean_set_size_fallback": float(round(np.mean(set_sizes_fallback), 4))
            }

            print(f"\nConfiguration: {key}")
            print(f"  Quantiles: Emergency={q_e:.4f}, Routine={q_r:.4f}")
            print(f"  Empty Sets: {empty_count} / {len(y_test)} ({empty_freq*100:.2f}%)")
            print(f"  TRUE Conformal Coverage (empty=0): {true_cov*100:.2f}%")
            print(f"  Fallback Coverage (argmax on empty): {fb_cov*100:.2f}%")
            print(f"  Coverage Delta (+fallback gain):   {delta_cov*100:.2f}%")
            print(f"  Mean Set Size (True vs Fallback):   {np.mean(set_sizes_true):.4f} vs {np.mean(set_sizes_fallback):.4f}")

    print("\n" + "=" * 80)
    print("TASK 4: PAIRED STATISTICAL TESTING (MCNEMAR'S EXACT TWO-TAILED TEST)")
    print("=" * 80)

    preds_dict = {
        "meta_ensemble": np.argmax(test_data["meta_probs"], axis=1),
        "soft_voting": np.argmax(test_data["soft_probs"], axis=1),
        "convnext_small": np.argmax(test_data["convnext_probs"], axis=1),
        "densenet201": np.argmax(test_data["densenet_probs"], axis=1),
        "effnet_v2_m": np.argmax(test_data["effnet_probs"], axis=1),
    }

    pairs_to_test = [
        ("meta_ensemble", "soft_voting"),
        ("meta_ensemble", "effnet_v2_m"),
        ("meta_ensemble", "densenet201"),
        ("meta_ensemble", "convnext_small"),
        ("soft_voting", "effnet_v2_m"),
        ("soft_voting", "densenet201"),
        ("soft_voting", "convnext_small"),
        ("effnet_v2_m", "densenet201"),
        ("effnet_v2_m", "convnext_small"),
        ("densenet201", "convnext_small"),
    ]

    mcnemar_results = {}
    for m1, m2 in pairs_to_test:
        res = mcnemar_exact_test(y_test, preds_dict[m1], preds_dict[m2])
        mcnemar_results[f"{m1}_vs_{m2}"] = res
        print(f"\nComparison: {m1} vs {m2}")
        print(f"  Contingency: Both Correct={res['a_both_correct']}, M1 Only={res['b_m1_only']}, M2 Only={res['c_m2_only']}, Both Incorrect={res['d_both_incorrect']}")
        print(f"  Discordant Pairs: {res['discordant_pairs']} ({m1} unique={res['b_m1_only']}, {m2} unique={res['c_m2_only']})")
        print(f"  Exact Two-tailed McNemar p-value: {res['p_value']:.4e} (Significant at 0.05: {res['significant_p05']})")

    out_file = ROOT_DIR / "research_audit" / "audit_testing_reproducibility_verified.json"
    with open(out_file, "w") as f:
        json.dump({
            "metrics": audit_metrics,
            "conformal": conformal_audit_results,
            "mcnemar": mcnemar_results
        }, f, indent=2)
    print(f"\n[OK] Fully serialized audit results to {out_file}")

if __name__ == "__main__":
    main()
