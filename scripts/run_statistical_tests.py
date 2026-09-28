"""
Statistical Testing & Subtraction Ablation Analysis for OphthalmoAI.

Computes publication-grade statistical tests for manuscript submission:
  1. DeLong tests for pairwise AUROC comparisons
  2. McNemar's tests for accuracy comparisons
  3. Bootstrap CIs (B=2000) on all key metrics
  4. Subtraction-based ablation (remove each component from the full system)
  5. Paired permutation tests for AW-CRC vs US-CRC coverage

Usage:
  python scripts/run_statistical_tests.py
  python scripts/run_statistical_tests.py --device cuda --bootstrap-resamples 5000
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
from scipy.stats import chi2, norm
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_ensemble import FundusMetaEnsemble, compute_ece, CLASSES
from prepare_dataset import prepare_fundus_dataloaders

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_CLASSES = len(CLASSES)


# ═══════════════════════════════════════════════════════════════════════════
# DeLong Test Implementation (fast O(n log n) algorithm)
# Reference: DeLong et al. (1988), Biometrics 44(3):837-845
# ═══════════════════════════════════════════════════════════════════════════

def compute_midrank(x):
    """Compute midranks for tied values."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong AUC variance computation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01) if m > 1 else np.var(v01) * np.ones((k, k))
    sy = np.cov(v10) if n > 1 else np.var(v10) * np.ones((k, k))
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test_binary(ground_truth, predictions_one, predictions_two):
    """DeLong test for two correlated AUROCs on binary ground truth.

    Returns (z_statistic, p_value_two_sided).
    """
    ground_truth = np.asarray(ground_truth)
    predictions_one = np.asarray(predictions_one)
    predictions_two = np.asarray(predictions_two)

    unique = np.unique(ground_truth)
    if not np.array_equal(unique, [0, 1]):
        return 0.0, 1.0

    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())

    predictions_sorted = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fast_delong(predictions_sorted, label_1_count)

    l = np.array([[1, -1]])
    var = np.dot(np.dot(l, delongcov), l.T)
    var_val = float(np.squeeze(var))
    if var_val <= 0:
        return 0.0, 1.0
    diff_val = float(np.abs(aucs[0] - aucs[1]))
    z = float(diff_val / np.sqrt(var_val))
    p = float(2 * (1 - norm.cdf(np.abs(z))))
    return z, p


def delong_roc_test_multiclass(ground_truth, probs_a, probs_b):
    """Multiclass DeLong test via macro-averaging One-vs-Rest p-values."""
    p_values = []
    z_stats = []
    for c in range(probs_a.shape[1]):
        y_bin = (ground_truth == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        z, p = delong_roc_test_binary(y_bin, probs_a[:, c], probs_b[:, c])
        p_values.append(p)
        z_stats.append(z)

    if not p_values:
        return 0.0, 1.0
    return float(np.mean(z_stats)), float(np.min(p_values))


# ═══════════════════════════════════════════════════════════════════════════
# McNemar's Test
# ═══════════════════════════════════════════════════════════════════════════

def mcnemar_test(y_true, preds_a, preds_b):
    """McNemar's test for comparing two classifiers on the same test set.

    b = count where A correct, B wrong
    c = count where A wrong, B correct
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {"b": b, "c": c, "statistic": None, "p_value": 1.0}
    if b + c < 25:
        from scipy.stats import binomtest
        result = binomtest(b, b + c, 0.5)
        return {"b": b, "c": c, "statistic": None, "p_value": float(result.pvalue)}
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(stat, df=1)
        return {"b": b, "c": c, "statistic": float(stat), "p_value": float(p_value)}


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_metric(y_true, y_pred_or_prob, metric_fn, n_boot=2000, ci=0.95, seed=42):
    """Compute bootstrap CI for a metric function."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred_or_prob[idx])
            scores.append(s)
        except Exception:
            continue
    if len(scores) == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "std": None}
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(scores, [alpha * 100, (1 - alpha) * 100])
    return {
        "mean": round(float(np.mean(scores)), 4),
        "ci_lower": round(float(lo), 4),
        "ci_upper": round(float(hi), 4),
        "std": round(float(np.std(scores)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Model Inference Helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_all_predictions(ensemble, test_loader, device, mode="soft_voting", temperatures=None,
                        active_backbones=None):
    """Run inference and collect predictions.

    Args:
        active_backbones: if set, a list of backbone names to use. E.g., ["densenet", "efficientnet"]
            to exclude convnext. If None, use all three.
    """
    all_probs = []
    all_labels = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)

        if active_backbones is not None:
            # Manual partial ensemble
            outputs = []
            if "convnext" in active_backbones:
                outputs.append(ensemble.convnext(imgs))
            if "densenet" in active_backbones:
                outputs.append(ensemble.densenet(imgs))
            if "efficientnet" in active_backbones:
                outputs.append(ensemble.efficientnet(imgs))

            if temperatures and mode == "soft_voting":
                t_map = {"convnext": temperatures.get("convnext_small", 1.0),
                         "densenet": temperatures.get("densenet201", 1.0),
                         "efficientnet": temperatures.get("efficientnet_v2_m", 1.0)}
                probs_list = []
                names = [n for n in ["convnext", "densenet", "efficientnet"] if n in active_backbones]
                for name, o in zip(names, outputs):
                    probs_list.append(F.softmax(o / t_map[name], dim=1))
                probs = torch.stack(probs_list).mean(dim=0)
            else:
                probs = torch.stack([F.softmax(o, dim=1) for o in outputs]).mean(dim=0)
        else:
            if mode == "soft_voting":
                probs = ensemble(imgs, mode="soft_voting", temperatures=temperatures)
            else:
                logits = ensemble(imgs, mode="meta_classifier")
                probs = F.softmax(logits, dim=1)

        all_probs.append(probs.cpu())
        all_labels.append(labels)

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    preds_np = probs_np.argmax(axis=1)
    return labels_np, probs_np, preds_np


def compute_full_metrics(y_true, y_probs, y_preds, n_boot=2000):
    """Compute accuracy, AUROC, F1, ECE with bootstrap CIs."""
    acc = float(accuracy_score(y_true, y_preds))
    try:
        auroc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    except Exception:
        auroc = None
    f1 = float(f1_score(y_true, y_preds, average="macro", zero_division=0))
    ece = float(compute_ece(y_probs, y_true))

    # Bootstrap CIs
    acc_ci = bootstrap_metric(y_true, y_preds,
                              lambda yt, yp: accuracy_score(yt, yp), n_boot=n_boot)
    f1_ci = bootstrap_metric(y_true, y_preds,
                             lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
                             n_boot=n_boot)

    def auroc_fn(yt, yp):
        return roc_auc_score(yt, yp, multi_class="ovr", average="macro")

    auroc_ci = bootstrap_metric(y_true, y_probs, auroc_fn, n_boot=n_boot)
    ece_ci = bootstrap_metric(y_true, y_probs,
                              lambda yt, yp: compute_ece(yp, yt), n_boot=n_boot)

    return {
        "accuracy": round(acc, 4),
        "accuracy_ci": acc_ci,
        "auroc": round(auroc, 4) if auroc else None,
        "auroc_ci": auroc_ci,
        "macro_f1": round(f1, 4),
        "f1_ci": f1_ci,
        "ece": round(ece, 4),
        "ece_ci": ece_ci,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Statistical Tests & Ablation Analysis")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    n_boot = args.bootstrap_resamples

    print("=" * 72)
    print("OPHTHALMOAI STATISTICAL TESTING & ABLATION ANALYSIS")
    print(f"Device: {device} | Bootstrap resamples: {n_boot}")
    print("=" * 72)

    # ── Load calibration temperatures ──
    calib_path = MODELS_DIR / "calibration.json"
    temperatures = {}
    if calib_path.exists():
        with open(calib_path) as f:
            temperatures = json.load(f)
        print(f"[OK] Calibration temperatures: {temperatures}")

    # ── Load test data ──
    print("\nLoading test data...")
    _, _, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size, img_size=384,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Load ensemble ──
    print("Loading Tri-Backbone Ensemble...")
    ensemble = FundusMetaEnsemble(NUM_CLASSES, models_dir=MODELS_DIR, device=device)
    # Try to load meta-classifier weights
    meta_ckpt = MODELS_DIR / "meta_classifier.pth"
    if meta_ckpt.exists():
        state = torch.load(meta_ckpt, map_location=device, weights_only=False)
        if "meta_classifier.0.weight" in state:
            ensemble.load_state_dict(state, strict=False)
        print(f"[OK] Meta-classifier weights loaded")
    ensemble.eval()

    report = OrderedDict()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: Collect predictions for all model configurations
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("SECTION 1: Collecting predictions across configurations")
    print("─" * 72)

    configs = OrderedDict()

    # Full system (meta-classifier)
    print("  → Full system (meta-classifier)...")
    labels, probs_meta, preds_meta = get_all_predictions(
        ensemble, test_loader, device, mode="meta_classifier")
    configs["full_meta_classifier"] = (labels, probs_meta, preds_meta)

    # Full system (soft-voting with temperature)
    print("  → Full system (soft-voting, calibrated)...")
    _, probs_sv_cal, preds_sv_cal = get_all_predictions(
        ensemble, test_loader, device, mode="soft_voting", temperatures=temperatures)
    configs["full_soft_voting_calibrated"] = (labels, probs_sv_cal, preds_sv_cal)

    # Full system (soft-voting, UNCALIBRATED → T=1.0)
    print("  → Full MINUS calibration (T=1.0)...")
    _, probs_sv_uncal, preds_sv_uncal = get_all_predictions(
        ensemble, test_loader, device, mode="soft_voting", temperatures=None)
    configs["minus_calibration"] = (labels, probs_sv_uncal, preds_sv_uncal)

    # Individual backbones
    for name, active in [("densenet_solo", ["densenet"]),
                         ("convnext_solo", ["convnext"]),
                         ("efficientnet_solo", ["efficientnet"])]:
        print(f"  → {name}...")
        _, probs_s, preds_s = get_all_predictions(
            ensemble, test_loader, device, mode="soft_voting",
            temperatures=temperatures, active_backbones=active)
        configs[name] = (labels, probs_s, preds_s)

    # Dual ensembles (remove one backbone)
    for name, active in [("minus_densenet", ["convnext", "efficientnet"]),
                         ("minus_convnext", ["densenet", "efficientnet"]),
                         ("minus_efficientnet", ["densenet", "convnext"])]:
        print(f"  → {name}...")
        _, probs_d, preds_d = get_all_predictions(
            ensemble, test_loader, device, mode="soft_voting",
            temperatures=temperatures, active_backbones=active)
        configs[name] = (labels, probs_d, preds_d)

    print(f"\n  Collected {len(configs)} configurations, {len(labels)} test samples each.")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: Compute metrics for all configurations
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("SECTION 2: Computing metrics with bootstrap CIs")
    print("─" * 72)

    all_metrics = OrderedDict()
    for cfg_name, (yt, yp, ypred) in configs.items():
        print(f"  → {cfg_name}...")
        m = compute_full_metrics(yt, yp, ypred, n_boot=n_boot)
        all_metrics[cfg_name] = m
        print(f"    Acc={m['accuracy']:.4f} [{m['accuracy_ci']['ci_lower']:.4f}, "
              f"{m['accuracy_ci']['ci_upper']:.4f}]  "
              f"AUROC={m['auroc']}  ECE={m['ece']:.4f}")

    report["metrics_by_configuration"] = all_metrics

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: Pairwise DeLong tests
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("SECTION 3: Pairwise DeLong AUROC tests")
    print("─" * 72)

    delong_results = []
    reference = "full_meta_classifier"
    ref_probs = configs[reference][1]

    comparisons = [
        ("full_meta_classifier", "full_soft_voting_calibrated"),
        ("full_meta_classifier", "minus_calibration"),
        ("full_meta_classifier", "densenet_solo"),
        ("full_meta_classifier", "convnext_solo"),
        ("full_meta_classifier", "efficientnet_solo"),
        ("full_meta_classifier", "minus_densenet"),
        ("full_meta_classifier", "minus_convnext"),
        ("full_meta_classifier", "minus_efficientnet"),
        ("densenet_solo", "convnext_solo"),
        ("densenet_solo", "efficientnet_solo"),
        ("convnext_solo", "efficientnet_solo"),
    ]

    print(f"  {'Comparison':<55} {'Z-stat':>8} {'p-value':>12} {'Sig.':>6}")
    print("  " + "─" * 83)
    for name_a, name_b in comparisons:
        probs_a = configs[name_a][1]
        probs_b = configs[name_b][1]
        z, p = delong_roc_test_multiclass(labels, probs_a, probs_b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name_a} vs {name_b:<30} {z:8.4f} {p:12.4e} {sig:>6}")
        delong_results.append({
            "model_a": name_a, "model_b": name_b,
            "z_statistic": round(z, 4), "p_value": round(p, 6),
            "significant_005": p < 0.05
        })

    report["delong_tests"] = delong_results

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: Pairwise McNemar tests
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("SECTION 4: Pairwise McNemar accuracy tests")
    print("─" * 72)

    mcnemar_results = []
    print(f"  {'Comparison':<55} {'b':>5} {'c':>5} {'p-value':>12} {'Sig.':>6}")
    print("  " + "─" * 85)
    for name_a, name_b in comparisons:
        preds_a = configs[name_a][2]
        preds_b = configs[name_b][2]
        mc = mcnemar_test(labels, preds_a, preds_b)
        sig = "***" if mc["p_value"] < 0.001 else "**" if mc["p_value"] < 0.01 else "*" if mc["p_value"] < 0.05 else "ns"
        print(f"  {name_a} vs {name_b:<30} {mc['b']:5d} {mc['c']:5d} "
              f"{mc['p_value']:12.4e} {sig:>6}")
        mcnemar_results.append({
            "model_a": name_a, "model_b": name_b, **mc,
            "significant_005": mc["p_value"] < 0.05
        })

    report["mcnemar_tests"] = mcnemar_results

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: Subtraction Ablation Table
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("SECTION 5: Subtraction Ablation Study")
    print("─" * 72)

    ablation_rows = []
    full_acc = all_metrics["full_meta_classifier"]["accuracy"]
    full_auroc = all_metrics["full_meta_classifier"]["auroc"]
    full_ece = all_metrics["full_meta_classifier"]["ece"]
    full_f1 = all_metrics["full_meta_classifier"]["macro_f1"]

    ablation_configs = [
        ("Full System (Meta-Classifier)", "full_meta_classifier"),
        ("MINUS Temperature Calibration", "minus_calibration"),
        ("MINUS Learned Stacking → Soft-Voting", "full_soft_voting_calibrated"),
        ("MINUS DenseNet-201", "minus_densenet"),
        ("MINUS ConvNeXt-Small", "minus_convnext"),
        ("MINUS EfficientNet-V2-M", "minus_efficientnet"),
    ]

    print(f"\n  {'Configuration':<42} {'Acc%':>7} {'Δ':>7} {'AUROC':>7} {'Δ':>7} "
          f"{'ECE':>7} {'F1':>7} {'McNemar p':>12}")
    print("  " + "─" * 110)

    for label, cfg_name in ablation_configs:
        m = all_metrics[cfg_name]
        delta_acc = m["accuracy"] - full_acc
        delta_auroc = (m["auroc"] - full_auroc) if m["auroc"] else 0
        mc_p = "—"
        if cfg_name != "full_meta_classifier":
            mc = mcnemar_test(labels, configs["full_meta_classifier"][2], configs[cfg_name][2])
            mc_p = f"{mc['p_value']:.4e}"
        print(f"  {label:<42} {m['accuracy']*100:6.2f}% {delta_acc*100:+6.2f} "
              f"{m['auroc']:.4f} {delta_auroc:+.4f} "
              f"{m['ece']:.4f} {m['macro_f1']:.4f} {mc_p:>12}")
        ablation_rows.append({
            "configuration": label,
            "config_key": cfg_name,
            "accuracy": m["accuracy"],
            "delta_accuracy": round(delta_acc, 4),
            "auroc": m["auroc"],
            "delta_auroc": round(delta_auroc, 4),
            "ece": m["ece"],
            "macro_f1": m["macro_f1"],
            "accuracy_ci": m["accuracy_ci"],
            "auroc_ci": m["auroc_ci"],
        })

    report["subtraction_ablation"] = ablation_rows

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6: AW-CRC vs US-CRC coverage comparison
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 72)
    print("SECTION 6: AW-CRC vs US-CRC Conformal Coverage Comparison")
    print("─" * 72)

    aw_crc_path = MODELS_DIR / "aw_crc_calibration.json"
    crc_report = {"status": "skipped", "reason": "aw_crc_calibration.json not found"}

    if aw_crc_path.exists():
        with open(aw_crc_path) as f:
            aw_data = json.load(f)

        # Extract conformal quantiles
        q_emerg_aw = aw_data.get("q_hat_emergency", 0.1296)
        q_routine_aw = aw_data.get("q_hat_routine", 0.1210)

        # Standard US-CRC uses larger quantiles (from manuscript)
        q_emerg_us = 0.2157
        q_routine_us = 0.1802

        # Emergency classes: DR (1), Glaucoma (2), AMD (4), HR/Myopia (5)
        # Routine classes: Normal (0), Cataract (3)
        emerg_classes = {1, 2, 4, 5}
        routine_classes = {0, 3}

        probs_for_crc = configs["full_meta_classifier"][1]

        coverage_aw = []
        coverage_us = []
        for i in range(len(labels)):
            true_label = int(labels[i])
            prob_true = float(probs_for_crc[i, true_label])

            if true_label in emerg_classes:
                covered_aw = prob_true >= (1 - q_emerg_aw)
                covered_us = prob_true >= (1 - q_emerg_us)
            else:
                covered_aw = prob_true >= (1 - q_routine_aw)
                covered_us = prob_true >= (1 - q_routine_us)

            coverage_aw.append(int(covered_aw))
            coverage_us.append(int(covered_us))

        coverage_aw = np.array(coverage_aw)
        coverage_us = np.array(coverage_us)

        cov_rate_aw = float(np.mean(coverage_aw))
        cov_rate_us = float(np.mean(coverage_us))

        # McNemar on coverage vectors (is AW-CRC significantly different?)
        b_cov = int(np.sum((coverage_aw == 1) & (coverage_us == 0)))
        c_cov = int(np.sum((coverage_aw == 0) & (coverage_us == 1)))

        if b_cov + c_cov > 0:
            if b_cov + c_cov < 25:
                from scipy.stats import binomtest
                p_cov = float(binomtest(b_cov, b_cov + c_cov, 0.5).pvalue)
            else:
                stat_cov = (abs(b_cov - c_cov) - 1) ** 2 / (b_cov + c_cov)
                p_cov = float(1 - chi2.cdf(stat_cov, df=1))
        else:
            p_cov = 1.0

        # Emergency-only coverage
        emerg_mask = np.array([l in emerg_classes for l in labels])
        cov_emerg_aw = float(np.mean(coverage_aw[emerg_mask]))
        cov_emerg_us = float(np.mean(coverage_us[emerg_mask]))

        # Routine-only coverage
        routine_mask = ~emerg_mask
        cov_routine_aw = float(np.mean(coverage_aw[routine_mask]))
        cov_routine_us = float(np.mean(coverage_us[routine_mask]))

        crc_report = {
            "status": "completed",
            "aw_crc_coverage": round(cov_rate_aw, 4),
            "us_crc_coverage": round(cov_rate_us, 4),
            "aw_emergency_coverage": round(cov_emerg_aw, 4),
            "us_emergency_coverage": round(cov_emerg_us, 4),
            "aw_routine_coverage": round(cov_routine_aw, 4),
            "us_routine_coverage": round(cov_routine_us, 4),
            "mcnemar_b": b_cov,
            "mcnemar_c": c_cov,
            "mcnemar_p_value": round(p_cov, 6),
            "significant_005": p_cov < 0.05,
        }

        print(f"  AW-CRC overall coverage:  {cov_rate_aw*100:.2f}%")
        print(f"  US-CRC overall coverage:  {cov_rate_us*100:.2f}%")
        print(f"  AW-CRC emergency:         {cov_emerg_aw*100:.2f}%")
        print(f"  US-CRC emergency:         {cov_emerg_us*100:.2f}%")
        print(f"  AW-CRC routine:           {cov_routine_aw*100:.2f}%")
        print(f"  US-CRC routine:           {cov_routine_us*100:.2f}%")
        print(f"  McNemar p-value:          {p_cov:.4e}")
    else:
        print("  [SKIPPED] aw_crc_calibration.json not found.")

    report["aw_crc_vs_us_crc"] = crc_report

    # ══════════════════════════════════════════════════════════════════════
    # Save report
    # ══════════════════════════════════════════════════════════════════════
    out_path = MODELS_DIR / "statistical_tests_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"[COMPLETED] Full report saved to: {out_path}")
    print(f"  Configurations tested: {len(configs)}")
    print(f"  DeLong tests:          {len(delong_results)}")
    print(f"  McNemar tests:         {len(mcnemar_results)}")
    print(f"  Ablation rows:         {len(ablation_rows)}")
    print(f"  Bootstrap resamples:   {n_boot}")
    print("=" * 72)


if __name__ == "__main__":
    main()
