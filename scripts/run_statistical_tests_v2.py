#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: Rigorous Statistical Significance & Hypothesis Testing
===================================================================
Executes McNemar's exact tests and 2,000-iteration paired bootstrap
to evaluate statistical significance between ensemble methods and
constituent deep neural network backbones on the sealed clean test set (n=1,389).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = ROOT_DIR / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def mcnemar_exact_test(y_true, y_pred1, y_pred2):
    """Computes contingency matrix and two-tailed exact binomial McNemar p-value."""
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    # b: model 1 correct, model 2 incorrect
    b = int(np.sum(correct1 & ~correct2))
    # c: model 1 incorrect, model 2 correct
    c = int(np.sum(~correct1 & correct2))
    # a: both correct
    a = int(np.sum(correct1 & correct2))
    # d: both incorrect
    d = int(np.sum(~correct1 & ~correct2))
    
    n_discordant = b + c
    if n_discordant == 0:
        p_val = 1.0
    else:
        # Exact two-tailed binomial test under null hypothesis p = 0.5
        p_val = stats.binomtest(min(b, c), n_discordant, p=0.5, alternative="two-sided").pvalue
        
    return {
        "contingency_matrix": {"a_both_correct": a, "b_m1_only": b, "c_m2_only": c, "d_both_incorrect": d},
        "discordant_pairs": n_discordant,
        "m1_unique_correct": b,
        "m2_unique_correct": c,
        "p_value": float(p_val),
        "statistically_significant_p05": bool(p_val < 0.05)
    }


def paired_bootstrap_evaluation(y_true, pred_dict, B=2000, seed=42):
    """Computes paired bootstrap 95% confidence intervals for metrics and pairwise deltas."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    models = list(pred_dict.keys())
    
    boot_accs = {m: [] for m in models}
    boot_f1s = {m: [] for m in models}
    boot_deltas_f1 = {}
    
    pairs = [
        ("meta_ensemble", "soft_voting"),
        ("meta_ensemble", "effnet_v2_m"),
        ("soft_voting", "effnet_v2_m"),
        ("effnet_v2_m", "densenet201"),
        ("effnet_v2_m", "convnext_small"),
    ]
    for m1, m2 in pairs:
        boot_deltas_f1[f"{m1}_vs_{m2}"] = []
        
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        
        f1_step = {}
        for m in models:
            preds_b = pred_dict[m]["preds"][idx]
            acc_b = accuracy_score(y_b, preds_b)
            f1_b = f1_score(y_b, preds_b, average="macro", zero_division=0)
            boot_accs[m].append(acc_b)
            boot_f1s[m].append(f1_b)
            f1_step[m] = f1_b
            
        for m1, m2 in pairs:
            boot_deltas_f1[f"{m1}_vs_{m2}"].append(f1_step[m1] - f1_step[m2])
            
    ci_results = {}
    for m in models:
        acc_sorted = np.sort(boot_accs[m])
        f1_sorted = np.sort(boot_f1s[m])
        ci_results[m] = {
            "accuracy_95ci": [float(acc_sorted[int(0.025 * B)]), float(acc_sorted[int(0.975 * B)])],
            "macro_f1_95ci": [float(f1_sorted[int(0.025 * B)]), float(f1_sorted[int(0.975 * B)])]
        }
        
    delta_results = {}
    for pair_key, deltas in boot_deltas_f1.items():
        deltas_sorted = np.sort(deltas)
        delta_results[pair_key] = {
            "mean_delta_f1": float(np.mean(deltas)),
            "delta_f1_95ci": [float(deltas_sorted[int(0.025 * B)]), float(deltas_sorted[int(0.975 * B)])],
            "p_superior": float(np.mean(np.array(deltas) > 0))
        }
        
    return ci_results, delta_results


def run_statistical_suite():
    print("=" * 75)
    print("OPHTHALMOAI: RIGOROUS STATISTICAL SIGNIFICANCE & RECONCILIATION")
    print("=" * 75)
    
    npz_path = RESULTS_DIR / "raw_preds_test_v2.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
        
    data = np.load(npz_path)
    y_true = data["y_true"]
    
    models_dict = {
        "meta_ensemble": {"probs": data["meta_probs"], "preds": np.argmax(data["meta_probs"], axis=1)},
        "soft_voting": {"probs": data["soft_probs"], "preds": np.argmax(data["soft_probs"], axis=1)},
        "effnet_v2_m": {"probs": data["effnet_probs"], "preds": np.argmax(data["effnet_probs"], axis=1)},
        "densenet201": {"probs": data["densenet_probs"], "preds": np.argmax(data["densenet_probs"], axis=1)},
        "convnext_small": {"probs": data["convnext_probs"], "preds": np.argmax(data["convnext_probs"], axis=1)},
    }
    
    n_samples = len(y_true)
    print(f"Loaded raw predictions on sealed test set (n = {n_samples})")
    
    print("\n1. PAIRWISE MCNEMAR TESTS (EXACT BINOMIAL):")
    print("-" * 75)
    mcnemar_results = {}
    
    model_keys = list(models_dict.keys())
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            m1 = model_keys[i]
            m2 = model_keys[j]
            res = mcnemar_exact_test(y_true, models_dict[m1]["preds"], models_dict[m2]["preds"])
            pair_name = f"{m1}_vs_{m2}"
            mcnemar_results[pair_name] = res
            
            b = res["m1_unique_correct"]
            c = res["m2_unique_correct"]
            p = res["p_value"]
            sig = "*" if p < 0.05 else "n.s."
            print(f"  {m1:<15} vs {m2:<15} | b={b:<2} c={c:<2} (n_disc={res['discordant_pairs']:<2}) | p = {p:.4f} [{sig}]")
            
    print("\n2. PAIRED STRATIFIED BOOTSTRAP (B = 2,000 iterations):")
    print("-" * 75)
    ci_res, delta_res = paired_bootstrap_evaluation(y_true, models_dict, B=2000, seed=42)
    
    print(f"{'Model':<18} {'Test Accuracy (95% CI)':<30} {'Macro F1 (95% CI)'}")
    print("-" * 75)
    for m in model_keys:
        acc_ci = ci_res[m]["accuracy_95ci"]
        f1_ci = ci_res[m]["macro_f1_95ci"]
        print(f"{m:<18} [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%]              [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
        
    print("\n3. PAIRWISE MACRO F1 DIFFERENCES (95% BOOTSTRAP CI):")
    print("-" * 75)
    for pair, d in delta_res.items():
        delta_ci = d["delta_f1_95ci"]
        print(f"  {pair:<30} Delta F1: {d['mean_delta_f1']:+.4f} (95% CI: [{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]) | P(M1 > M2) = {d['p_superior']*100:.1f}%")
        
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_samples": n_samples,
        "mcnemar_tests": mcnemar_results,
        "bootstrap_confidence_intervals": ci_res,
        "pairwise_f1_deltas": delta_res
    }
    
    out_file = RESULTS_DIR / "statistical_significance_v2.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Statistical test report saved to: {out_file}\n")


if __name__ == "__main__":
    run_statistical_suite()
