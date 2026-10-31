#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: Empirical Admissibility-Weighted Conformal Risk Control (AW-CRC)
=============================================================================
Calculates conformal risk control thresholds using real empirical test & calibration
image distributions and optical admissibility scores S(X) computed via fundus_validator.

Formulation:
    s_AW(X, y) = (1 - P_ensemble(y | X)) * S(X)^gamma
Inclusion condition:
    P_ensemble(c | X) >= 1 - (q_hat / S(X)^gamma)

Guarantees:
- As optical admissibility S(X) decreases toward tau_guard = 0.50,
  the inclusion probability threshold relaxes: 1 - q / S(X)^gamma < 1 - q.
- Conformal set size dynamically expands for marginal scans, reflecting optical uncertainty.
- Distribution-free finite-sample coverage guarantee: P(Y in C(X)) >= 1 - alpha.
"""

import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.fundus_validator import validate_fundus_image

MODELS_DIR = ROOT_DIR / "models"
RESEARCH_RES_DIR = ROOT_DIR / "research" / "results"
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"

CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]

EMERGENCY_CLASSES = {1, 2, 4} # DR, Glaucoma, AMD
ROUTINE_CLASSES = {0, 3, 5}   # Normal, Cataract, HR/Myopia


def compute_optical_scores(df: pd.DataFrame, root_path: Path):
    """Computes real optical admissibility scores S(X) using validate_fundus_image."""
    scores = []
    tiers = []
    for _, row in df.iterrows():
        img_path = root_path / row["relative_path"]
        try:
            with Image.open(img_path) as img:
                _, score, _, _ = validate_fundus_image(img)
        except Exception as e:
            score = 0.50
        
        # Clip to [0.50, 1.0] for admitted fundus photographs
        score = max(0.50, min(1.0, float(score)))
        scores.append(score)
        
        if score >= 0.70:
            tiers.append("Grade A")
        elif score >= 0.60:
            tiers.append("Grade B")
        else:
            tiers.append("Grade C")
            
    return np.array(scores), np.array(tiers)


def run_empirical_aw_crc(gamma: float = 1.0, alpha_emerg: float = 0.01, alpha_rout: float = 0.05):
    print("=" * 75)
    print("EMPIRICAL ADMISSIBILITY-WEIGHTED CONFORMAL RISK CONTROL (AW-CRC)")
    print("=" * 75)
    
    calib_csv = PROCESSED_DIR / "calibration_v2.csv"
    test_csv = PROCESSED_DIR / "test_v2.csv"
    
    calib_npz_path = RESEARCH_RES_DIR / "raw_preds_calibration_v2.npz"
    test_npz_path = RESEARCH_RES_DIR / "raw_preds_test_v2.npz"
    
    if not (calib_csv.exists() and test_csv.exists() and calib_npz_path.exists() and test_npz_path.exists()):
        raise FileNotFoundError("Missing v2 calibration/test datasets or raw predictions.")
        
    calib_df = pd.read_csv(calib_csv)
    test_df = pd.read_csv(test_csv)
    
    calib_npz = np.load(calib_npz_path)
    test_npz = np.load(test_npz_path)
    
    val_y = calib_npz["y_true"]
    val_probs = calib_npz["soft_probs"] # Uniform soft ensemble
    n_val = len(val_y)
    
    test_y = test_npz["y_true"]
    test_probs = test_npz["soft_probs"]
    n_test = len(test_y)
    
    print(f"Loaded real empirical predictions:")
    print(f" - Calibration set: n = {n_val}")
    print(f" - Test set (sealed): n = {n_test}")
    
    print("\nComputing biophysical optical admissibility S(X) on calibration and test cohorts...")
    val_s, val_tiers = compute_optical_scores(calib_df, PROCESSED_DIR)
    test_s, test_tiers = compute_optical_scores(test_df, PROCESSED_DIR)
    
    print(f"Calibration Admissibility: Mean S(X) = {np.mean(val_s):.4f} (Grade A: {np.mean(val_tiers=='Grade A')*100:.1f}%, Grade B: {np.mean(val_tiers=='Grade B')*100:.1f}%, Grade C: {np.mean(val_tiers=='Grade C')*100:.1f}%)")
    print(f"Test Admissibility:        Mean S(X) = {np.mean(test_s):.4f} (Grade A: {np.mean(test_tiers=='Grade A')*100:.1f}%, Grade B: {np.mean(test_tiers=='Grade B')*100:.1f}%, Grade C: {np.mean(test_tiers=='Grade C')*100:.1f}%)")
    
    # 1. Standard US-CRC Calibration
    scores_emerg_std = []
    scores_rout_std = []
    for i in range(n_val):
        score = 1.0 - val_probs[i, val_y[i]]
        if val_y[i] in EMERGENCY_CLASSES:
            scores_emerg_std.append(score)
        else:
            scores_rout_std.append(score)
            
    q_level_e = min(1.0, math.ceil((len(scores_emerg_std) + 1) * (1.0 - alpha_emerg)) / len(scores_emerg_std))
    q_level_r = min(1.0, math.ceil((len(scores_rout_std) + 1) * (1.0 - alpha_rout)) / len(scores_rout_std))
    q_hat_e_std = float(np.quantile(scores_emerg_std, q_level_e))
    q_hat_r_std = float(np.quantile(scores_rout_std, q_level_r))
    
    # 2. AW-CRC Calibration
    scores_emerg_aw = []
    scores_rout_aw = []
    for i in range(n_val):
        score_aw = (1.0 - val_probs[i, val_y[i]]) * (val_s[i] ** gamma)
        if val_y[i] in EMERGENCY_CLASSES:
            scores_emerg_aw.append(score_aw)
        else:
            scores_rout_aw.append(score_aw)
            
    q_hat_e_aw = float(np.quantile(scores_emerg_aw, q_level_e))
    q_hat_r_aw = float(np.quantile(scores_rout_aw, q_level_r))
    
    print("\nCalibration Quantiles (Computed from Real Empirical Ensembles):")
    print(f" - Standard US-CRC: Emergency (99%) = {q_hat_e_std:.4f}, Routine (95%) = {q_hat_r_std:.4f}")
    print(f" - Proposed AW-CRC:  Emergency (99%) = {q_hat_e_aw:.4f}, Routine (95%) = {q_hat_r_aw:.4f}")
    
    # 3. Test Evaluation on Real Test Set
    def evaluate_conformal(mode="standard"):
        cov_emerg = []
        cov_rout = []
        set_sizes = []
        empty_sets = 0
        set_sizes_by_tier = {"Grade A": [], "Grade B": [], "Grade C": []}
        cov_by_tier = {"Grade A": [], "Grade B": [], "Grade C": []}
        
        for i in range(n_test):
            y = test_y[i]
            s = test_s[i]
            tier = test_tiers[i]
            p = test_probs[i]
            
            c_set = []
            for c in range(6):
                if c in EMERGENCY_CLASSES:
                    q_std = q_hat_e_std
                    q_aw = q_hat_e_aw
                else:
                    q_std = q_hat_r_std
                    q_aw = q_hat_r_aw
                    
                if mode == "standard":
                    if p[c] >= (1.0 - q_std):
                        c_set.append(c)
                else: # aw_crc
                    thresh = max(0.0, 1.0 - (q_aw / (s ** gamma)))
                    if p[c] >= thresh:
                        c_set.append(c)
            
            # Non-empty safety fallback
            if len(c_set) == 0:
                empty_sets += 1
                c_set = [int(np.argmax(p))]
                
            is_covered = (y in c_set)
            set_sizes.append(len(c_set))
            set_sizes_by_tier[tier].append(len(c_set))
            cov_by_tier[tier].append(float(is_covered))
            
            if y in EMERGENCY_CLASSES:
                cov_emerg.append(float(is_covered))
            else:
                cov_rout.append(float(is_covered))
                
        return {
            "overall_cov": float(np.mean(cov_emerg + cov_rout)),
            "cov_emerg": float(np.mean(cov_emerg)),
            "cov_rout": float(np.mean(cov_rout)),
            "mean_set_size": float(np.mean(set_sizes)),
            "singletons_pct": float(np.mean(np.array(set_sizes) == 1) * 100),
            "empty_sets_rate": float(empty_sets / n_test * 100),
            "tier_set_sizes": {t: float(np.mean(set_sizes_by_tier[t])) if len(set_sizes_by_tier[t]) > 0 else 0.0 for t in set_sizes_by_tier},
            "tier_coverage": {t: float(np.mean(cov_by_tier[t])) * 100 if len(cov_by_tier[t]) > 0 else 0.0 for t in cov_by_tier}
        }
        
    res_std = evaluate_conformal(mode="standard")
    res_aw = evaluate_conformal(mode="aw_crc")
    
    print("\n" + "-" * 75)
    print("REAL EMPIRICAL TEST EVALUATION: STANDARD US-CRC vs. PROPOSED AW-CRC")
    print("-" * 75)
    print(f"{'Conformal Metric':<38} {'Standard US-CRC':<20} {'Proposed AW-CRC'}")
    print("-" * 75)
    print(f"{'Emergency Stratum Coverage (Target >= 99%)':<38} {res_std['cov_emerg']*100:.2f}%               {res_aw['cov_emerg']*100:.2f}%")
    print(f"{'Routine Stratum Coverage (Target >= 95%)':<38} {res_std['cov_rout']*100:.2f}%               {res_aw['cov_rout']*100:.2f}%")
    print(f"{'Overall Empirical Coverage':<38} {res_std['overall_cov']*100:.2f}%               {res_aw['overall_cov']*100:.2f}%")
    print(f"{'Mean Prediction Set Cardinality':<38} {res_std['mean_set_size']:.2f} classes/pt       {res_aw['mean_set_size']:.2f} classes/pt")
    print(f"{'Singleton Prediction Proportion (%)':<38} {res_std['singletons_pct']:.1f}%                {res_aw['singletons_pct']:.1f}%")
    print(f"{'Empty Set Proportion (Fallback Applied)':<38} {res_std['empty_sets_rate']:.2f}%               {res_aw['empty_sets_rate']:.2f}%")
    print("-" * 75)
    print("BREAKDOWN ACROSS EMPIRICAL OPTICAL QUALITY TIERS:")
    for t in ["Grade A", "Grade B", "Grade C"]:
        print(f"  {t:<25} Set Size: {res_std['tier_set_sizes'][t]:.2f} -> {res_aw['tier_set_sizes'][t]:.2f} | Cov: {res_std['tier_coverage'][t]:.1f}% -> {res_aw['tier_coverage'][t]:.1f}%")
    print("=" * 75)
    
    out_calib = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "source": "empirical_v2_clean",
        "n_calibration": n_val,
        "n_test": n_test,
        "gamma": gamma,
        "tau_guard": 0.50,
        "standard_us_crc": {
            "q_emergency": q_hat_e_std,
            "q_routine": q_hat_r_std,
            "results": res_std
        },
        "admissibility_weighted_crc": {
            "q_emergency": q_hat_e_aw,
            "q_routine": q_hat_r_aw,
            "results": res_aw
        }
    }
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_RES_DIR.mkdir(parents=True, exist_ok=True)
    
    out_file1 = MODELS_DIR / "aw_crc_calibration.json"
    out_file2 = RESEARCH_RES_DIR / "aw_crc_empirical_benchmark.json"
    
    with open(out_file1, "w") as f:
        json.dump(out_calib, f, indent=2)
    with open(out_file2, "w") as f:
        json.dump(out_calib, f, indent=2)
        
    print(f"\n[OK] AW-CRC calibration artifacts successfully saved to:")
    print(f"     1. {out_file1}")
    print(f"     2. {out_file2}\n")


if __name__ == "__main__":
    run_empirical_aw_crc()
