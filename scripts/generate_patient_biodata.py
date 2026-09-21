#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: High-Dimensional Synthetic Patient Bio-Data Generator
===================================================================
Generates clinically grounded electronic health record (EHR) profiles
conditioned on posterior pole ophthalmic pathologies, derived from
published epidemiological distributions (NHANES, UK Biobank, CDC).

Bio-Data Schema (12 Dimensions):
  1. age (years)
  2. sex (Male / Female)
  3. eye_laterality (OD / OS)
  4. hba1c_pct (Glycated Hemoglobin, %)
  5. systolic_bp (mmHg)
  6. diastolic_bp (mmHg)
  7. mean_arterial_pressure (mmHg: DBP + (SBP - DBP) / 3)
  8. pulse_pressure (mmHg: SBP - DBP)
  9. intraocular_pressure_mmhg (IOP, mmHg)
 10. visual_acuity_logmar (LogMAR score: 0.0=20/20, 1.0=20/200)
 11. diabetes_duration_years (years with diabetes diagnosis)
 12. smoking_status (Never / Former / Current)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"

# Epidemiological prior distributions conditioned on target pathology
# Format: (mean, std, min_clip, max_clip)
CLINICAL_PRIORS = {
    "Normal": {
        "age": (44.5, 13.8, 18, 88),
        "female_prob": 0.52,
        "hba1c": (5.2, 0.35, 4.2, 5.9),
        "sbp": (118.0, 10.5, 95, 138),
        "dbp": (76.0, 7.5, 60, 88),
        "iop": (15.2, 2.4, 10.0, 20.5),
        "logmar": (0.02, 0.05, -0.1, 0.15),
        "dm_duration": (0.0, 0.0, 0.0, 0.0),
        "smoking_probs": [0.65, 0.22, 0.13]  # [Never, Former, Current]
    },
    "Diabetic Retinopathy": {
        "age": (59.2, 10.4, 28, 86),
        "female_prob": 0.48,
        "hba1c": (8.6, 1.45, 6.5, 13.8),
        "sbp": (142.0, 16.2, 110, 195),
        "dbp": (86.0, 10.1, 65, 115),
        "iop": (17.8, 3.2, 11.0, 27.0),
        "logmar": (0.38, 0.28, 0.0, 1.6),
        "dm_duration": (13.4, 6.2, 2.0, 35.0),
        "smoking_probs": [0.45, 0.35, 0.20]
    },
    "Glaucoma": {
        "age": (67.8, 8.9, 42, 92),
        "female_prob": 0.54,
        "hba1c": (5.6, 0.6, 4.5, 7.8),
        "sbp": (136.0, 15.0, 105, 180),
        "dbp": (82.0, 9.5, 62, 108),
        "iop": (24.8, 4.6, 17.5, 38.0),
        "logmar": (0.24, 0.22, 0.0, 1.2),
        "dm_duration": (1.5, 3.5, 0.0, 20.0),
        "smoking_probs": [0.55, 0.32, 0.13]
    },
    "Cataract": {
        "age": (71.4, 7.2, 50, 94),
        "female_prob": 0.58,
        "hba1c": (5.7, 0.8, 4.5, 8.5),
        "sbp": (138.0, 14.8, 105, 185),
        "dbp": (81.0, 8.8, 60, 105),
        "iop": (16.0, 2.8, 10.5, 23.0),
        "logmar": (0.58, 0.25, 0.15, 1.8),
        "dm_duration": (2.2, 4.5, 0.0, 25.0),
        "smoking_probs": [0.48, 0.38, 0.14]
    },
    "Age-related Macular Degeneration": {
        "age": (74.8, 6.5, 56, 96),
        "female_prob": 0.60,
        "hba1c": (5.5, 0.55, 4.5, 7.5),
        "sbp": (139.0, 15.5, 108, 185),
        "dbp": (81.5, 9.0, 62, 106),
        "iop": (15.6, 2.6, 10.0, 22.0),
        "logmar": (0.65, 0.34, 0.1, 2.0),
        "dm_duration": (1.2, 3.0, 0.0, 18.0),
        "smoking_probs": [0.32, 0.46, 0.22]  # High smoking prevalence in AMD
    },
    "Hypertensive Retinopathy / Pathological Myopia": {
        "age": (56.4, 12.8, 22, 85),
        "female_prob": 0.50,
        "hba1c": (5.8, 0.9, 4.5, 9.0),
        "sbp": (162.0, 17.5, 135, 215),
        "dbp": (98.0, 11.2, 82, 130),
        "iop": (17.2, 3.1, 11.0, 25.0),
        "logmar": (0.35, 0.30, 0.0, 1.5),
        "dm_duration": (3.5, 5.8, 0.0, 25.0),
        "smoking_probs": [0.42, 0.36, 0.22]
    }
}

def generate_patient_profile(disease_class: str, patient_seed: int, eye_side: str = "OD") -> dict:
    """
    Generates a realistic multidimensional patient record conditioned on diagnosis.
    """
    rng = np.random.default_rng(patient_seed)
    priors = CLINICAL_PRIORS.get(disease_class, CLINICAL_PRIORS["Normal"])

    # Demographics
    age_mean, age_std, age_min, age_max = priors["age"]
    age = int(np.clip(rng.normal(age_mean, age_std), age_min, age_max))
    sex = "Female" if rng.random() < priors["female_prob"] else "Male"

    # Systemic Glycemia & Blood Pressure
    h_mean, h_std, h_min, h_max = priors["hba1c"]
    hba1c = float(np.round(np.clip(rng.normal(h_mean, h_std), h_min, h_max), 2))

    sbp_m, sbp_s, sbp_min, sbp_max = priors["sbp"]
    sbp = float(np.round(np.clip(rng.normal(sbp_m, sbp_s), sbp_min, sbp_max), 1))

    dbp_m, dbp_s, dbp_min, dbp_max = priors["dbp"]
    dbp = float(np.round(np.clip(rng.normal(dbp_m, dbp_s), dbp_min, dbp_max), 1))

    # Hemodynamic derived parameters
    pulse_pressure = float(np.round(sbp - dbp, 1))
    mean_arterial_pressure = float(np.round(dbp + (sbp - dbp) / 3.0, 1))

    # Ophthalmic Biometrics
    iop_m, iop_s, iop_min, iop_max = priors["iop"]
    # Correlate IOP slightly with SBP and HbA1c
    iop_shift = 0.03 * (sbp - 120.0) + 0.25 * (hba1c - 5.5)
    iop = float(np.round(np.clip(rng.normal(iop_m + iop_shift, iop_s), iop_min, iop_max), 1))

    va_m, va_s, va_min, va_max = priors["logmar"]
    logmar = float(np.round(np.clip(rng.normal(va_m, va_s), va_min, va_max), 2))

    dm_m, dm_s, dm_min, dm_max = priors["dm_duration"]
    dm_dur = float(np.round(np.clip(rng.normal(dm_m, dm_s), dm_min, dm_max), 1)) if dm_m > 0 else 0.0

    smoking = rng.choice(["Never", "Former", "Current"], p=priors["smoking_probs"])

    # Clinical risk category based on composite biomarkers
    cardiovascular_risk = "High" if (sbp >= 140 or dbp >= 90 or hba1c >= 8.0) else ("Moderate" if (sbp >= 130 or hba1c >= 6.5) else "Normal")
    ocular_hypertension = bool(iop > 21.0)

    return {
        "age": age,
        "sex": sex,
        "eye_laterality": eye_side,
        "hba1c_pct": hba1c,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "pulse_pressure": pulse_pressure,
        "mean_arterial_pressure": mean_arterial_pressure,
        "intraocular_pressure_mmhg": iop,
        "visual_acuity_logmar": logmar,
        "diabetes_duration_years": dm_dur,
        "smoking_status": smoking,
        "cardiovascular_risk": cardiovascular_risk,
        "ocular_hypertension": ocular_hypertension
    }

def synthesize_corpus_biodata(manifest_csv: Path, output_csv: Path, output_parquet: Path):
    """
    Synthesizes clinical bio-data for every patient entry in the manifest.
    Ensures bilateral scans for the same patient share identical systemic parameters.
    """
    print(f"Reading manifest from {manifest_csv}...")
    df = pd.read_csv(manifest_csv)
    print(f"Total scan entries: {len(df)}")

    # Extract or create patient IDs
    patient_records = {}
    enhanced_rows = []

    # Hash table to keep systemic data constant across bilateral eyes of the same patient
    for idx, row in df.iterrows():
        img_id = str(row.get("image_id", f"img_{idx}"))
        cls_name = str(row.get("class", row.get("diagnostic_class", "Normal")))
        
        # Standardize class name
        if "diabet" in cls_name.lower():
            norm_cls = "Diabetic Retinopathy"
        elif "glauc" in cls_name.lower():
            norm_cls = "Glaucoma"
        elif "cataract" in cls_name.lower():
            norm_cls = "Cataract"
        elif "age" in cls_name.lower() or "amd" in cls_name.lower() or "macul" in cls_name.lower():
            norm_cls = "Age-related Macular Degeneration"
        elif "hypertens" in cls_name.lower() or "myopia" in cls_name.lower():
            norm_cls = "Hypertensive Retinopathy / Pathological Myopia"
        else:
            norm_cls = "Normal"

        # Determine patient seed and laterality
        eye_side = "OS" if ("_left" in img_id.lower() or "_os" in img_id.lower()) else "OD"
        
        # Deduce patient ID from filename if possible
        if "patient" in img_id.lower():
            pid = img_id.split("_")[1]
        elif "odir5k" in img_id.lower():
            # odir5k format: fundus_odir5k_XXXXX.jpg
            pid = f"odir_{int(img_id.replace('fundus_odir5k_', '').split('.')[0]) // 2}"
        else:
            pid = f"pt_{idx // 2}"

        if pid not in patient_records:
            seed_val = abs(hash(pid)) % (2**31 - 1)
            patient_records[pid] = generate_patient_profile(norm_cls, seed_val, eye_side=eye_side)
        
        bio = patient_records[pid].copy()
        bio["patient_id"] = pid
        bio["image_id"] = img_id
        bio["diagnostic_class"] = norm_cls
        bio["eye_laterality"] = eye_side
        
        # Merge existing metadata
        for k, v in row.items():
            if k not in bio:
                bio[k] = v

        enhanced_rows.append(bio)

    df_out = pd.DataFrame(enhanced_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} enhanced patient records to {output_csv}")
    
    try:
        df_out.to_parquet(output_parquet, index=False)
        print(f"Saved parquet to {output_parquet}")
    except Exception as e:
        print(f"Parquet export skipped: {e}")

    # Summary audit
    print("\n" + "=" * 60)
    print("PATIENT BIO-DATA SUMMARY STATS BY DIAGNOSTIC CLASS:")
    print("=" * 60)
    summary = df_out.groupby("diagnostic_class")[["age", "hba1c_pct", "systolic_bp", "intraocular_pressure_mmhg", "visual_acuity_logmar"]].mean()
    print(summary.round(2))
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clinical patient bio-data")
    parser.add_argument("--manifest", type=str, default=str(PROCESSED_DIR / "all_manifest.csv"))
    parser.add_argument("--output_csv", type=str, default=str(PROCESSED_DIR / "master_patient_manifest.csv"))
    parser.add_argument("--output_parquet", type=str, default=str(PROCESSED_DIR / "master_patient_manifest.parquet"))
    args = parser.parse_args()

    synthesize_corpus_biodata(Path(args.manifest), Path(args.output_csv), Path(args.output_parquet))
