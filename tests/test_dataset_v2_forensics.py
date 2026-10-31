#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for OphthalmoAI Dataset Forensics and Manifest v2 Integrity
"""

import pytest
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
METADATA_DIR = ROOT_DIR / "dataset" / "metadata"

def test_label_harmonization_exists():
    harm_path = METADATA_DIR / "label_harmonization.csv"
    assert harm_path.exists(), "label_harmonization.csv must exist in dataset/metadata"
    df = pd.read_csv(harm_path)
    expected_cols = {
        "source_dataset", "original_label", "target_label", "mapping_rationale",
        "clinical_equivalence", "confidence", "inclusion_status", "reviewer_notes", "evidence_source"
    }
    assert expected_cols.issubset(set(df.columns)), f"Missing required columns in label_harmonization: {expected_cols - set(df.columns)}"

def test_manifest_v2_and_splits_exist():
    for name in ["manifest_v2.csv", "train_v2.csv", "validation_v2.csv", "calibration_v2.csv", "test_v2.csv", "excluded_samples_v2.csv", "duplicate_clusters_v2.csv"]:
        p = PROCESSED_DIR / name
        assert p.exists(), f"Expected manifest file {name} must exist"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Manifest {name} must not be empty"

def test_splits_zero_overlap():
    train_df = pd.read_csv(PROCESSED_DIR / "train_v2.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "validation_v2.csv")
    cal_df = pd.read_csv(PROCESSED_DIR / "calibration_v2.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_v2.csv")

    splits = {
        "Train": train_df,
        "Val": val_df,
        "Cal": cal_df,
        "Test": test_df
    }

    # Zero image_id and zero sha256 overlap
    for n1, d1 in splits.items():
        for n2, d2 in splits.items():
            if n1 >= n2:
                continue
            img_overlap = set(d1["image_id"]).intersection(set(d2["image_id"]))
            assert len(img_overlap) == 0, f"{n1} and {n2} share image_id: {img_overlap}"
            sha_overlap = set(d1["sha256"]).intersection(set(d2["sha256"]))
            assert len(sha_overlap) == 0, f"{n1} and {n2} share sha256 hash: {sha_overlap}"
            group_overlap = set(d1["group_id"]).intersection(set(d2["group_id"]))
            assert len(group_overlap) == 0, f"{n1} and {n2} share group_id: {group_overlap}"

def test_no_synthetic_ehr_in_v2_manifests():
    synthetic_cols = {"hba1c_pct", "systolic_bp", "diastolic_bp", "pulse_pressure", "mean_arterial_pressure", "intraocular_pressure_mmhg", "visual_acuity_logmar", "diabetes_duration_years"}
    for name in ["manifest_v2.csv", "train_v2.csv", "validation_v2.csv", "calibration_v2.csv", "test_v2.csv"]:
        df = pd.read_csv(PROCESSED_DIR / name)
        present = synthetic_cols.intersection(set(df.columns))
        assert len(present) == 0, f"Synthetic EHR columns {present} must not be present in {name}"
