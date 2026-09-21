#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: Clinical Ophthalmic Training Data Augmentation Suite
=================================================================
Expands the patient-clean training partition 3x (from ~4,462 to ~13,386 samples)
using clinically plausible ophthalmic optical transformations:
  1. Telecentric In-Plane Rotations (0 - 360 deg)
  2. Flash / Exposure Jitter (+/- 15%)
  3. Media Haze / Mild Cataract Lens Dispersion
  4. Subtle Field-of-View (FOV) Affine Scaling (30 - 50 deg equivalent)
  5. Horizontal Flip (with biological OD <-> OS laterality swap)

CRITICAL: Applied exclusively to the training partition. Validation and test
partitions remain 100% untouched to preserve unbiased clinical benchmarks.
"""

import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"

def apply_clinical_augmentations(img: np.ndarray, aug_type: int) -> np.ndarray:
    """
    Applies a specific clinically grounded augmentation.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    if aug_type == 1:
        # Telecentric rotation (e.g. 45, 90, 135, 180, 225, 270 deg)
        angle = np.random.choice([35, 70, 110, 160, 215, 290])
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    elif aug_type == 2:
        # Camera flash exposure & contrast variation (mild pupillary dilation variance)
        alpha = np.random.uniform(0.88, 1.14)  # contrast
        beta = np.random.uniform(-12, 14)      # brightness
        aug = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    elif aug_type == 3:
        # Lens media haze / mild cataract simulation (diffuse optical dispersion)
        k_size = np.random.choice([3, 5])
        blurred = cv2.GaussianBlur(img, (k_size, k_size), 0)
        # Blend haze
        haze = np.full_like(img, 45, dtype=np.uint8)
        aug = cv2.addWeighted(blurred, 0.85, haze, 0.15, 0)
    elif aug_type == 4:
        # Subtle FOV Zoom (simulating 30 to 50 deg objective lens change)
        scale = np.random.uniform(0.94, 1.06)
        rot_mat = cv2.getRotationMatrix2D(center, 0, scale)
        aug = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    else:
        # Horizontal flip (mirror)
        aug = cv2.flip(img, 1)

    # Re-enforce circular aperture boundary
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center, int(min(h, w) * 0.485), 255, -1)
    circle_mask = cv2.GaussianBlur(circle_mask, (5, 5), 1.5)
    mask_3d = circle_mask[:, :, None] / 255.0
    
    return (aug * mask_3d).astype(np.uint8)

def augment_training_partition(train_csv: Path, output_csv: Path, multiplier: int = 3):
    print("=" * 70)
    print(f"CLINICAL TRAINING CORPUS 3X AUGMENTATION (MULTIPLIER = {multiplier})")
    print("=" * 70)

    if not train_csv.exists():
        print(f"Error: {train_csv} not found.")
        return

    df_train = pd.read_csv(train_csv)
    print(f"Source clean training samples: {len(df_train)}")
    
    # Load bio-data if available
    bio_path = PROCESSED_DIR / "master_patient_manifest.csv"
    bio_map = {}
    if bio_path.exists():
        df_bio = pd.read_csv(bio_path)
        for _, row in df_bio.iterrows():
            bio_map[row["image_id"]] = row.to_dict()

    augmented_rows = []
    
    # 1. Keep original samples
    for _, row in df_train.iterrows():
        augmented_rows.append(row.to_dict())

    # 2. Synthesize augmented variations
    print(f"\nGenerating {multiplier - 1} augmented variations per training scan...")
    generated_count = 0

    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Augmenting"):
        img_id = row.get("image_id", row.get("relative_path", ""))
        cls_name = row.get("class", row.get("diagnostic_class", "Normal"))
        pid = row.get("patient_id", f"pt_{idx}")
        
        img_path = IMAGES_DIR / img_id
        if not img_path.exists():
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            continue

        base_name = Path(img_id).stem
        ext = Path(img_id).suffix or ".jpg"

        # Generate (multiplier - 1) variations
        for v in range(1, multiplier):
            aug_img = apply_clinical_augmentations(img, aug_type=(v % 5) + 1)
            aug_filename = f"{base_name}_aug_v{v}{ext}"
            aug_out_path = IMAGES_DIR / aug_filename

            if not aug_out_path.exists():
                Image.fromarray(aug_img).save(aug_out_path, format="JPEG", quality=95)

            # Construct row
            new_row = row.to_dict().copy()
            new_row["image_id"] = aug_filename
            new_row["relative_path"] = aug_filename
            new_row["is_augmented"] = True
            new_row["augmentation_variant"] = v
            new_row["patient_id"] = pid  # Keep identical patient ID to preserve clustering
            
            # Laterality tracking: if horizontal flip (aug_type 5), swap OD <-> OS
            if (v % 5) + 1 == 5:
                lat = new_row.get("eye_laterality", "OD")
                new_row["eye_laterality"] = "OS" if lat == "OD" else "OD"

            augmented_rows.append(new_row)
            generated_count += 1

    df_augmented = pd.DataFrame(augmented_rows)
    # Shuffle training set
    df_augmented = df_augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_augmented.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 70)
    print(f"AUGMENTATION SUMMARY:")
    print(f"  Original Clean Samples:  {len(df_train)}")
    print(f"  Synthesized Variations:  {generated_count}")
    print(f"  Total Augmented Train:   {len(df_augmented)}")
    print(f"  Output Manifest:         {output_csv}")
    print("=" * 70)
    print("\nClass Distribution in Augmented Training Set:")
    target_col = "class" if "class" in df_augmented.columns else "diagnostic_class"
    print(df_augmented[target_col].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment training corpus 3x")
    parser.add_argument("--train_csv", type=str, default=str(PROCESSED_DIR / "train_patient_clean.csv"))
    parser.add_argument("--output_csv", type=str, default=str(PROCESSED_DIR / "train_augmented_3x.csv"))
    parser.add_argument("--multiplier", type=int, default=3)
    args = parser.parse_args()

    augment_training_partition(Path(args.train_csv), Path(args.output_csv), args.multiplier)
