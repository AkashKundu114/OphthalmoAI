"""
Ingestion & Harmonization of External Training Data (IDRiD Train Split).

Downloads the official IDRiD training cohort (413 scans), applies Ben Graham
preprocessing (circular mask and Gaussian color subtraction at 384x384),
and creates an augmented training dataset manifest (dataset/processed/train_augmented.csv).
"""

import io
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
IMAGES_OUT_DIR = PROCESSED_DIR / "images"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_external_dataset import ben_graham_crop


def ingest_idrid_training_data():
    print("=" * 70)
    print("INGESTING EXTERNAL IDRiD TRAINING COHORT (413 SCANS)")
    print("=" * 70)

    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_file = hf_hub_download(
        repo_id="amin-nejad/idrid-disease-grading",
        filename="data/train-00000-of-00001-d81b05cdbfbe95cd.parquet",
        repo_type="dataset",
    )
    df_raw = pd.read_parquet(parquet_file)
    print(f"Downloaded IDRiD train split: {len(df_raw)} records")

    new_records = []

    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Processing IDRiD Train"):
        raw_label = int(row["label"])
        # Map to OphthalmoAI taxonomy
        cls_name = "Normal" if raw_label == 0 else "Diabetic Retinopathy"

        img_filename = f"fundus_idrid_train_{idx:05d}.jpg"
        img_out_path = IMAGES_OUT_DIR / img_filename

        if not img_out_path.exists():
            img_bytes = row["image"]["bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(pil_img)
            proc_np = ben_graham_crop(img_np, target_size=384)
            proc_pil = Image.fromarray(proc_np)
            proc_pil.save(img_out_path, format="JPEG", quality=95)

        new_records.append({
            "image_id": img_filename,
            "class": cls_name,
            "source": "idrid_external_train"
        })

    df_new = pd.DataFrame(new_records)
    print("\nExternal IDRiD class distribution:")
    print(df_new["class"].value_counts())

    # Read existing train.csv
    base_train_csv = PROCESSED_DIR / "train.csv"
    if base_train_csv.exists():
        df_base = pd.read_csv(base_train_csv)
        print(f"\nExisting base train samples: {len(df_base)}")
        df_augmented = pd.concat([df_base, df_new], ignore_index=True)
    else:
        df_augmented = df_new

    # Shuffle augmented manifest
    df_augmented = df_augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_csv = PROCESSED_DIR / "train_augmented.csv"
    df_augmented.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved augmented training manifest ({len(df_augmented)} total samples) to:")
    print(f"     {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    ingest_idrid_training_data()
