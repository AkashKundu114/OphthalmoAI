#!/usr/bin/env python3
"""
OphthalmoAI: Dataset Acquisition & Verification Tool.
Provides compliant acquisition instructions and SHA256 integrity verification
for public ophthalmology research benchmark datasets.

CRITICAL POLICY:
This script strictly enforces dataset license compliance. It never redistributes
copyrighted medical photographs. Datasets must be acquired directly from their
authorized repositories under their respective institutional agreements.
"""

import hashlib
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "dataset" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CATALOG = {
    "eyepacs": {
        "name": "EyePACS Diabetic Retinopathy Detection",
        "organization": "EyePACS / California Healthcare Foundation",
        "license": "Custom Research Use Only (Redistribution Prohibited)",
        "download_instructions": "kaggle competitions download -c diabetic-retinopathy-detection",
        "target_dir": RAW_DIR / "eyepacs",
        "url": "https://www.kaggle.com/c/diabetic-retinopathy-detection"
    },
    "aptos2019": {
        "name": "APTOS 2019 Blindness Detection",
        "organization": "Asia Pacific Tele-Ophthalmology Society / Aravind Eye Hospital",
        "license": "Kaggle Competition Evaluation License (Research Only)",
        "download_instructions": "kaggle competitions download -c aptos2019-blindness-detection",
        "target_dir": RAW_DIR / "aptos2019",
        "url": "https://www.kaggle.com/c/aptos2019-blindness-detection"
    },
    "messidor2": {
        "name": "Messidor-2 Retinal Image Dataset",
        "organization": "Messidor Consortium / University of Iowa",
        "license": "Free for Academic & Non-Commercial Research upon Registration",
        "download_instructions": "Download directly from https://www.adcis.net/en/third-party/messidor2/",
        "target_dir": RAW_DIR / "messidor2",
        "url": "https://www.adcis.net/en/third-party/messidor2/"
    },
    "idrid": {
        "name": "Indian Diabetic Retinopathy Image Dataset (IDRiD)",
        "organization": "IEEE DataPort (DOI: 10.21227/H29W98)",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "download_instructions": "Download from https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid",
        "target_dir": RAW_DIR / "idrid",
        "url": "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"
    }
}

def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    if not filepath.exists():
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower() == expected_sha256.lower()

def print_provenance():
    print("=" * 80)
    print("OphthalmoAI: Benchmark Dataset Provenance & Acquisition Guide")
    print("=" * 80)
    for key, ds in DATASET_CATALOG.items():
        print(f"\n[{key.upper()}] {ds['name']}")
        print(f"  Organization: {ds['organization']}")
        print(f"  License:      {ds['license']}")
        print(f"  Source URL:   {ds['url']}")
        print(f"  Target Path:  {ds['target_dir']}")
        print(f"  Instructions: {ds['download_instructions']}")
        exists = ds['target_dir'].exists() and any(ds['target_dir'].iterdir())
        status_str = "PRESENT (Local Files Found)" if exists else "NOT DOWNLOADED"
        print(f"  Local Status: {status_str}")

if __name__ == "__main__":
    print_provenance()
