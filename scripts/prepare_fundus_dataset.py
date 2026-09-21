"""
Unified Preprocessing and Harmonization Pipeline for Retinal Fundus Images.
Features:
1. Ben Graham Preprocessing (Circular mask, local color constancy, high-frequency enhancement).
2. Maps multi-dataset labels into unified 6-class posterior pole taxonomy:
   - 'Normal'
   - 'Diabetic Retinopathy'
   - 'Glaucoma'
   - 'Cataract'
   - 'Age-related Macular Degeneration'
   - 'Hypertensive Retinopathy / Pathological Myopia'
3. Standardizes images to 384x384 & 224x224.
4. Generates stratified train.csv, val.csv, and test.csv splits.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
IMAGES_OUT_DIR = PROCESSED_DIR / "images"

UNIFIED_CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]

def clinical_anatomical_crop(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Anatomical Color-Preserving Preprocessing:
    1. Aspect-Ratio-Preserving Aperture Cropping: Avoids squashing circular lesions or CDR.
    2. Luminance-Isolated CLAHE: Enhances contrast strictly in L* channel (CIE-LAB),
       preserving true diagnostic colors (hemorrhages, exudates, drusen, rim pallor).
    3. Anti-Aliased Circular Mask: Suppresses high-frequency border step gradients.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3 and img.dtype == np.uint8:
        pass

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 30:
            crop = img[y:y+h, x:x+w]
            # Aspect-ratio preserved padding to square canvas
            max_dim = max(w, h)
            pad_x = (max_dim - w) // 2
            pad_y = (max_dim - h) // 2
            img = cv2.copyMakeBorder(
                crop, pad_y, max_dim - h - pad_y, pad_x, max_dim - w - pad_x,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
    
    # High-dimensional resize with anti-aliasing area interpolation
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # LAB-CLAHE: Enhance luminance only, preserving true diagnostic chrominance
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    
    # Anti-aliased circular aperture mask
    circle_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    cv2.circle(circle_mask, (target_size // 2, target_size // 2), int(target_size * 0.485), 255, -1)
    circle_mask = cv2.GaussianBlur(circle_mask, (5, 5), 1.5)
    mask_3d = circle_mask[:, :, None] / 255.0
    
    result = (enhanced * mask_3d).astype(np.uint8)
    return result

# Backward-compatible alias
ben_graham_crop = clinical_anatomical_crop

def find_images_and_labels():
    records = []
    
    # 1. ODIR-5K dataset
    odir_dir = RAW_DIR / "odir5k"
    if odir_dir.exists():
        for f in odir_dir.rglob("*.png"):
            parent = f.parent.name.lower()
            cls_name = None
            if "normal" in parent:
                cls_name = "Normal"
            elif "diabetes" in parent or "diabetic" in parent:
                cls_name = "Diabetic Retinopathy"
            elif "glaucoma" in parent:
                cls_name = "Glaucoma"
            elif "cataract" in parent:
                cls_name = "Cataract"
            elif "agedegeneration" in parent or "amd" in parent or "macular" in parent:
                cls_name = "Age-related Macular Degeneration"
            elif "hypertension" in parent or "myopia" in parent:
                cls_name = "Hypertensive Retinopathy / Pathological Myopia"
            
            if cls_name:
                records.append({"source": "odir5k", "file_path": str(f), "class": cls_name})

    # 2. APTOS DR dataset
    aptos_dir = RAW_DIR / "aptos_dr"
    if aptos_dir.exists():
        for f in aptos_dir.rglob("*.png"):
            parent = f.parent.name.lower()
            cls_name = None
            if "0" in parent or "normal" in parent or "no_dr" in parent:
                cls_name = "Normal"
            elif any(k in parent for k in ["1", "2", "3", "4", "mild", "moderate", "severe", "proliferative"]):
                cls_name = "Diabetic Retinopathy"
            if cls_name:
                records.append({"source": "aptos_dr", "file_path": str(f), "class": cls_name})

    # 3. Preprocessed fundus dataset
    prep_dir = RAW_DIR / "preprocessed_fundus"
    if prep_dir.exists():
        for f in prep_dir.rglob("*.*"):
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            parent = f.parent.name.lower()
            cls_name = None
            if "cataract" in parent:
                cls_name = "Cataract"
            elif "glaucoma" in parent:
                cls_name = "Glaucoma"
            elif "diabetic" in parent or "retinopathy" in parent:
                cls_name = "Diabetic Retinopathy"
            elif "normal" in parent:
                cls_name = "Normal"
            elif "amd" in parent or "macular" in parent:
                cls_name = "Age-related Macular Degeneration"
            
            if cls_name:
                records.append({"source": "preprocessed_fundus", "file_path": str(f), "class": cls_name})

    return pd.DataFrame(records)

def main():
    print("=" * 70)
    print("HARMONIZING & PREPROCESSING RETINAL FUNDUS CORPUS")
    print("=" * 70)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = find_images_and_labels()
    print(f"Total raw candidate records discovered: {len(df)}")
    if len(df) == 0:
        print("No raw images found yet. Ensure raw dataset download is complete.")
        return
        
    print("\nClass distribution across raw datasets:")
    print(df["class"].value_counts())
    
    # Process images with Ben Graham enhancement
    processed_records = []
    print("\nExecuting Ben Graham circular masking and color normalization...")
    
    # Balance or cap per class to ensure high quality and fast training
    max_per_class = 1500
    dfs = []
    for cls_name, group in df.groupby("class"):
        dfs.append(group.sample(n=min(len(group), max_per_class), random_state=42))
    balanced_df = pd.concat(dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    for idx, row in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
        src_path = Path(row["file_path"])
        out_name = f"fundus_{row['source']}_{idx:05d}.jpg"
        out_file = IMAGES_OUT_DIR / out_name
        
        try:
            raw_img = cv2.imread(str(src_path))
            if raw_img is None:
                continue
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            processed_img = ben_graham_crop(raw_img, target_size=384)
            
            # Save processed image as high quality JPEG
            Image.fromarray(processed_img).save(out_file, quality=95)
            
            processed_records.append({
                "image_id": out_name,
                "relative_path": f"images/{out_name}",
                "class": row["class"],
                "source_dataset": row["source"]
            })
        except Exception as e:
            print(f"Error processing image {src_path}: {e}")
            continue
            
    proc_df = pd.DataFrame(processed_records)
    print(f"\nSuccessfully processed and saved {len(proc_df)} standardized fundus images.")
    
    # Create Stratified Splits (70% Train, 15% Val, 15% Test)
    train_df, temp_df = train_test_split(
        proc_df, test_size=0.30, stratify=proc_df["class"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["class"], random_state=42
    )
    
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    proc_df.to_csv(PROCESSED_DIR / "all_manifest.csv", index=False)
    
    print("\nDataset split summary:")
    print(f" - Train samples: {len(train_df)}")
    print(f" - Validation samples: {len(val_df)}")
    print(f" - Test samples: {len(test_df)}")
    print("\n[OK] Fundus dataset preparation completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()
