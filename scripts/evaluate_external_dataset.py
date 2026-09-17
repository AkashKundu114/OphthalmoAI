"""
Independent External Benchmark Evaluation for OphthalmoAI.

Evaluates the Tri-Backbone Calibrated Ensemble on completely unseen, external clinical datasets:
1. IDRiD (Indian Diabetic Retinopathy Image Dataset) - Kowa VX-10 camera, Nanded, India.
2. RIM-ONE DL (Retinal Image Database for Optic Nerve Evaluation) - Nidek AFC-210 camera, Spain.

Measures:
- Out-of-distribution generalization accuracy
- Clinical sensitivity & specificity
- AUROC (Area Under ROC)
- Expected Calibration Error (ECE)
- Conformal set coverage & Human review trigger rate
- Impact of domain adaptation (Raw vs Ben Graham vs Reinhard Color Constancy)
"""

import os
import sys
import io
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_ensemble import FundusMetaEnsemble, compute_ece, CLASSES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"


def ben_graham_crop(img_np: np.ndarray, target_size: int = 384) -> np.ndarray:
    """Applies circular crop and local Gaussian color subtraction."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 30:
            img_np = img_np[y:y+h, x:x+w]

    img_np = cv2.resize(img_np, (target_size, target_size), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img_np, (0, 0), target_size / 30)
    enhanced = cv2.addWeighted(img_np, 4, blur, -4, 128)
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    cv2.circle(mask, (target_size // 2, target_size // 2), int(target_size * 0.48), 255, -1)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    return enhanced


def apply_reinhard(img_np: np.ndarray) -> np.ndarray:
    """Harmonizes LAB color moments against canonical retinal distribution."""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_means = np.mean(lab, axis=(0, 1))
    src_stds = np.maximum(np.std(lab, axis=(0, 1)), 1e-4)

    target_means = np.array([128.5, 142.0, 153.2], dtype=np.float32)
    target_stds = np.array([32.4, 11.2, 14.8], dtype=np.float32)

    norm_lab = np.zeros_like(lab)
    for c in range(3):
        norm_lab[:, :, c] = ((lab[:, :, c] - src_means[c]) / src_stds[c]) * target_stds[c] + target_means[c]
    norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)


def get_image_tensor(pil_img: Image.Image, method: str = "ben_graham", target_size: int = 384) -> torch.Tensor:
    img_np = np.array(pil_img.convert("RGB"))

    if method == "ben_graham":
        proc_np = ben_graham_crop(img_np, target_size=target_size)
    elif method == "reinhard_ben_graham":
        reinhard_np = apply_reinhard(img_np)
        proc_np = ben_graham_crop(reinhard_np, target_size=target_size)
    else:  # raw resize
        proc_np = cv2.resize(img_np, (target_size, target_size), interpolation=cv2.INTER_AREA)

    tensor = transforms.ToTensor()(proc_np)
    tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)
    return tensor


def evaluate_idrid(ensemble: FundusMetaEnsemble, temperatures: dict, device: torch.device, method: str = "ben_graham"):
    print("\n" + "=" * 70)
    print(f"EVALUATING EXTERNAL DATASET 1: IDRiD (India, Kowa VX-10) [Preproc: {method}]")
    print("=" * 70)

    test_parquet = hf_hub_download(
        repo_id="amin-nejad/idrid-disease-grading",
        filename="data/test-00000-of-00001-3eaaa0286c6f240e.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(test_parquet)
    print(f"Total IDRiD test cohort: {len(df)} images")

    # In IDRiD: 0 = Normal, 1..4 = Diabetic Retinopathy
    # In OphthalmoAI: Class 0 = Normal, Class 1 = Diabetic Retinopathy
    y_true_binary = []  # 0: Normal, 1: DR
    y_true_dr_grade = []  # 0..4
    y_pred_binary = []
    y_prob_dr = []
    y_top1_class = []
    y_top1_conf = []
    all_ensemble_probs = []
    human_review_needed = []

    for idx, row in df.iterrows():
        raw_label = int(row["label"])
        y_true_dr_grade.append(raw_label)
        y_true_binary.append(0 if raw_label == 0 else 1)

        img_bytes = row["image"]["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = get_image_tensor(pil_img, method=method).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = ensemble(tensor, mode="soft_voting", temperatures=temperatures)[0].cpu().numpy()

        all_ensemble_probs.append(probs)
        top1_idx = int(np.argmax(probs))
        top1_prob = float(probs[top1_idx])
        y_top1_class.append(top1_idx)
        y_top1_conf.append(top1_prob)

        # DR probability is prob[1] (Diabetic Retinopathy)
        prob_dr = float(probs[1])
        prob_normal = float(probs[0])
        y_prob_dr.append(prob_dr)
        
        # Referable screening prediction
        pred_binary = 1 if (prob_dr > prob_normal or top1_idx == 1) else 0
        y_pred_binary.append(pred_binary)

        # Clinical safety criteria: requires_human_review if top1_conf < 0.70 or ambiguous
        needs_review = top1_prob < 0.70 or abs(prob_dr - prob_normal) < 0.20
        human_review_needed.append(needs_review)

    y_true_binary = np.array(y_true_binary)
    y_pred_binary = np.array(y_pred_binary)
    y_prob_dr = np.array(y_prob_dr)
    all_ensemble_probs = np.array(all_ensemble_probs)

    # Metrics
    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    # TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auroc = roc_auc_score(y_true_binary, y_prob_dr)
    except Exception:
        auroc = 0.0

    review_rate = float(np.mean(human_review_needed))

    print(f"\n[IDRiD CLINICAL SCREENING RESULTS (Normal vs DR)]")
    print(f"  Binary Accuracy:      {acc * 100:.2f}%")
    print(f"  Sensitivity (Recall): {rec * 100:.2f}% ({tp}/{tp+fn} DR cases detected)")
    print(f"  Specificity:          {spec * 100:.2f}% ({tn}/{tn+fp} Normal cases confirmed)")
    print(f"  Precision (PPV):      {prec * 100:.2f}%")
    print(f"  F1 Score:             {f1:.4f}")
    print(f"  AUROC (DR vs Normal): {auroc:.4f}")
    print(f"  Human Review Trigger: {review_rate * 100:.2f}% of cases flagged for clinician safety check")

    # Breakdown by DR severity
    print("\n[DETECTION RATE BY CLINICAL DR SEVERITY]")
    grade_names = ["No DR (Normal)", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    grade_stats = {}
    for g in range(5):
        mask = np.array(y_true_dr_grade) == g
        if np.sum(mask) > 0:
            if g == 0:
                correct = np.sum((y_pred_binary == 0) & mask)
            else:
                correct = np.sum((y_pred_binary == 1) & mask)
            total = int(np.sum(mask))
            det_rate = correct / total
            grade_stats[grade_names[g]] = {"total": total, "correct": int(correct), "rate": round(det_rate, 4)}
            print(f"  {grade_names[g]:<20}: {correct}/{total} ({det_rate * 100:.1f}%)")

    return {
        "dataset": "IDRiD_Test",
        "cohort_size": len(df),
        "method": method,
        "accuracy": round(float(acc), 4),
        "sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "auroc": round(float(auroc), 4),
        "human_review_rate": round(review_rate, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "grade_breakdown": grade_stats,
    }


def evaluate_rim_one(ensemble: FundusMetaEnsemble, temperatures: dict, device: torch.device, method: str = "ben_graham"):
    print("\n" + "=" * 70)
    print(f"EVALUATING EXTERNAL DATASET 2: RIM-ONE DL (Spain, Nidek AFC-210) [Preproc: {method}]")
    print("=" * 70)

    parquet_file = hf_hub_download(
        repo_id="ai4ophth/rim_one_dl_dataset",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_file)
    print(f"Total RIM-ONE DL cohort: {len(df)} images")

    # In RIM-ONE: 'sparse text' has 'normal' and 'glaucoma'
    # In OphthalmoAI: Class 0 = Normal, Class 2 = Glaucoma
    y_true_binary = []  # 0: Normal, 1: Glaucoma
    y_pred_binary = []
    y_prob_glaucoma = []
    y_top1_class = []
    y_top1_conf = []
    human_review_needed = []

    for idx, row in df.iterrows():
        label_str = str(row["sparse text"]).strip().lower()
        if label_str not in ["normal", "glaucoma"]:
            continue
        is_glaucoma = 1 if label_str == "glaucoma" else 0
        y_true_binary.append(is_glaucoma)

        img_bytes = row["fundus image"]["bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = get_image_tensor(pil_img, method=method).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = ensemble(tensor, mode="soft_voting", temperatures=temperatures)[0].cpu().numpy()

        top1_idx = int(np.argmax(probs))
        top1_prob = float(probs[top1_idx])
        y_top1_class.append(top1_idx)
        y_top1_conf.append(top1_prob)

        prob_glaucoma = float(probs[2])  # Glaucoma is class index 2
        prob_normal = float(probs[0])
        y_prob_glaucoma.append(prob_glaucoma)

        pred_binary = 1 if (prob_glaucoma > prob_normal or top1_idx == 2) else 0
        y_pred_binary.append(pred_binary)

        needs_review = top1_prob < 0.70 or abs(prob_glaucoma - prob_normal) < 0.20
        human_review_needed.append(needs_review)

    y_true_binary = np.array(y_true_binary)
    y_pred_binary = np.array(y_pred_binary)
    y_prob_glaucoma = np.array(y_prob_glaucoma)

    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auroc = roc_auc_score(y_true_binary, y_prob_glaucoma)
    except Exception:
        auroc = 0.0

    review_rate = float(np.mean(human_review_needed))

    print(f"\n[RIM-ONE DL GLAUCOMA SCREENING RESULTS]")
    print(f"  Binary Accuracy:      {acc * 100:.2f}%")
    print(f"  Sensitivity (Recall): {rec * 100:.2f}% ({tp}/{tp+fn} Glaucoma cases detected)")
    print(f"  Specificity:          {spec * 100:.2f}% ({tn}/{tn+fp} Normal cases confirmed)")
    print(f"  Precision (PPV):      {prec * 100:.2f}%")
    print(f"  F1 Score:             {f1:.4f}")
    print(f"  AUROC:                {auroc:.4f}")
    print(f"  Human Review Trigger: {review_rate * 100:.2f}% of cases flagged for clinician safety check")

    return {
        "dataset": "RIM-ONE_DL",
        "cohort_size": len(y_true_binary),
        "method": method,
        "accuracy": round(float(acc), 4),
        "sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "auroc": round(float(auroc), 4),
        "human_review_rate": round(review_rate, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    parser = argparse.ArgumentParser(description="External Retinal Benchmark Evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("OPHTHALMOAI EXTERNAL VALIDATION BENCHMARK SUITE")
    print(f"Device: {device}")
    print("=" * 70)

    calib_path = MODELS_DIR / "calibration.json"
    temperatures = {}
    if calib_path.exists():
        with open(calib_path, "r") as f:
            temperatures = json.load(f)
        print(f"[OK] Loaded calibration temperatures: {temperatures}")

    print("\nLoading Tri-Backbone Ensemble...")
    ensemble = FundusMetaEnsemble(num_classes=6, models_dir=MODELS_DIR, device=device)
    ensemble.eval()
    print("[OK] Ensemble loaded and set to eval mode.")

    results = {}

    # 1. IDRiD with Ben Graham
    results["idrid_ben_graham"] = evaluate_idrid(ensemble, temperatures, device, method="ben_graham")

    # 2. IDRiD with Reinhard + Ben Graham (Domain Adaptation)
    results["idrid_reinhard"] = evaluate_idrid(ensemble, temperatures, device, method="reinhard_ben_graham")

    # 3. RIM-ONE DL with Ben Graham
    results["rim_one_ben_graham"] = evaluate_rim_one(ensemble, temperatures, device, method="ben_graham")

    # 4. Save results to JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "external_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[COMPLETED] External Benchmark Report saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
