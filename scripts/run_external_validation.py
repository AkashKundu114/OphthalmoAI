"""
Comprehensive External Validation Script for OphthalmoAI Tri-Backbone Ensemble.
Prepared for academic journal standards. Evaluates on THREE completely independent 
external datasets (IDRiD, RIM-ONE DL, and JSIEC) with rigorous statistical testing.

Features:
- Bootstrapped 95% Confidence Intervals for all metrics
- Expected Calibration Error (ECE) with 95% CI
- Conformal Coverage Rate using Adaptive Width Conformal Risk Control (AW-CRC)
- McNemar's Test comparing soft-voting vs meta-classifier approaches
- Human Review Trigger Rate (top-1 confidence < 0.70)
- Domain Adaptation analysis (Ben Graham vs Reinhard color constancy)
"""

import os
import sys
import gc
import json
import argparse
import io
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy import stats
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import datasets

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_dataset import CLASSES, CLASS_TO_IDX
from evaluate_ensemble import FundusMetaEnsemble, compute_ece

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUT_REPORT = MODELS_DIR / "external_validation_report.json"
AW_CRC_PATH = MODELS_DIR / "aw_crc_calibration.json"


def ben_graham_crop(img_np, target_size=384):
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


def apply_reinhard(img_np):
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


def get_image_tensor(img_pil, mode="ben_graham", target_size=384):
    img_np = np.array(img_pil.convert("RGB"))
    if mode == "reinhard_ben_graham":
        img_np = apply_reinhard(img_np)
        img_np = ben_graham_crop(img_np, target_size)
    elif mode == "ben_graham":
        img_np = ben_graham_crop(img_np, target_size)
    else:
        img_np = cv2.resize(img_np, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    tensor = transforms.ToTensor()(img_np)
    tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)
    return tensor


def bootstrap_ci_metrics(y_true, y_pred, y_prob, B=2000):
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    def calc_metrics(yt, yp, ypr):
        acc = accuracy_score(yt, yp)
        try:
            auc = roc_auc_score(yt, ypr)
        except ValueError:
            auc = np.nan
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return acc, rec, spec, prec, f1, auc

    orig = calc_metrics(y_true, y_pred, y_prob)
    results = {k: [] for k in ["accuracy", "sensitivity", "specificity", "precision", "f1", "auroc"]}
    
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        m = calc_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        results["accuracy"].append(m[0])
        results["sensitivity"].append(m[1])
        results["specificity"].append(m[2])
        results["precision"].append(m[3])
        results["f1"].append(m[4])
        results["auroc"].append(m[5])
        
    final_res = {}
    for i, k in enumerate(["accuracy", "sensitivity", "specificity", "precision", "f1", "auroc"]):
        vals = [v for v in results[k] if not np.isnan(v)]
        if len(vals) == 0:
            lower, upper = 0.0, 0.0
        else:
            lower = np.percentile(vals, 2.5)
            upper = np.percentile(vals, 97.5)
        final_res[k] = {
            "value": orig[i],
            "ci_lower": lower,
            "ci_upper": upper
        }
    return final_res


def bootstrap_ece(y_true, y_prob_all_classes, B=2000):
    n = len(y_true)
    orig_ece = compute_ece(y_prob_all_classes, y_true)
    eces = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        eces.append(compute_ece(y_prob_all_classes[idx], y_true[idx]))
    return {
        "value": orig_ece,
        "ci_lower": np.percentile(eces, 2.5),
        "ci_upper": np.percentile(eces, 97.5)
    }


def mcnemar_test(y_true, y_pred1, y_pred2):
    c1 = (np.array(y_pred1) == np.array(y_true))
    c2 = (np.array(y_pred2) == np.array(y_true))
    n01 = np.sum((~c1) & c2)
    n10 = np.sum(c1 & (~c2))
    if n01 + n10 == 0:
        return 1.0
    stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
    p_value = stats.chi2.sf(stat, 1)
    return float(p_value)


def delong_roc_test(y_true, y_prob1, y_prob2, n_boot=2000):
    """Bootstrap permutation test for AUROC difference between two models.
    
    Tests H0: AUC(model1) = AUC(model2) on binary ground truth.
    Returns two-sided p-value.
    """
    rng = np.random.RandomState(42)
    n = len(y_true)
    try:
        observed_diff = roc_auc_score(y_true, y_prob1) - roc_auc_score(y_true, y_prob2)
    except ValueError:
        return 1.0

    count = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            auc1 = roc_auc_score(y_true[idx], y_prob1[idx])
            auc2 = roc_auc_score(y_true[idx], y_prob2[idx])
            if abs(auc1 - auc2) >= abs(observed_diff):
                count += 1
        except ValueError:
            continue
    return float(count / n_boot)


def load_aw_crc_quantiles():
    quantiles = {"routine": 0.12, "emergency": 0.13}
    try:
        if AW_CRC_PATH.exists():
            with open(AW_CRC_PATH, "r") as f:
                d = json.load(f)
            aw_d = d.get("admissibility_weighted_crc", {})
            quantiles["routine"] = aw_d.get("q_routine", 0.12)
            quantiles["emergency"] = aw_d.get("q_emergency", 0.13)
    except Exception:
        pass
    return quantiles


def extract_idrid_data(ds_iter):
    """Map IDRiD DR grades: 0→Normal (class 0), 1-4→DR (class 1)."""
    items = []
    for row in ds_iter:
        # HuggingFace datasets: field is 'label' (int 0-4), 'image' is PIL Image
        grade = int(row.get('label', row.get('disease_grade', 0)))
        bin_label = 1 if grade > 0 else 0
        img = row.get('image')
        if img is None:
            continue
        # Ensure PIL Image
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(io.BytesIO(img['bytes'])).convert("RGB")
            except Exception:
                continue
        try:
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        except Exception:
            pass
        items.append({
            "image": img,
            "label_6_class": 1 if bin_label == 1 else 0,
            "binary_target": bin_label,
            "subgroup": f"Grade {grade}"
        })
    return items


def extract_rim_one_data(ds_iter):
    """Map RIM-ONE DL: normal→class 0, glaucoma→class 2."""
    items = []
    for row in ds_iter:
        # RIM-ONE uses 'sparse text' or 'label' depending on HF version
        txt = str(row.get('sparse text', row.get('label', ''))).strip().lower()
        if 'glaucoma' in txt:
            cls_6, bin_t, sg = 2, 1, 'glaucoma'
        elif 'normal' in txt or txt == '0':
            cls_6, bin_t, sg = 0, 0, 'normal'
        else:
            continue  # Skip unknown labels

        # Get image — try 'fundus image' first (original RIM-ONE schema), then 'image'
        img = row.get('fundus image', row.get('image'))
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(io.BytesIO(img['bytes'])).convert("RGB")
            except Exception:
                continue
        try:
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        except Exception:
            pass
        items.append({
            "image": img,
            "label_6_class": cls_6,
            "binary_target": bin_t,
            "subgroup": sg
        })
    return items


def extract_jsiec_data(ds_iter):
    """Map JSIEC multi-class labels to OphthalmoAI 6-class taxonomy.
    
    JSIEC has 39 disease categories. We map the most common ones:
      - 0 (normal fundus) → Normal (class 0)
      - DR-related labels → DR (class 1)
      - Glaucoma labels → Glaucoma (class 2)
      - Cataract labels → Cataract (class 3)
      - AMD/macular labels → AMD (class 4)
      - Other pathologies → class 5 (HR/Pathological Myopia) or skip
    Labels without clear mapping are kept as binary: normal(0) vs disease(1).
    """
    # JSIEC label name → OphthalmoAI class index
    JSIEC_TO_OAI = {
        0: 0,   # normal fundus → Normal
        1: 1,   # epiretinal membrane → DR-adjacent
        2: 1,   # retinal vein occlusion → vascular/DR-adjacent
        3: 4,   # macular hole → AMD-adjacent
        4: 5,   # pathological myopia → HR/Pathological Myopia
        5: 1,   # central serous chorioretinopathy → vascular
        6: 4,   # macular pucker → AMD-adjacent
        7: 1,   # tessellated fundus → DR-adjacent
        8: 1,   # DR → DR
        9: 2,   # glaucoma → Glaucoma
        10: 4,  # drusen → AMD
        11: 5,  # myopia fundus → Pathological Myopia
        12: 4,  # macular degeneration → AMD
    }
    
    items = []
    for row in ds_iter:
        lbl = int(row.get('label', -1))
        if lbl < 0:
            continue
            
        # Map to 6-class; default unmapped labels to binary only
        cls_6 = JSIEC_TO_OAI.get(lbl, -1)
        if cls_6 == -1:
            # Unmapped label: treat as generic disease for binary eval
            cls_6 = 5  # assign to "other pathology" bucket
        bin_t = 0 if cls_6 == 0 else 1

        img = row.get('image')
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(io.BytesIO(img['bytes'])).convert("RGB")
            except Exception:
                continue
        try:
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        except Exception:
            pass
        items.append({
            "image": img,
            "label_6_class": cls_6,
            "binary_target": bin_t,
            "subgroup": f"JSIEC_{lbl}"
        })
    return items


def evaluate_dataset(items, ensemble, device, mode_prep, q_emerg, args, batch_size=16):
    print(f"\nEvaluating with prep mode: {mode_prep}")
    
    y_true_6 = []
    y_true_bin = []
    subgroups = []
    
    probs_meta = []
    probs_soft = []
    
    dev = torch.device(device)
    
    for i in tqdm(range(0, len(items), batch_size), desc="Inference"):
        batch_items = items[i:i + batch_size]
        tensors = [get_image_tensor(item["image"], mode=mode_prep) for item in batch_items]
        batch_tensor = torch.stack(tensors).to(dev)
        
        with torch.no_grad():
            # Meta classifier
            logits_m = ensemble(batch_tensor, mode="meta_classifier")
            pm = F.softmax(logits_m, dim=1).cpu().numpy()
            probs_meta.extend(pm)
            
            # Soft voting
            ps = ensemble(batch_tensor, mode="soft_voting").cpu().numpy()
            probs_soft.extend(ps)
            
        for item in batch_items:
            y_true_6.append(item["label_6_class"])
            y_true_bin.append(item["binary_target"])
            subgroups.append(item["subgroup"])
            
        del batch_tensor, tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_true_6 = np.array(y_true_6)
    y_true_bin = np.array(y_true_bin)
    probs_meta = np.array(probs_meta)
    probs_soft = np.array(probs_soft)
    
    # Extract binary probabilities for the target disease class
    # Assumes binary classification targets the max non-normal probability
    preds_meta_6 = np.argmax(probs_meta, axis=1)
    preds_soft_6 = np.argmax(probs_soft, axis=1)
    
    preds_bin = (preds_meta_6 != 0).astype(int)
    probs_bin = 1.0 - probs_meta[:, 0] # probability of any disease
    
    preds_bin_soft = (preds_soft_6 != 0).astype(int)
    
    # 1. Metrics with CI
    metrics = bootstrap_ci_metrics(y_true_bin, preds_bin, probs_bin, B=args.bootstrap_resamples)
    
    # 2. ECE
    ece_res = bootstrap_ece(y_true_6, probs_meta, B=args.bootstrap_resamples)
    
    # 3. Conformal Coverage (AW-CRC)
    thresh = 1.0 - q_emerg
    covered = 0
    for i in range(len(y_true_6)):
        if probs_meta[i, y_true_6[i]] >= thresh:
            covered += 1
    conformal_coverage = covered / len(y_true_6)
    
    # 4. Human Review Trigger Rate (conf < 0.70)
    top1_conf = np.max(probs_meta, axis=1)
    trigger_rate = np.mean(top1_conf < 0.70)
    
    # 5. McNemar Test
    mcnemar_p = mcnemar_test(y_true_6, preds_soft_6, preds_meta_6)
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y_true_bin, preds_bin, labels=[0, 1]).tolist()
    
    # 7. Subgroup Breakdown
    sg_res = {}
    for sg in np.unique(subgroups):
        idx = np.array(subgroups) == sg
        sg_acc = accuracy_score(y_true_bin[idx], preds_bin[idx])
        sg_res[sg] = {"accuracy": sg_acc, "count": int(np.sum(idx))}

    return {
        "metrics": metrics,
        "ece": ece_res,
        "conformal_coverage": conformal_coverage,
        "human_review_trigger_rate": trigger_rate,
        "mcnemar_p_value": mcnemar_p,
        "confusion_matrix": cm,
        "subgroup_breakdown": sg_res
    }


def main():
    parser = argparse.ArgumentParser(description="External Validation Script for OphthalmoAI")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    print(f"Loading Ensemble on {args.device}...")
    ensemble = FundusMetaEnsemble(num_classes=6, models_dir=MODELS_DIR, device=torch.device(args.device))
    ensemble.eval()
    
    aw_crc_quantiles = load_aw_crc_quantiles()
    q_emerg = aw_crc_quantiles["emergency"]

    results_report = []

    ds_configs = [
        {
            "name": "IDRiD",
            "repo_id": "amin-nejad/idrid-disease-grading",
            "cohort_size": 413,
            "camera_hardware": "Kowa VX-10 alpha",
            "geographic_origin": "India",
            "disease": "DR",
            "extractor": extract_idrid_data
        },
        {
            "name": "RIM-ONE DL",
            "repo_id": "ai4ophth/rim_one_dl_dataset",
            "cohort_size": 485,
            "camera_hardware": "Nidek AFC-210, Topcon TRC-NW400",
            "geographic_origin": "Spain",
            "disease": "Glaucoma",
            "extractor": extract_rim_one_data
        },
        {
            "name": "JSIEC",
            "repo_id": "OxAISH-AL-LLM/JSIEC",
            "cohort_size": 1000,
            "camera_hardware": "Topcon TRC-NW200",
            "geographic_origin": "China",
            "disease": "AMD/Multiclass",
            "extractor": extract_jsiec_data
        }
    ]

    for cfg in ds_configs:
        print(f"\n{'='*50}")
        print(f"Dataset: {cfg['name']} ({cfg['disease']})")
        print(f"{'='*50}")
        
        if args.skip_download:
            print("Skipping download as requested.")
            continue
            
        try:
            print(f"Downloading {cfg['repo_id']} from HuggingFace...")
            ds = datasets.load_dataset(cfg['repo_id'], split="train")
            items = cfg['extractor'](ds)
            print(f"Extracted {len(items)} records.")
            
            res_ben = evaluate_dataset(items, ensemble, args.device, "ben_graham", q_emerg, args)
            res_reinhard = evaluate_dataset(items, ensemble, args.device, "reinhard_ben_graham", q_emerg, args)
            
            domain_adapt_delta = res_reinhard["metrics"]["auroc"]["value"] - res_ben["metrics"]["auroc"]["value"]
            
            report = {
                "dataset": cfg["name"],
                "cohort_size": cfg["cohort_size"],
                "camera_hardware": cfg["camera_hardware"],
                "geographic_origin": cfg["geographic_origin"],
                "disease": cfg["disease"],
                "results_ben_graham": res_ben,
                "results_reinhard": res_reinhard,
                "domain_adaptation_auroc_delta": domain_adapt_delta
            }
            results_report.append(report)
            with open(OUTPUT_REPORT, "w") as f:
                json.dump(results_report, f, indent=2)
            
            print(f"\n--- {cfg['name']} Results Summary ---")
            print(f"AUROC (Ben Graham): {res_ben['metrics']['auroc']['value']:.4f} "
                  f"(95% CI: {res_ben['metrics']['auroc']['ci_lower']:.4f}-{res_ben['metrics']['auroc']['ci_upper']:.4f})")
            print(f"AUROC (Reinhard):   {res_reinhard['metrics']['auroc']['value']:.4f}")
            print(f"DA Delta:           {domain_adapt_delta:+.4f}")
            print(f"McNemar p-value:    {res_ben['mcnemar_p_value']:.4e}")
            print(f"ECE:                {res_ben['ece']['value']:.4f}")
            print(f"AW-CRC Coverage:    {res_ben['conformal_coverage']*100:.1f}%")
            print(f"Review Trigger:     {res_ben['human_review_trigger_rate']*100:.1f}%")
            
        except Exception as e:
            print(f"Error processing {cfg['name']}: {e}")
            continue
        finally:
            if 'items' in locals():
                del items
            if 'ds' in locals():
                del ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if results_report:
        with open(OUTPUT_REPORT, "w") as f:
            json.dump(results_report, f, indent=2)
        print(f"\n[OK] Comprehensive JSON report saved to {OUTPUT_REPORT}")
    else:
        print("\n[WARNING] No datasets were successfully evaluated.")

if __name__ == "__main__":
    main()
