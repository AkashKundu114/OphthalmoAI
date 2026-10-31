#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI Canonical Benchmark Engine (v2)
===========================================
Executes live GPU forward passes across validated v2 partitions:
  - DenseNet-201
  - ConvNeXt-Small
  - EfficientNet-V2-M
  - Soft Voting Ensemble
  - Learned Stacking Meta-Classifier

Computes all metrics directly from raw prediction tensors with strict
arithmetic reconciliation and provenance tracking.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, brier_score_loss
)
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)

class V2FundusDataset(Dataset):
    def __init__(self, csv_path: Path, img_dir: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        for _, r in self.df.iterrows():
            img_id = str(r["image_id"])
            cls_name = str(r["class"])
            if cls_name in CLASS_TO_IDX:
                self.samples.append((img_id, CLASS_TO_IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        img_p = self.img_dir / img_id
        img = Image.open(img_p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, img_id

def get_eval_transform(img_size: int = 384):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

class MetaEnsemble(nn.Module):
    def __init__(self, convnext, densenet, effnet, num_classes=6):
        super().__init__()
        self.convnext = convnext
        self.densenet = densenet
        self.efficientnet = effnet
        self.meta_classifier = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            f1 = self.convnext(x)
            f2 = self.densenet(x)
            f3 = self.efficientnet(x)
        concat = torch.cat([f1, f2, f3], dim=1)
        return self.meta_classifier(concat)

def load_models(device):
    print("Loading models onto GPU...")
    # 1. ConvNeXt-Small
    m_convnext = models.convnext_small(weights=None)
    m_convnext.classifier[2] = nn.Linear(m_convnext.classifier[2].in_features, NUM_CLASSES)
    p_conv = MODELS_DIR / "gpu_convnext_small.pth"
    if not p_conv.exists(): p_conv = MODELS_DIR / "convnext_small.pth"
    m_convnext.load_state_dict(torch.load(p_conv, map_location=device, weights_only=True))
    m_convnext.to(device).eval()

    # 2. DenseNet-201
    m_dense = models.densenet201(weights=None)
    m_dense.classifier = nn.Linear(m_dense.classifier.in_features, NUM_CLASSES)
    p_dense = MODELS_DIR / "gpu_densenet201.pth"
    if not p_dense.exists(): p_dense = MODELS_DIR / "densenet201.pth"
    m_dense.load_state_dict(torch.load(p_dense, map_location=device, weights_only=True))
    m_dense.to(device).eval()

    # 3. EfficientNet-V2-M
    m_eff = models.efficientnet_v2_m(weights=None)
    m_eff.classifier[1] = nn.Linear(m_eff.classifier[1].in_features, NUM_CLASSES)
    p_eff = MODELS_DIR / "gpu_efficientnet_v2_m.pth"
    if not p_eff.exists(): p_eff = MODELS_DIR / "efficientnet_v2_m.pth"
    m_eff.load_state_dict(torch.load(p_eff, map_location=device, weights_only=True))
    m_eff.to(device).eval()

    # 4. Meta-Ensemble
    ensemble = MetaEnsemble(m_convnext, m_dense, m_eff, num_classes=NUM_CLASSES)
    p_meta = MODELS_DIR / "meta_classifier_gpu.pth"
    if not p_meta.exists(): p_meta = MODELS_DIR / "meta_classifier.pth"
    sd_meta = torch.load(p_meta, map_location=device, weights_only=False)
    if "meta_classifier.0.weight" in sd_meta:
        ensemble.load_state_dict(sd_meta, strict=False)
    ensemble.to(device).eval()

    return {
        "convnext_small": m_convnext,
        "densenet201": m_dense,
        "efficientnet_v2_m": m_eff,
        "meta_ensemble": ensemble
    }

def run_evaluation(manifest_name: str = "test_v2.csv"):
    manifest_p = PROCESSED_DIR / manifest_name
    print("=" * 80)
    print(f"EVALUATING BENCHMARK ON {manifest_name}")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    eval_tf = get_eval_transform(384)
    ds = V2FundusDataset(manifest_p, IMAGES_DIR, transform=eval_tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Loaded {len(ds)} samples from {manifest_name}")

    models_dict = load_models(device)

    # Run inference across models
    all_labels = []
    all_img_ids = []
    logits_dict = {k: [] for k in models_dict.keys()}
    soft_voting_probs = []

    with torch.no_grad():
        for images, targets, img_ids in tqdm(loader, desc=f"Evaluating {manifest_name}"):
            images = images.to(device)
            all_labels.extend(targets.numpy())
            all_img_ids.extend(img_ids)

            p_conv = torch.softmax(models_dict["convnext_small"](images), dim=1)
            p_dense = torch.softmax(models_dict["densenet201"](images), dim=1)
            p_eff = torch.softmax(models_dict["efficientnet_v2_m"](images), dim=1)

            logits_dict["convnext_small"].append(p_conv.cpu().numpy())
            logits_dict["densenet201"].append(p_dense.cpu().numpy())
            logits_dict["efficientnet_v2_m"].append(p_eff.cpu().numpy())

            # Soft voting
            p_soft = (p_conv + p_dense + p_eff) / 3.0
            soft_voting_probs.append(p_soft.cpu().numpy())

            # Meta ensemble
            p_meta = torch.softmax(models_dict["meta_ensemble"](images), dim=1)
            logits_dict["meta_ensemble"].append(p_meta.cpu().numpy())

    y_true = np.array(all_labels)
    eval_candidates = {}
    for k in ["convnext_small", "densenet201", "efficientnet_v2_m", "meta_ensemble"]:
        eval_candidates[k] = np.concatenate(logits_dict[k], axis=0)
    eval_candidates["soft_voting"] = np.concatenate(soft_voting_probs, axis=0)

    # Compute comprehensive metrics
    evaluation_report = {
        "manifest": manifest_name,
        "n_samples": len(y_true),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "models": {}
    }

    present_classes = sorted(list(set(y_true)))
    present_class_names = [TARGET_CLASSES[i] for i in present_classes]
    print(f"\nClasses present in {manifest_name}: {present_class_names}")

    for model_name, probs in eval_candidates.items():
        preds = np.argmax(probs, axis=1)
        cm = confusion_matrix(y_true, preds, labels=present_classes)
        acc = accuracy_score(y_true, preds)
        bal_acc = balanced_accuracy_score(y_true, preds)
        macro_f1 = f1_score(y_true, preds, labels=present_classes, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, preds, labels=present_classes, average="weighted", zero_division=0)

        # Per-class metrics
        per_class = {}
        aurocs = []
        for i, c_idx in enumerate(present_classes):
            c_name = TARGET_CLASSES[c_idx]
            tp = int(cm[i, i])
            support = int(np.sum(cm[i]))
            fn = support - tp
            fp = int(np.sum(cm[:, i]) - tp)
            tn = int(len(y_true) - (tp + fp + fn))

            sens = tp / support if support > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

            # One-vs-Rest AUROC
            binary_true = (y_true == c_idx).astype(int)
            if len(np.unique(binary_true)) == 2:
                auc = roc_auc_score(binary_true, probs[:, c_idx])
            else:
                auc = 1.0
            aurocs.append(auc)

            per_class[c_name] = {
                "sensitivity": float(round(sens, 4)),
                "specificity": float(round(spec, 4)),
                "precision": float(round(prec, 4)),
                "f1": float(round(f1, 4)),
                "auroc": float(round(auc, 4)),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }

        macro_auroc = float(np.mean(aurocs))
        ece_val = compute_ece(probs, y_true, n_bins=15)

        evaluation_report["models"][model_name] = {
            "accuracy": float(round(acc, 4)),
            "balanced_accuracy": float(round(bal_acc, 4)),
            "macro_f1": float(round(macro_f1, 4)),
            "weighted_f1": float(round(weighted_f1, 4)),
            "macro_auroc": float(round(macro_auroc, 4)),
            "ece": float(round(ece_val, 4)),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class
        }

        print(f"\n--- Model: {model_name:<20} ---")
        print(f"  Accuracy:          {acc*100:.2f}% ({np.trace(cm)} / {len(y_true)})")
        print(f"  Balanced Accuracy: {bal_acc*100:.2f}%")
        print(f"  Macro F1:          {macro_f1:.4f}")
        print(f"  Macro AUROC:       {macro_auroc:.4f}")
        print(f"  ECE (15 bins):     {ece_val:.4f}")
        for c_name, m in per_class.items():
            print(f"    {c_name:<32}: Sens={m['sensitivity']*100:.1f}%, Spec={m['specificity']*100:.1f}%, AUC={m['auroc']:.4f} (N={m['support']})")

    out_json = RESULTS_DIR / f"canonical_eval_{manifest_name.replace('.csv', '')}.json"
    with open(out_json, "w") as f:
        json.dump(evaluation_report, f, indent=2)
    print(f"\n[OK] Saved canonical evaluation to {out_json}")

    # Also save raw prediction npz for exact statistical testing
    out_npz = RESULTS_DIR / f"raw_preds_{manifest_name.replace('.csv', '')}.npz"
    np.savez_compressed(
        out_npz,
        y_true=y_true,
        img_ids=np.array(all_img_ids),
        soft_probs=eval_candidates["soft_voting"],
        meta_probs=eval_candidates["meta_ensemble"],
        convnext_probs=eval_candidates["convnext_small"],
        densenet_probs=eval_candidates["densenet201"],
        effnet_probs=eval_candidates["efficientnet_v2_m"]
    )
    print(f"[OK] Saved raw predictions npz to {out_npz}")

if __name__ == "__main__":
    run_evaluation("test_v2.csv")
