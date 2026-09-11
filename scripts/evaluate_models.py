"""
Model Evaluation Script for Retinal Fundus Classifiers.
Evaluates model checkpoints against the held-out test split (dataset/processed/test.csv).
Calculates:
- Test Accuracy & Macro/Weighted F1
- Per-Class Sensitivity, Specificity, Precision, Recall
- Expected Calibration Error (ECE)
- Multi-class AUROC (one-vs-rest)
- Confusion Matrix
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_dataset import prepare_fundus_dataloaders, CLASSES, CLASS_TO_IDX

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_CLASSES = len(CLASSES)

def build_model(arch: str, num_classes: int = NUM_CLASSES):
    if arch == "convnext_small":
        m = models.convnext_small(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif arch == "densenet201":
        m = models.densenet201(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif arch == "efficientnet_v2_m":
        m = models.efficientnet_v2_m(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "efficientnet_b4":
        m = models.efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    total_samples = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences >= lo) & (confidences < hi)
        bin_count = in_bin.sum()
        if bin_count > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += np.abs(bin_acc - bin_conf) * (bin_count / total_samples)
    return float(ece)

def evaluate_checkpoint(checkpoint_path: Path, arch: str, device: torch.device, batch_size: int = 32):
    print("=" * 70)
    print(f"EVALUATING MODEL: {arch.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("=" * 70)

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        return None

    _, _, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=batch_size,
        img_size=384,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    model = build_model(arch, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits_tensor = torch.cat(all_logits)
    labels_np = torch.cat(all_labels).numpy()
    probs_np = F.softmax(logits_tensor, dim=1).numpy()
    preds_np = probs_np.argmax(axis=1)

    acc = float(accuracy_score(labels_np, preds_np))
    macro_f1 = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels_np, preds_np, average="weighted", zero_division=0))
    ece = compute_ece(probs_np, labels_np)

    try:
        auroc = float(roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro"))
    except Exception:
        auroc = None

    cm = confusion_matrix(labels_np, preds_np)

    print(f"\n[SUMMARY METRICS]")
    print(f"  Test Accuracy:        {acc * 100:.2f}%")
    print(f"  Macro F1 Score:       {macro_f1:.4f}")
    print(f"  Weighted F1 Score:    {weighted_f1:.4f}")
    print(f"  ECE (Calibration):    {ece:.4f}")
    if auroc:
        print(f"  Macro AUROC:          {auroc:.4f}")

    print(f"\n[PER-CLASS CLASSIFICATION REPORT]")
    print(classification_report(labels_np, preds_np, target_names=CLASSES, digits=4, zero_division=0))

    per_class_stats = {}
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - (tp + fn + fp))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_stats[name] = {
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn
        }

    report = {
        "architecture": arch,
        "checkpoint": str(checkpoint_path),
        "test_samples": len(labels_np),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "ece": round(ece, 4),
            "macro_auroc": round(auroc, 4) if auroc else None
        },
        "per_class": per_class_stats
    }

    report_path = MODELS_DIR / f"evaluation_{arch}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] Comprehensive report written to: {report_path}")
    return report

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Retinal Disease Classifiers")
    parser.add_argument("--model", type=str, default="efficientnet_b4",
                        choices=["convnext_small", "densenet201", "efficientnet_v2_m", "efficientnet_b4", "resnet50"],
                        help="Neural backbone architecture")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else MODELS_DIR / f"{args.model}.pth"
    evaluate_checkpoint(ckpt_path, args.model, device, args.batch_size)

if __name__ == "__main__":
    main()
