"""
Ensemble Evaluation Script for Retinal Fundus Classification.
Evaluates the Tri-Backbone Meta-Ensemble (ConvNeXt + DenseNet + EfficientNet-V2)
against the held-out test split (dataset/processed/test.csv).
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_dataset import prepare_fundus_dataloaders, CLASSES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_CLASSES = len(CLASSES)

class FundusMetaEnsemble(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        c = models.convnext_small(weights=None)
        c.classifier[2] = nn.Linear(c.classifier[2].in_features, num_classes)
        self.convnext = c

        d = models.densenet201(weights=None)
        d.classifier = nn.Linear(d.classifier.in_features, num_classes)
        self.densenet = d

        e = models.efficientnet_v2_m(weights=None)
        e.classifier[1] = nn.Linear(e.classifier[1].in_features, num_classes)
        self.efficientnet = e

        self.meta_classifier = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            o1 = self.convnext(x)
            o2 = self.densenet(x)
            o3 = self.efficientnet(x)
        concat = torch.cat([o1, o2, o3], dim=1)
        return self.meta_classifier(concat)

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate Retinal Meta-Ensemble")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to meta_classifier.pth")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else MODELS_DIR / "meta_classifier.pth"

    print("=" * 70)
    print("META-ENSEMBLE EVALUATION ON TEST SET")
    print(f"Device: {device} | Checkpoint: {ckpt_path}")
    print("=" * 70)

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
        return

    _, _, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size,
        img_size=384,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    ensemble = FundusMetaEnsemble(NUM_CLASSES).to(device)
    ensemble.load_state_dict(torch.load(ckpt_path, map_location=device))
    ensemble.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = ensemble(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits_t = torch.cat(all_logits)
    labels_np = torch.cat(all_labels).numpy()
    probs_np = F.softmax(logits_t, dim=1).numpy()
    preds_np = probs_np.argmax(axis=1)

    acc = float(accuracy_score(labels_np, preds_np))
    macro_f1 = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels_np, preds_np, average="weighted", zero_division=0))
    ece = compute_ece(probs_np, labels_np)

    print(f"\n[ENSEMBLE RESULTS]")
    print(f"  Test Accuracy:        {acc * 100:.2f}%")
    print(f"  Macro F1 Score:       {macro_f1:.4f}")
    print(f"  Weighted F1 Score:    {weighted_f1:.4f}")
    print(f"  ECE:                  {ece:.4f}")

    print(f"\n[CLASSIFICATION REPORT]")
    print(classification_report(labels_np, preds_np, target_names=CLASSES, digits=4, zero_division=0))

    report = {
        "model": "meta_ensemble",
        "checkpoint": str(ckpt_path),
        "test_samples": len(labels_np),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "ece": round(ece, 4)
    }
    out_json = MODELS_DIR / "evaluation_meta_ensemble.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] Saved report to: {out_json}")
    print("=" * 70)

if __name__ == "__main__":
    main()
