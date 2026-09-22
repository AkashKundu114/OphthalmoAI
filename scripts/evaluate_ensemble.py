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
    def __init__(self, num_classes=NUM_CLASSES, models_dir=MODELS_DIR, device=torch.device("cpu")):
        super().__init__()
        c = models.convnext_small(weights=None)
        c.classifier[2] = nn.Linear(c.classifier[2].in_features, num_classes)
        ckpt_c = models_dir / "convnext_small.pth"
        if ckpt_c.exists():
            c.load_state_dict(torch.load(ckpt_c, map_location=device))
        self.convnext = c.to(device).eval()

        d = models.densenet201(weights=None)
        d.classifier = nn.Linear(d.classifier.in_features, num_classes)
        ckpt_d = models_dir / "densenet201.pth"
        if ckpt_d.exists():
            d.load_state_dict(torch.load(ckpt_d, map_location=device))
        self.densenet = d.to(device).eval()

        e = models.efficientnet_v2_m(weights=None)
        e.classifier[1] = nn.Linear(e.classifier[1].in_features, num_classes)
        ckpt_e = models_dir / "efficientnet_v2_m.pth"
        if ckpt_e.exists():
            e.load_state_dict(torch.load(ckpt_e, map_location=device))
        self.efficientnet = e.to(device).eval()

        self.meta_classifier = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        ).to(device)

    def forward(self, x, mode="soft_voting", temperatures=None):
        with torch.no_grad():
            o1 = self.convnext(x)
            o2 = self.densenet(x)
            o3 = self.efficientnet(x)

        if mode == "soft_voting":
            if temperatures:
                t1 = temperatures.get("convnext_small", 1.0)
                t2 = temperatures.get("densenet201", 1.0)
                t3 = temperatures.get("efficientnet_v2_m", 1.0)
                p1 = F.softmax(o1 / t1, dim=1)
                p2 = F.softmax(o2 / t2, dim=1)
                p3 = F.softmax(o3 / t3, dim=1)
            else:
                p1 = F.softmax(o1, dim=1)
                p2 = F.softmax(o2, dim=1)
                p3 = F.softmax(o3, dim=1)
            return (p1 + p2 + p3) / 3.0

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
    parser.add_argument("--mode", type=str, default="soft_voting", choices=["soft_voting", "meta_classifier"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("=" * 70)
    print("RETINAL ENSEMBLE EVALUATION ON TEST SET")
    print(f"Device: {device} | Mode: {args.mode}")
    print("=" * 70)

    # Load calibration temperatures if available
    calib_path = MODELS_DIR / "calibration.json"
    temperatures = {}
    if calib_path.exists():
        try:
            with open(calib_path, "r") as f:
                temperatures = json.load(f)
            print(f"[OK] Loaded calibration temperatures: {temperatures}")
        except Exception:
            temperatures = {}

    _, _, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size,
        img_size=384,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    ensemble = FundusMetaEnsemble(NUM_CLASSES, models_dir=MODELS_DIR, device=device)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            # Filter if state_dict has full model or just meta_classifier
            if "meta_classifier.0.weight" in state:
                ensemble.load_state_dict(state, strict=False)
                print(f"[OK] Loaded trained meta-classifier weights from {ckpt_path}")
            args.mode = "meta_classifier"

    ensemble.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            if args.mode == "soft_voting":
                probs = ensemble(imgs, mode="soft_voting", temperatures=temperatures)
            else:
                logits = ensemble(imgs, mode="meta_classifier")
                probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    preds_np = probs_np.argmax(axis=1)

    acc = float(accuracy_score(labels_np, preds_np))
    macro_f1 = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels_np, preds_np, average="weighted", zero_division=0))
    ece = compute_ece(probs_np, labels_np)

    # Multi-class AUROC (one-vs-rest)
    try:
        auroc = float(roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro"))
    except Exception:
        auroc = None

    print(f"\n[ENSEMBLE RESULTS]")
    print(f"  Test Accuracy:        {acc * 100:.2f}%")
    print(f"  Macro AUROC:          {auroc:.4f}" if auroc else "  Macro AUROC: N/A")
    print(f"  Macro F1 Score:       {macro_f1:.4f}")
    print(f"  Weighted F1 Score:    {weighted_f1:.4f}")
    print(f"  ECE (Calibration):    {ece:.4f}")

    print(f"\n[CLASSIFICATION REPORT]")
    print(classification_report(labels_np, preds_np, target_names=CLASSES, digits=4, zero_division=0))

    report = {
        "model": "tri_backbone_ensemble",
        "mode": args.mode,
        "test_samples": len(labels_np),
        "accuracy": round(acc, 4),
        "macro_auroc": round(auroc, 4) if auroc else None,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "ece": round(ece, 4),
        "temperatures_applied": temperatures
    }
    mode_json = MODELS_DIR / f"evaluation_meta_ensemble_{args.mode}.json"
    with open(mode_json, "w") as f:
        json.dump(report, f, indent=2)
    out_json = MODELS_DIR / "evaluation_meta_ensemble.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] Saved reports to: {mode_json.name} & {out_json.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
