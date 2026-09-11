"""
Temperature Calibration Script for Retinal Fundus Classifiers.
Calibrates Platt / Temperature Scaling on the validation set to produce
calibrated softmax probabilities (minimizing Expected Calibration Error).
Saves calibrated temperatures to models/calibration.json.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_dataset import prepare_fundus_dataloaders, CLASSES
from backend.calibration import TemperatureScaler

MODELS_DIR = project_root / "models"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
NUM_CLASSES = len(CLASSES)

def build_model(arch: str, num_classes: int = NUM_CLASSES):
    if arch == "efficientnet_b4":
        m = models.efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "convnext_small":
        m = models.convnext_small(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif arch == "densenet201":
        m = models.densenet201(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m = models.efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def calibrate_model(model_name: str, device: torch.device):
    ckpt_path = MODELS_DIR / f"{model_name}.pth"
    if not ckpt_path.exists():
        print(f"[SKIP] Model checkpoint not found: {ckpt_path}")
        return 1.0

    print(f"\n==================================================")
    print(f"Calibrating {model_name} on validation split...")
    print(f"==================================================")

    _, val_loader, _, _ = prepare_fundus_dataloaders(
        batch_size=32,
        img_size=384,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    model = build_model(model_name, NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)

    scaler = TemperatureScaler()
    temperature = scaler.fit(logits_tensor, labels_tensor)
    print(f"Optimal Temperature for {model_name}: {temperature:.4f}")
    return temperature

def main():
    parser = argparse.ArgumentParser(description="Calibrate Retinal Disease Models")
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Running Temperature Calibration on {device}...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    registry = {}
    if CALIBRATION_PATH.exists():
        try:
            with open(CALIBRATION_PATH, "r") as f:
                registry = json.load(f)
        except Exception:
            registry = {}

    models_to_run = ["efficientnet_b4", "convnext_small", "densenet201", "resnet50"] if args.model == "all" else [args.model]

    for m in models_to_run:
        T = calibrate_model(m, device)
        registry[m] = round(T, 4)

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\n[SAVED] Updated calibration parameters saved to {CALIBRATION_PATH}:")
    print(json.dumps(registry, indent=2))

if __name__ == "__main__":
    main()
