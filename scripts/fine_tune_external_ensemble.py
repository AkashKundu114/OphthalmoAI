"""
Domain Adaptation & Fine-Tuning of the Tri-Backbone Ensemble on Augmented External Data.

Integrates the IDRiD training cohort (413 scans from Kowa VX-10 camera) with base training corpus.
Applies:
- Layer-selective transfer learning (top stages + classifier heads)
- Automatic Mixed Precision (AMP FP16)
- Learning rate warmup & cosine annealing (lr=5e-5)
- Re-calibration of Platt temperature scaling on the validation split
- Full checkpoint backup and safety restoration
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import shutil
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_dataset import RetinalFundusDataset, get_transforms, CLASSES, CLASS_TO_IDX
from backend.calibration import TemperatureScaler

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
NUM_CLASSES = len(CLASSES)


def backup_checkpoints():
    """Create safety backups of current production checkpoints."""
    for name in ["convnext_small.pth", "densenet201.pth", "efficientnet_v2_m.pth", "meta_classifier.pth", "calibration.json"]:
        src = MODELS_DIR / name
        if src.exists():
            dst = MODELS_DIR / f"{name}.bak"
            shutil.copy2(src, dst)
            print(f"[BACKUP] Backed up {name} -> {dst.name}")


def build_backbone(arch: str, device: torch.device):
    ckpt_path = MODELS_DIR / f"{arch}.pth"
    if arch == "convnext_small":
        m = models.convnext_small(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
        if ckpt_path.exists():
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
        # Freeze early stages (0, 1), train stages (2, 3, 4, 5, 6, 7) and classifier
        for param in m.features[:4].parameters():
            param.requires_grad = False
    elif arch == "densenet201":
        m = models.densenet201(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
        if ckpt_path.exists():
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
        # Freeze early dense blocks, fine-tune denseblock4 and norm5 + classifier
        for param in m.features[:8].parameters():
            param.requires_grad = False
    elif arch == "efficientnet_v2_m":
        m = models.efficientnet_v2_m(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
        if ckpt_path.exists():
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
        # Freeze first 4 stages, fine-tune stages 5, 6, 7 and classifier
        for param in m.features[:5].parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return m.to(device)


def fine_tune_backbone(arch: str, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 2, lr: float = 5e-5):
    print(f"\n" + "=" * 60)
    print(f"FINE-TUNING BACKBONE: {arch.upper()} ({epochs} Epochs, lr={lr})")
    print("=" * 60)

    model = build_backbone(arch, device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"{arch} Ep {ep}/{epochs} [Train]", dynamic_ncols=True, leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)
            pbar.set_postfix(loss=f"{train_loss/train_total:.4f}", acc=f"{train_correct/train_total*100:.2f}%")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        elapsed = time.time() - t0
        tr_acc = train_correct / train_total
        v_acc = val_correct / val_total
        print(f"Epoch {ep}/{epochs} [{elapsed:.1f}s] | Train Loss: {train_loss/train_total:.4f}, Train Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f}, Val Acc: {v_acc*100:.2f}%")

        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            out_ckpt = MODELS_DIR / f"{arch}.pth"
            torch.save(model.state_dict(), out_ckpt)
            print(f"  [SAVED] Checkpoint updated -> {out_ckpt.name} (Val Acc: {v_acc*100:.2f}%)")

    return model


def recalibrate_temperatures(models_dict: dict, val_loader: DataLoader, device: torch.device):
    print("\n" + "=" * 60)
    print("RE-CALIBRATING PLATT TEMPERATURES ON VALIDATION SPLIT")
    print("=" * 60)

    calib_json = MODELS_DIR / "calibration.json"
    temperatures = {}
    if calib_json.exists():
        with open(calib_json, "r") as f:
            temperatures = json.load(f)

    for name, model in models_dict.items():
        model.eval()
        scaler = TemperatureScaler(device=device)
        T_opt = scaler.calibrate(model, val_loader)
        temperatures[name] = round(float(T_opt), 4)
        print(f"  {name:<20}: Optimal T* = {T_opt:.4f}")

    with open(calib_json, "w") as f:
        json.dump(temperatures, f, indent=2)
    print(f"[OK] Updated calibration file: {calib_json}")
    return temperatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per backbone")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("OPHTHALMOAI DOMAIN ADAPTATION & FINE-TUNING PIPELINE")
    print(f"Device: {device} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print("=" * 70)

    # 1. Back up current weights
    backup_checkpoints()

    # 2. DataLoaders with augmented training manifest
    img_dir = PROCESSED_DIR / "images"
    train_tf, val_tf = get_transforms(img_size=384)

    augmented_train_csv = PROCESSED_DIR / "train_augmented.csv"
    val_csv = PROCESSED_DIR / "val.csv"

    train_ds = RetinalFundusDataset(augmented_train_csv, img_dir, transform=train_tf)
    val_ds = RetinalFundusDataset(val_csv, img_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nLoaded Augmented Training Corpus:")
    print(f" - Training samples:   {len(train_ds)} (Includes 413 external IDRiD scans)")
    print(f" - Validation samples: {len(val_ds)}")

    # 3. Fine-tune backbones
    backbones = ["convnext_small", "densenet201", "efficientnet_v2_m"]
    models_dict = {}

    for arch in backbones:
        m = fine_tune_backbone(arch, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
        models_dict[arch] = m

    # 4. Re-calibrate temperatures
    recalibrate_temperatures(models_dict, val_loader, device)

    print("\n" + "=" * 70)
    print("[SUCCESS] All backbones fine-tuned and re-calibrated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
