"""
Unified Meta-Ensemble Training Script for Retinal Fundus Classification.
Supports:
- Backbones: ConvNeXt-Small + DenseNet-201 + EfficientNet-V2-M
- Precision: FP16, BF16, or FP32
- Compute: CUDA GPU or CPU
- Batch sizes: 16, 32, 64
- Saves to models/meta_classifier.pth, models/meta_classifier_bf16.pth, or models/meta_classifier_fp16_bs32.pth
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_dataset import prepare_fundus_dataloaders, CLASSES
from metric_logger import HardwareTelemetry

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_CLASSES = len(CLASSES)

class RetinalMetaEnsemble(nn.Module):
    def __init__(self, convnext, densenet, efficientnet, num_classes=NUM_CLASSES):
        super().__init__()
        self.convnext = convnext
        self.densenet = densenet
        self.efficientnet = efficientnet

        # Freeze base feature extractors during meta-layer convergence
        for m in [self.convnext, self.densenet, self.efficientnet]:
            for p in m.parameters():
                p.requires_grad = False

        self.meta_classifier = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            out1 = self.convnext(x)
            out2 = self.densenet(x)
            out3 = self.efficientnet(x)
        concat = torch.cat([out1, out2, out3], dim=1)
        return self.meta_classifier(concat)

def build_models(device):
    c = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    c.classifier[2] = nn.Linear(c.classifier[2].in_features, NUM_CLASSES)

    d = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
    d.classifier = nn.Linear(d.classifier.in_features, NUM_CLASSES)

    e = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    e.classifier[1] = nn.Linear(e.classifier[1].in_features, NUM_CLASSES)

    ensemble = RetinalMetaEnsemble(c, d, e, num_classes=NUM_CLASSES).to(device)
    return ensemble

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    precision_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else torch.float32)

    print("=" * 70)
    print(f"RETINAL FUNDUS META-ENSEMBLE TRAINING ({args.precision.upper()})")
    print(f"Device: {device} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print("=" * 70)

    train_loader, val_loader, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size,
        img_size=384,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    ensemble = build_models(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(ensemble.meta_classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision_dtype == torch.float16))

    telemetry = HardwareTelemetry(use_gpu=(device.type == "cuda"), model_name=f"MetaEnsemble_{args.precision}_bs{args.batch_size}")
    best_f1 = 0.0

    tag = f"_{args.precision}" if args.precision != "fp32" else ""
    ckpt_name = f"meta_classifier{tag}.pth" if args.batch_size == 32 else f"meta_classifier{tag}_bs{args.batch_size}.pth"
    save_path = MODELS_DIR / ckpt_name

    for epoch in range(1, args.epochs + 1):
        telemetry.start_epoch()
        t0 = time.time()
        ensemble.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16]:
                with torch.amp.autocast("cuda", dtype=precision_dtype):
                    logits = ensemble(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = ensemble(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        # Validation
        ensemble.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                if device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16]:
                    with torch.amp.autocast("cuda", dtype=precision_dtype):
                        logits = ensemble(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = ensemble(imgs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        scheduler.step()
        dt = time.time() - t0
        train_acc = correct / total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        telemetry.end_epoch(epoch, val_loss / val_total, val_acc * 100)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) - Train Loss: {total_loss/total:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ensemble.state_dict(), save_path)
            print(f"  --> Saved new best ensemble checkpoint to {ckpt_name} (Val F1: {val_f1:.4f})")

    print(f"\n[OK] Meta-Ensemble Training Completed. Checkpoint saved to: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
