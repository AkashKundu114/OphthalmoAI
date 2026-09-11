"""
Batch Size 32 Optimized Meta-Ensemble Training Script.
Combines: ConvNeXt-Small + DenseNet-201 + EfficientNet-V2-M
Tuned for 8GB VRAM budgets on NVIDIA RTX 5060 Laptop GPU.
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_dataset import prepare_fundus_dataloaders, CLASSES
from metric_logger import HardwareTelemetry

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_CLASSES = len(CLASSES)

class FundusMetaEnsemble(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        c = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        c.classifier[2] = nn.Linear(c.classifier[2].in_features, num_classes)
        self.convnext = c

        d = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        d.classifier = nn.Linear(d.classifier.in_features, num_classes)
        self.densenet = d

        e = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        e.classifier[1] = nn.Linear(e.classifier[1].in_features, num_classes)
        self.efficientnet = e

        for net in [self.convnext, self.densenet, self.efficientnet]:
            for p in net.parameters():
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
            o1 = self.convnext(x)
            o2 = self.densenet(x)
            o3 = self.efficientnet(x)
        concat = torch.cat([o1, o2, o3], dim=1)
        return self.meta_classifier(concat)

def parse_args():
    parser = argparse.ArgumentParser(description="BS32 Meta-Ensemble Training")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--img-size", type=int, default=384)
    return parser.parse_args()

def main():
    args = parse_args()
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    precision_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else torch.float32)

    print("=" * 70)
    print(f"BATCH SIZE 32 META-ENSEMBLE TRAINING ({args.precision.upper()})")
    print(f"Device: {device} | Batch Size: {batch_size} | Epochs: {args.epochs}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    telemetry = HardwareTelemetry(use_gpu=(device.type == "cuda"), model_name=f"MetaClassifier_{args.precision}_bs32")

    train_loader, val_loader, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=batch_size,
        img_size=args.img_size,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    ensemble = FundusMetaEnsemble(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(ensemble.meta_classifier.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision_dtype == torch.float16))

    out_ckpt = MODELS_DIR / f"meta_classifier_{args.precision}_bs32.pth"
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        telemetry.start_epoch()
        t0 = time.time()
        ensemble.train()
        total_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Train", dynamic_ncols=True, leave=False)
        for imgs, labels in train_bar:
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

            train_bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total*100:.2f}%")

        # Validation
        ensemble.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Val  ", dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                if device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16]:
                    with torch.amp.autocast("cuda", dtype=precision_dtype):
                        logits = ensemble(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = ensemble(imgs)
                    loss = criterion(logits, labels)

                v_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_bar.set_postfix(loss=f"{v_loss/v_total:.4f}", acc=f"{v_correct/v_total*100:.2f}%")

        scheduler.step()
        dt = time.time() - t0
        val_acc = v_correct / v_total
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        telemetry.end_epoch(epoch, v_loss / v_total, val_acc * 100)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) - Train Loss: {total_loss/total:.4f}, Acc: {correct/total*100:.2f}% | Val Loss: {v_loss/v_total:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ensemble.state_dict(), out_ckpt)
            print(f"  --> Saved new best BS32 checkpoint to {out_ckpt.name} (Val F1: {val_f1:.4f})")

    print(f"\nBS32 Meta-Ensemble Training Complete. Best Val F1: {best_f1:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
