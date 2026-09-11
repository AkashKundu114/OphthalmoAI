"""
Dedicated CPU-Optimized Training Script for ResNet50 on Retinal Fundus Images.
Demonstrates CPU execution, multi-core thread scheduling, and CPU telemetry.
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

def get_model(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="CPU-Only ResNet50 Retinal Fundus Training")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="CPU threads for PyTorch")
    parser.add_argument("--img-size", type=int, default=384, help="Image resolution")
    return parser.parse_args()

def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    print("=" * 70)
    print("CPU-OPTIMIZED RESNET50 RETINAL FUNDUS TRAINING")
    print(f"Device: CPU | Active Threads: {torch.get_num_threads()} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    telemetry = HardwareTelemetry(use_gpu=False, model_name=f"CPU-ResNet50_bs{args.batch_size}")

    train_loader, val_loader, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=0,
        pin_memory=False
    )

    model = get_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_ckpt = MODELS_DIR / f"cpu_resnet50_bs{args.batch_size}.pth"
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        telemetry.start_epoch()
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            if (idx + 1) % 20 == 0 or (idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch}/{args.epochs}] Batch {idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}, Batch Acc: {correct/total*100:.1f}%")

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                v_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        scheduler.step()
        dt = time.time() - t0
        val_acc = v_correct / v_total
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        telemetry.end_epoch(epoch, v_loss / v_total, val_acc * 100)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) - Train Loss: {total_loss/total:.4f} | Val Loss: {v_loss/v_total:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), out_ckpt)
            print(f"  --> Saved CPU checkpoint to {out_ckpt.name} (F1: {val_f1:.4f})")

    # Final test evaluation
    print(f"\n[EVALUATION] Testing CPU model on held-out test split...")
    if out_ckpt.exists():
        model.load_state_dict(torch.load(out_ckpt, map_location=device))
    model.eval()
    t_correct, t_total = 0, 0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            t_correct += (preds == labels).sum().item()
            t_total += imgs.size(0)
            test_preds.extend(preds.numpy())
            test_labels.extend(labels.numpy())

    test_acc = t_correct / t_total
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    print(f"FINAL TEST SET -> Accuracy: {test_acc*100:.2f}% | Macro F1: {test_f1:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
