"""
EfficientNet-V2 Retinal Fundus Classifier Training Script.
Supports:
- Models: EfficientNet-V2-S, EfficientNet-V2-M
- Precision: FP16 (AMP), BF16 (bfloat16), FP32
- Compute: CUDA GPU or CPU
- Batch sizes: 16, 32, 64
- Hardware Telemetry & Test Evaluation
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

def get_efficientnet_v2(variant="s", num_classes=NUM_CLASSES):
    if variant.lower() == "m":
        m = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    else:
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-V2 on Retinal Fundus Dataset")
    parser.add_argument("--variant", type=str, default="s", choices=["s", "m"], help="V2 variant (s or m)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--img-size", type=int, default=384, help="Image resolution")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    precision_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else torch.float32)

    model_name = f"efficientnet_v2_{args.variant}"
    print("=" * 70)
    print(f"{model_name.upper()} RETINAL FUNDUS TRAINING")
    print(f"Device: {device} | Precision: {args.precision.upper()} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    telemetry = HardwareTelemetry(use_gpu=(device.type == "cuda"), model_name=f"{model_name}_{args.precision}_bs{args.batch_size}")

    train_loader, val_loader, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    model = get_efficientnet_v2(args.variant, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision_dtype == torch.float16))

    out_ckpt = MODELS_DIR / f"{model_name}.pth"
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        telemetry.start_epoch()
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Train", dynamic_ncols=True, leave=False)
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16]:
                with torch.amp.autocast("cuda", dtype=precision_dtype):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            train_bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total*100:.2f}%")

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Val  ", dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                if device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16]:
                    with torch.amp.autocast("cuda", dtype=precision_dtype):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                v_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
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
            torch.save(model.state_dict(), out_ckpt)
            print(f"  --> Saved new best checkpoint to {out_ckpt.name} (Val F1: {val_f1:.4f})")

    # Final test evaluation
    print(f"\n[EVALUATION] Evaluating {out_ckpt.name} on test split...")
    if out_ckpt.exists():
        model.load_state_dict(torch.load(out_ckpt, map_location=device))
    model.eval()
    t_correct, t_total = 0, 0
    test_preds, test_labels = [], []
    test_bar = tqdm(test_loader, desc="Evaluating Test", dynamic_ncols=True)
    with torch.no_grad():
        for imgs, labels in test_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            t_correct += (preds == labels).sum().item()
            t_total += imgs.size(0)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = t_correct / t_total
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    print(f"FINAL TEST SET -> Accuracy: {test_acc*100:.2f}% | Macro F1: {test_f1:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
