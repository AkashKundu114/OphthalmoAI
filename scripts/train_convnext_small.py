"""
ConvNeXt-Small Retinal Fundus Classifier Training Script.
Modern pure convolutional architecture featuring 7x7 depthwise convolutions,
inverted bottleneck design, and LayerNorm.
Optimized for 8GB VRAM budgets on NVIDIA RTX 5060 Laptop GPU.
Supports:
- Precision: FP16 (AMP), BF16 (bfloat16), FP32
- Compute: CUDA GPU (Tensor Cores) or CPU
- Batch sizes: 16, 32, 64
- Hardware Telemetry & Conformal Test Evaluation
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

def get_convnext_small(num_classes=NUM_CLASSES):
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Small on Retinal Fundus Dataset")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"], help="Floating-point precision")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (16, 32, 64)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute target")
    parser.add_argument("--img-size", type=int, default=384, help="Image resolution square")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    precision_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else torch.float32)
    if device.type == "cpu":
        precision_dtype = torch.float32

    print("=" * 70)
    print("CONVNEXT-SMALL RETINAL FUNDUS TRAINING")
    print(f"Device: {device} | Precision: {args.precision.upper() if device.type == 'cuda' else 'FP32 (CPU)'} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    telemetry = HardwareTelemetry(use_gpu=(device.type == "cuda" and torch.cuda.is_available()), model_name=f"ConvNeXt-Small_{args.precision}_bs{args.batch_size}")

    train_loader, val_loader, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    model = get_convnext_small(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_scaler = (device.type == "cuda" and precision_dtype == torch.float16 and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if torch.cuda.is_available() else None

    out_ckpt = MODELS_DIR / f"convnext_small_{args.precision}_bs{args.batch_size}.pth"
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

            if device.type == "cuda" and torch.cuda.is_available() and precision_dtype in [torch.float16, torch.bfloat16]:
                with torch.amp.autocast("cuda", dtype=precision_dtype):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
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
                if device.type == "cuda" and torch.cuda.is_available() and precision_dtype in [torch.float16, torch.bfloat16]:
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
        curr_lr = optimizer.param_groups[0]["lr"]
        telemetry.end_epoch(epoch, total_loss / total, correct / total * 100,
                            val_loss=v_loss / v_total, val_acc=val_acc * 100, val_f1=val_f1,
                            samples_count=total, lr=curr_lr)

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
