"""
Universal Training Script for Retinal Fundus Deep Learning Models.
Supports:
- Architectures: convnext_small, densenet201, efficientnet_v2_m, efficientnet_b4, resnet50, ensemble
- Precision: FP32, FP16 (AMP), BF16 (bfloat16)
- Compute: CPU or CUDA GPU
- Batch Sizes: Configurable via CLI arguments or environment variables
- Telemetry: Hardware utilization, memory footprint, time per epoch
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

from prepare_dataset import prepare_fundus_dataloaders, CLASSES, CLASS_TO_IDX
from metric_logger import HardwareTelemetry

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_CLASSES = len(CLASSES)

def build_backbone(arch: str, num_classes: int = NUM_CLASSES):
    if arch == "convnext_small":
        m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif arch == "densenet201":
        m = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif arch == "efficientnet_v2_m":
        m = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "efficientnet_b4":
        m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return m

def train_epoch(model, loader, criterion, optimizer, scaler, device, precision_dtype, desc="Train"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = (device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16] and torch.cuda.is_available())
    bar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)

    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
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
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total*100:.2f}%")

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, precision_dtype, desc="Evaluating"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    use_amp = (device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16] and torch.cuda.is_available())
    bar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)

    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total*100:.2f}%")

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, acc, f1

def parse_args():
    parser = argparse.ArgumentParser(description="Train Retinal Fundus Classifier")
    parser.add_argument("--model", type=str, default="efficientnet_b4",
                        choices=["convnext_small", "densenet201", "efficientnet_v2_m", "efficientnet_b4", "resnet50"],
                        help="Neural backbone architecture")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Floating point precision")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Hardware device")
    parser.add_argument("--img-size", type=int, default=384, help="Image resolution square")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 70)
    print(f"RETINAL FUNDUS MODEL TRAINING: {args.model.upper()}")
    print("=" * 70)

    # Determine device
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("[WARN] CUDA requested but PyTorch environment does not have CUDA enabled.")
            print("       Falling back seamlessly to CPU compute engine.")
            device = torch.device("cpu")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Precision mapping
    if device.type == "cpu":
        precision_dtype = torch.float32
    elif args.precision == "bf16":
        precision_dtype = torch.bfloat16
    elif args.precision == "fp16":
        precision_dtype = torch.float16
    else:
        precision_dtype = torch.float32

    print(f"Compute Engine: {device}")
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Capacity: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Selected Precision: {args.precision.upper() if device.type == 'cuda' else 'FP32 (CPU)'} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    telemetry = HardwareTelemetry(use_gpu=(device.type == "cuda" and torch.cuda.is_available()), model_name=f"{args.model}_{args.precision}_bs{args.batch_size}")

    train_loader, val_loader, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    print(f"\nBuilding {args.model} backbone (Output classes: {NUM_CLASSES})...")
    model = build_backbone(args.model, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision_dtype == torch.float16))

    out_ckpt_name = f"{args.model}_{args.precision}_bs{args.batch_size}.pth" if args.precision != "fp32" else f"{args.model}.pth"
    out_ckpt_path = MODELS_DIR / out_ckpt_name
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        telemetry.start_epoch()
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, precision_dtype, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Train")
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, precision_dtype, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Val  ")
        scheduler.step()
        dt = time.time() - t0

        telemetry.end_epoch(epoch, val_loss, val_acc * 100)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), out_ckpt_path)
            print(f"  --> Saved new best checkpoint to {out_ckpt_name} (Val F1: {val_f1:.4f})")

    # Evaluation on held-out test split
    print(f"\n[EVALUATION] Evaluating best model ({out_ckpt_name}) on test split...")
    if out_ckpt_path.exists():
        model.load_state_dict(torch.load(out_ckpt_path, map_location=device))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device, precision_dtype, desc="Evaluating Test")
    print(f"FINAL TEST SET METRICS -> Accuracy: {test_acc*100:.2f}% | Macro F1: {test_f1:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
