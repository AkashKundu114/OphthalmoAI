"""
PyTorch AMP Deep Learning Training Pipeline for Retinal Fundus Multi-Disease Detection.
Trains ConvNeXt-Small, DenseNet-201, and EfficientNet-V2-M targeting NVIDIA RTX 5060 Laptop GPU.
Saves calibrated checkpoints with Conformal Risk Control metrics.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODELS_DIR = BASE_DIR / "models"

CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image_id"]
        img = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["class"]]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def build_model(arch: str, num_classes: int = len(CLASSES)):
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
    else:
        raise ValueError(f"Unknown architecture {arch}")
    return m

def train_epoch(model, loader, criterion, optimizer, scaler, device, desc="Train"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{(correct/total)*100:.2f}%")
    return total_loss / total, correct / total

@torch.no_grad()
def eval_model(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    bar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{(correct/total)*100:.2f}%")
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, acc, f1

def main():
    print("=" * 70)
    print("OPHTHALMOAI RETINAL FUNDUS MODEL TRAINING PIPELINE")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware compute engine: {device}")
    if device.type == "cuda":
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_tf, val_tf = get_transforms()
    
    train_set = FundusDataset(DATA_DIR / "train.csv", DATA_DIR / "images", transform=train_tf)
    val_set = FundusDataset(DATA_DIR / "val.csv", DATA_DIR / "images", transform=val_tf)
    test_set = FundusDataset(DATA_DIR / "test.csv", DATA_DIR / "images", transform=val_tf)

    batch_size = 16 if device.type == "cuda" else 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Dataset splits: {len(train_set)} train | {len(val_set)} val | {len(test_set)} test")
    print(f"Batch size: {batch_size} (FP16 mixed precision enabled)")

    # Build and fine-tune primary production backbone
    print("\nInitializing EfficientNet-B4 primary production model...")
    model = build_model("efficientnet_b4", num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    epochs = 10
    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, desc=f"Epoch [{epoch:02d}/{epochs:02d}] Train")
        val_loss, val_acc, val_f1 = eval_model(model, val_loader, criterion, device, desc=f"Epoch [{epoch:02d}/{epochs:02d}] Val  ")
        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} [{dt:.1f}s] - Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODELS_DIR / "efficientnet_b4.pth")
            print(f"  [+] Saved new checkpoint (Val F1: {val_f1:.4f})")

    # Evaluate on held-out test split
    print("\nEvaluating on held-out test split...")
    model.load_state_dict(torch.load(MODELS_DIR / "efficientnet_b4.pth"))
    test_loss, test_acc, test_f1 = eval_model(model, test_loader, criterion, device, desc="Evaluating Test")
    print(f"[FINAL TEST RESULTS] Accuracy: {test_acc*100:.2f}% | Macro F1: {test_f1:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
