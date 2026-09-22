"""
Retinal Fundus Dataset Loader.
Supports loading from processed CSV manifests (train.csv, val.csv, test.csv)
and standardized Ben Graham enhanced images for PyTorch training and evaluation.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia"
]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

class RetinalFundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"]
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["class"]]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(img_size: int = 384):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

def prepare_fundus_dataloaders(
    data_dir: str = str(PROCESSED_DIR),
    batch_size: int = 16,
    img_size: int = 384,
    num_workers: int = 2,
    pin_memory: bool = True,
    train_manifest: str = None,
    val_manifest: str = None,
    test_manifest: str = None
):
    data_path = Path(data_dir)
    img_dir = data_path / "images"
    train_tf, val_tf = get_transforms(img_size)

    if train_manifest is None:
        if (data_path / "train_augmented_40k.csv").exists():
            train_manifest = "train_augmented_40k.csv"
        elif (data_path / "train_augmented_3x.csv").exists():
            train_manifest = "train_augmented_3x.csv"
        elif (data_path / "train_patient_clean.csv").exists():
            train_manifest = "train_patient_clean.csv"
        else:
            train_manifest = "train.csv"

    if val_manifest is None:
        val_manifest = "val_patient_clean.csv" if (data_path / "val_patient_clean.csv").exists() else "val.csv"

    if test_manifest is None:
        test_manifest = "test_patient_clean.csv" if (data_path / "test_patient_clean.csv").exists() else "test.csv"

    train_ds = RetinalFundusDataset(data_path / train_manifest, img_dir, transform=train_tf)
    val_ds = RetinalFundusDataset(data_path / val_manifest, img_dir, transform=val_tf)
    test_ds = RetinalFundusDataset(data_path / test_manifest, img_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"DataLoaders prepared successfully:")
    print(f" - Train samples: {len(train_ds)}")
    print(f" - Validation samples: {len(val_ds)}")
    print(f" - Test samples: {len(test_ds)}")
    print(f" - Batch size: {batch_size} | Resolution: {img_size}x{img_size}")

    return train_loader, val_loader, test_loader, CLASS_TO_IDX

# Backward compatibility alias
prepare_dataloaders = prepare_fundus_dataloaders
RetinalDataset = RetinalFundusDataset
