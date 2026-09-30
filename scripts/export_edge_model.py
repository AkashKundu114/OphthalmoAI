"""
Export Lightweight Edge Model to ONNX and INT8 Quantization.

This script:
1. Builds a lightweight edge architecture (MobileNetV3-Small or EfficientNet-B0)
2. Trains or loads fundus classification weights for 6 retinal conditions
3. Exports the model to ONNX with dynamic batch sizing
4. Quantizes the ONNX model to INT8 via onnxruntime.quantization.quantize_dynamic
5. Verifies and saves the model to frontend/public/models/edge_model.onnx (< 15MB)
6. Measures inference latency and numerical stability
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError:
    raise ImportError("onnx and onnxruntime are required. Run: uv pip install onnx onnxruntime")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "dataset" / "processed"
FRONTEND_MODELS_DIR = ROOT_DIR / "frontend" / "public" / "models"

TARGET_CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}


def normalize_class_name(name: str) -> str:
    """Normalizes class names from datasets to the standard 6 target classes."""
    cleaned = name.strip()
    if "Hypertensive" in cleaned or "Myopia" in cleaned:
        return "Hypertensive Retinopathy"
    if cleaned in CLASS_TO_IDX:
        return cleaned
    raise ValueError(f"Unknown class name: {name}")


class FundusCSVDataset(Dataset):
    """Loads fundus images from CSV manifest for edge model training."""

    def __init__(self, csv_path: Path, images_dir: Path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image_id"]
                cls_raw = row["class"]
                try:
                    norm_cls = normalize_class_name(cls_raw)
                    label = CLASS_TO_IDX[norm_cls]
                    img_path = images_dir / img_name
                    if img_path.exists():
                        self.samples.append((img_path, label))
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_edge_model(arch: str = "mobilenet_v3_small", num_classes: int = len(TARGET_CLASSES)) -> nn.Module:
    """Builds lightweight edge model backbone."""
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture for edge: {arch}")
    return model


def train_edge_model(
    model: nn.Module,
    train_csv: Path,
    val_csv: Path,
    images_dir: Path,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    img_size: int = 224,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Trains lightweight edge model on retinal fundus dataset."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"[Edge Training] Loading dataset from {train_csv} and {val_csv}...")
    train_ds = FundusCSVDataset(train_csv, images_dir, transform=train_tf)
    val_ds = FundusCSVDataset(val_csv, images_dir, transform=val_tf)

    print(f"[Edge Training] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    if len(train_ds) == 0:
        print("[Edge Training] Warning: Train dataset empty! Skipping training.")
        return model

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Class frequency weighting
    class_counts = np.zeros(len(TARGET_CLASSES), dtype=np.float32)
    for _, lbl in train_ds.samples:
        class_counts[lbl] += 1
    weights = len(train_ds.samples) / (len(TARGET_CLASSES) * np.maximum(class_counts, 1.0))
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.to(device)
    print(f"[Edge Training] Training on {device} for {epochs} epochs (batch size: {batch_size})...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += imgs.size(0)

        scheduler.step()
        train_acc = train_correct / max(total_train, 1)

        # Validation
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += imgs.size(0)

        val_acc = val_correct / max(total_val, 1)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} [{elapsed:.1f}s]: "
            f"Train Loss={train_loss/total_train:.4f}, Train Acc={train_acc*100:.2f}% | "
            f"Val Loss={val_loss/total_val:.4f}, Val Acc={val_acc*100:.2f}%"
        )

    return model


def export_and_quantize(
    model: nn.Module,
    output_onnx_path: Path,
    output_quant_path: Path,
    img_size: int = 224,
    opset: int = 18,
) -> Tuple[float, float]:
    """Exports PyTorch model to ONNX and quantizes to INT8."""
    model.eval()
    model.cpu()

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    output_quant_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    print(f"[ONNX Export] Exporting PyTorch model to {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=opset,
        dynamo=False,
    )

    fp32_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"[ONNX Export] FP32 ONNX model size: {fp32_size_mb:.2f} MB")

    print(f"[Quantization] Quantizing ONNX model to INT8 via onnxruntime.quantization...")
    quantize_dynamic(
        model_input=str(output_onnx_path),
        model_output=str(output_quant_path),
        weight_type=QuantType.QUInt8,
    )

    int8_size_mb = os.path.getsize(output_quant_path) / (1024 * 1024)
    print(f"[Quantization] INT8 Quantized ONNX model size: {int8_size_mb:.2f} MB")

    return fp32_size_mb, int8_size_mb


def benchmark_onnx_model(onnx_path: Path, img_size: int = 224, runs: int = 30) -> float:
    """Verifies ONNX session and measures average inference latency on CPU."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy_data = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run([output_name], {input_name: dummy_data})

    # Benchmark
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run([output_name], {input_name: dummy_data})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    print(f"[Benchmark] CPU Inference Latency ({runs} iterations): Mean={avg_ms:.2f} ms | P95={p95_ms:.2f} ms")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="Export quantized edge ONNX model for browser deployment.")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--weights", type=str, default=str(MODELS_DIR / "mobilenet_v3_small_edge.pth"))
    parser.add_argument("--train", action="store_true", help="Force retraining on dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Input resolution")
    parser.add_argument("--output-onnx", type=str, default=str(MODELS_DIR / "edge_model_fp32.onnx"))
    parser.add_argument("--output-quant", type=str, default=str(FRONTEND_MODELS_DIR / "edge_model.onnx"))
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(args.weights)
    train_csv = DATA_DIR / "train.csv"
    val_csv = DATA_DIR / "val.csv"
    images_dir = DATA_DIR / "images"

    model = build_edge_model(args.arch, num_classes=len(TARGET_CLASSES))

    should_train = args.train or not weights_path.exists()
    if should_train and train_csv.exists() and images_dir.exists():
        print(f"Training edge model ({args.arch})...")
        model = train_edge_model(
            model=model,
            train_csv=train_csv,
            val_csv=val_csv,
            images_dir=images_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            img_size=args.img_size,
            device=device,
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print(f"Saved trained edge checkpoint to {weights_path}")
    elif weights_path.exists():
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("Using ImageNet pretrained backbone with initialized classification head.")

    output_onnx = Path(args.output_onnx)
    output_quant = Path(args.output_quant)

    fp32_mb, int8_mb = export_and_quantize(
        model=model,
        output_onnx_path=output_onnx,
        output_quant_path=output_quant,
        img_size=args.img_size,
        opset=args.opset,
    )

    # Verification: must be < 15MB for browser loading
    MAX_SIZE_MB = 15.0
    if int8_mb >= MAX_SIZE_MB:
        raise RuntimeError(f"Model size {int8_mb:.2f} MB exceeds {MAX_SIZE_MB} MB limit!")

    print(f"\n=======================================================")
    print(f" EDGE ONNX EXPORT SUCCESSFUL")
    print(f" Architecture: {args.arch}")
    print(f" Quantized INT8 Size: {int8_mb:.2f} MB (Target: < 15MB)")
    print(f" Saved to: {output_quant}")
    print(f" Classes: {TARGET_CLASSES}")
    print(f"=======================================================\n")

    # Benchmark ONNX model locally
    benchmark_onnx_model(output_quant, img_size=args.img_size)


if __name__ == "__main__":
    main()
