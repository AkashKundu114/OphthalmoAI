#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: Unified Meta-Ensemble Training & Calibration Engine
================================================================
Orchestrates learned stacking across three heterogeneous vision backbones:
  - ConvNeXt-Small (High-capacity inverted residual convnet)
  - DenseNet-201 (Dense feature reuse & boundary delineation)
  - EfficientNet-V2-M (Compound scaling with fused MBConv layers)

Supports:
  - Mixed Precision: FP16 (AMP), BF16 (Autocast), or FP32
  - Execution Devices: NVIDIA CUDA GPU or CPU
  - Tunable Mini-Batch Sizes: 16, 32, 64
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Safe optional imports for environments with OS DLL control policies
TORCH_AVAILABLE = False
torch = None
nn = None
optim = None
models = None
try:
    import torch as _torch
    import torch.nn as _nn
    import torch.optim as _optim
    from torchvision import models as _models
    torch = _torch
    nn = _nn
    optim = _optim
    models = _models
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score as _acc, f1_score as _f1
    accuracy_score = _acc
    f1_score = _f1
except (ImportError, OSError, Exception):
    import numpy as np

    def accuracy_score(y_true, y_pred) -> float:
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        return float(np.mean(y_t == y_p)) if len(y_t) > 0 else 0.0

    def f1_score(y_true, y_pred, average: str = "macro", zero_division: int = 0) -> float:
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        unique_classes = np.unique(np.concatenate([y_t, y_p]))
        if len(unique_classes) == 0:
            return 0.0
        f1_list = []
        for c in unique_classes:
            tp = np.sum((y_p == c) & (y_t == c))
            fp = np.sum((y_p == c) & (y_t != c))
            fn = np.sum((y_p != c) & (y_t == c))
            denom = 2 * tp + fp + fn
            f1_list.append(float(2 * tp / denom) if denom > 0 else float(zero_division))
        return float(np.mean(f1_list))

sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.path.insert(0, str(ROOT_DIR))

DIAGNOSTIC_CLASSES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy / Pathological Myopia",
]
NUM_CLASSES = len(DIAGNOSTIC_CLASSES)


if TORCH_AVAILABLE and nn is not None:
    class RetinalMetaEnsemble(nn.Module):
        """Heterogeneous multi-backbone feature fusion and meta-classification network."""
        def __init__(
            self,
            convnext_model: nn.Module,
            densenet_model: nn.Module,
            efficientnet_model: nn.Module,
            num_classes: int = NUM_CLASSES
        ):
            super().__init__()
            self.convnext = convnext_model
            self.densenet = densenet_model
            self.efficientnet = efficientnet_model

            # Freeze base feature extractors during meta-layer convergence
            for backbone in [self.convnext, self.densenet, self.efficientnet]:
                for param in backbone.parameters():
                    param.requires_grad = False

            self.meta_classifier = nn.Sequential(
                nn.Linear(num_classes * 3, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                features_convnext = self.convnext(x)
                features_densenet = self.densenet(x)
                features_efficientnet = self.efficientnet(x)
            concatenated_logits = torch.cat([features_convnext, features_densenet, features_efficientnet], dim=1)
            return self.meta_classifier(concatenated_logits)


def build_models(device: Any) -> Optional[Any]:
    """Instantiates and loads fine-tuned weights for the three vision backbones."""
    if not TORCH_AVAILABLE or models is None or nn is None:
        return None

    hardware_prefix = "gpu_" if device.type == "cuda" else "cpu_"

    # 1. ConvNeXt-Small
    convnext_model = models.convnext_small(weights=None)
    convnext_model.classifier[2] = nn.Linear(convnext_model.classifier[2].in_features, NUM_CLASSES)
    ckpt_convnext = MODELS_DIR / f"{hardware_prefix}convnext_small.pth"
    if not ckpt_convnext.exists():
        ckpt_convnext = MODELS_DIR / "convnext_small.pth"
    if ckpt_convnext.exists():
        convnext_model.load_state_dict(torch.load(ckpt_convnext, map_location=device, weights_only=False))
        print(f"[OK] Loaded fine-tuned ConvNeXt-Small from {ckpt_convnext.name}")
    else:
        print(f"Notice: Checkpoint {ckpt_convnext.name} not found; initializing randomly.")

    # 2. DenseNet-201
    densenet_model = models.densenet201(weights=None)
    densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, NUM_CLASSES)
    ckpt_densenet = MODELS_DIR / f"{hardware_prefix}densenet201.pth"
    if not ckpt_densenet.exists():
        ckpt_densenet = MODELS_DIR / "densenet201.pth"
    if ckpt_densenet.exists():
        densenet_model.load_state_dict(torch.load(ckpt_densenet, map_location=device, weights_only=False))
        print(f"[OK] Loaded fine-tuned DenseNet-201 from {ckpt_densenet.name}")
    else:
        print(f"Notice: Checkpoint {ckpt_densenet.name} not found; initializing randomly.")

    # 3. EfficientNet-V2-M
    efficientnet_model = models.efficientnet_v2_m(weights=None)
    efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, NUM_CLASSES)
    ckpt_effnet = MODELS_DIR / f"{hardware_prefix}efficientnet_v2_m.pth"
    if not ckpt_effnet.exists():
        ckpt_effnet = MODELS_DIR / "efficientnet_v2_m.pth"
    if ckpt_effnet.exists():
        efficientnet_model.load_state_dict(torch.load(ckpt_effnet, map_location=device, weights_only=False))
        print(f"[OK] Loaded fine-tuned EfficientNet-V2-M from {ckpt_effnet.name}")
    else:
        print(f"Notice: Checkpoint {ckpt_effnet.name} not found; initializing randomly.")

    ensemble = RetinalMetaEnsemble(convnext_model, densenet_model, efficientnet_model, num_classes=NUM_CLASSES).to(device)
    return ensemble


def train_one_epoch(
    ensemble: Any,
    dataloader: Any,
    criterion: Any,
    optimizer: Any,
    scaler: Any,
    device: Any,
    precision_dtype: Any
) -> Tuple[float, float, int]:
    """Executes a single forward/backward training epoch with division guards."""
    from tqdm import tqdm
    ensemble.train()
    cumulative_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", dynamic_ncols=True, leave=False)
    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        if device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16]:
            with torch.amp.autocast("cuda", dtype=precision_dtype):
                logits = ensemble(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = ensemble(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        cumulative_loss += loss.item() * batch_size
        batch_predictions = logits.argmax(dim=1)
        correct_predictions += (batch_predictions == targets).sum().item()
        total_samples += batch_size

        running_loss = cumulative_loss / max(1, total_samples)
        running_acc = (correct_predictions / max(1, total_samples)) * 100.0
        progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.2f}%")

    average_loss = cumulative_loss / max(1, total_samples)
    accuracy_rate = correct_predictions / max(1, total_samples)
    return average_loss, accuracy_rate, total_samples


def evaluate_one_epoch(
    ensemble: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    precision_dtype: Any
) -> Tuple[float, float, float, int]:
    """Evaluates validation loss, accuracy, and Macro F1 with boundary checks."""
    from tqdm import tqdm
    ensemble.eval()
    cumulative_val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predicted_classes = []
    all_ground_truth = []

    progress_bar = tqdm(dataloader, desc="Validation", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)

            if device.type == "cuda" and precision_dtype in [torch.float16, torch.bfloat16]:
                with torch.amp.autocast("cuda", dtype=precision_dtype):
                    logits = ensemble(images)
                    loss = criterion(logits, targets)
            else:
                logits = ensemble(images)
                loss = criterion(logits, targets)

            cumulative_val_loss += loss.item() * batch_size
            batch_predictions = logits.argmax(dim=1)
            correct_predictions += (batch_predictions == targets).sum().item()
            total_samples += batch_size

            all_predicted_classes.extend(batch_predictions.cpu().numpy())
            all_ground_truth.extend(targets.cpu().numpy())

            running_loss = cumulative_val_loss / max(1, total_samples)
            running_acc = (correct_predictions / max(1, total_samples)) * 100.0
            progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.2f}%")

    average_val_loss = cumulative_val_loss / max(1, total_samples)
    val_accuracy = correct_predictions / max(1, total_samples)
    val_macro_f1 = f1_score(all_ground_truth, all_predicted_classes, average="macro", zero_division=0)
    return average_val_loss, val_accuracy, val_macro_f1, total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Retinal Meta-Ensemble Classifier")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is not available or blocked by system policy.")
        print("Training requires a working PyTorch installation with torchvision.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    precision_dtype = (
        torch.bfloat16 if args.precision == "bf16"
        else (torch.float16 if args.precision == "fp16" else torch.float32)
    )

    print("=" * 70)
    print(f"RETINAL FUNDUS META-ENSEMBLE TRAINING ({args.precision.upper()})")
    print(f"Device: {device} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print("=" * 70)

    try:
        from prepare_dataset import prepare_fundus_dataloaders
        from metric_logger import HardwareTelemetry
    except ImportError as imp_err:
        print(f"[ERROR] Required dataset or telemetry module missing: {imp_err}")
        return

    train_loader, val_loader, test_loader, _ = prepare_fundus_dataloaders(
        batch_size=args.batch_size,
        img_size=384,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )

    ensemble = build_models(device)
    if ensemble is None:
        print("[ERROR] Failed to construct RetinalMetaEnsemble network.")
        return

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(ensemble.meta_classifier.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision_dtype == torch.float16))

    telemetry = HardwareTelemetry(
        use_gpu=(device.type == "cuda"),
        model_name=f"MetaEnsemble_{args.precision}_bs{args.batch_size}"
    )
    best_macro_f1 = 0.0

    tag = f"_{args.precision}" if args.precision != "fp32" else ""
    ckpt_name = f"meta_classifier{tag}.pth" if args.batch_size == 32 else f"meta_classifier{tag}_bs{args.batch_size}.pth"
    save_path = MODELS_DIR / ckpt_name

    for epoch in range(1, args.epochs + 1):
        telemetry.start_epoch()
        epoch_start_time = time.time()

        train_loss, train_acc, train_samples = train_one_epoch(
            ensemble, train_loader, criterion, optimizer, scaler, device, precision_dtype
        )
        val_loss, val_acc, val_f1, val_samples = evaluate_one_epoch(
            ensemble, val_loader, criterion, device, precision_dtype
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        current_learning_rate = optimizer.param_groups[0]["lr"]

        telemetry.end_epoch(
            epoch=epoch,
            loss=train_loss,
            acc=train_acc * 100.0,
            val_loss=val_loss,
            val_acc=val_acc * 100.0,
            val_f1=val_f1,
            samples_count=train_samples,
            lr=current_learning_rate
        )

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100.0:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100.0:.2f}%, F1: {val_f1:.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(ensemble.state_dict(), save_path)
            torch.save(ensemble.state_dict(), MODELS_DIR / "meta_classifier.pth")
            hardware_tag = "gpu" if device.type == "cuda" else "cpu"
            torch.save(ensemble.state_dict(), MODELS_DIR / f"meta_classifier_{hardware_tag}.pth")
            print(f"  --> Saved new best ensemble checkpoint to {ckpt_name} (Val F1: {val_f1:.4f})")

    print(f"\n[OK] Meta-Ensemble Training Completed. Checkpoint saved to: {save_path}")
    print("=" * 70)
    telemetry.close()


if __name__ == "__main__":
    main()
