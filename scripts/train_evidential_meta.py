"""
OphthalmoAI - Train Evidential Meta-Classifier with Asymmetric Clinical-Cost Loss (AC-HDL)
========================================================================================
Trains a DirichletMetaClassifier on concatenated multi-backbone representations
using the Asymmetric Clinical-Cost Loss across the 6 Retinal Fundus classes.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.evidential import (
    DirichletMetaClassifier,
    AsymmetricClinicalLoss,
    CLASS_NAMES
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_CLASSES = len(CLASS_NAMES)
IN_FEATURES = NUM_CLASSES * 3  # 6 logits * 3 base models (ConvNeXt, DenseNet, EfficientNet) = 18

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--samples", type=int, default=2000)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("=" * 70)
    print(f"TRAINING EVIDENTIAL META-CLASSIFIER (AC-HDL)")
    print(f"Device: {device} | Classes: {NUM_CLASSES} | In Features: {IN_FEATURES}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = DirichletMetaClassifier(in_features=IN_FEATURES, num_classes=NUM_CLASSES).to(device)
    loss_fn = AsymmetricClinicalLoss(num_classes=NUM_CLASSES, lambda_cost=1.0, lambda_kl=0.01).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Simulated backbone ensemble representation tensors
    torch.manual_seed(42)
    synthetic_targets = torch.randint(0, NUM_CLASSES, (args.samples,))
    synthetic_inputs = torch.randn(args.samples, IN_FEATURES) * 0.5
    for i in range(args.samples):
        c = synthetic_targets[i].item()
        synthetic_inputs[i, c] += 3.2
        synthetic_inputs[i, c + NUM_CLASSES] += 3.0
        synthetic_inputs[i, c + (NUM_CLASSES * 2)] += 2.8

    dataset = torch.utils.data.TensorDataset(synthetic_inputs, synthetic_targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total, total_vacuity = 0.0, 0, 0, 0.0
        bar = tqdm(loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]", dynamic_ncols=True, leave=False)

        for batch_x, batch_y in bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)

            probs, vacuity, alpha = model(batch_x)
            loss = loss_fn(alpha, batch_y, epoch=epoch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = probs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            total_vacuity += vacuity.sum().item()
            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{(correct/total)*100:.2f}%")

        scheduler.step()
        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100
        epoch_vac = total_vacuity / total

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Avg Vacuity: {epoch_vac:.4f}")

    save_path = MODELS_DIR / "evidential_meta_classifier.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[OK] Evidential Meta-Classifier weights saved to: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
