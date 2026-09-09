"""
OphthalmoAI - Train Evidential Meta-Classifier with Asymmetric Clinical-Cost Loss
================================================================================
Trains a DirichletMetaClassifier on concatenated multi-backbone representations
using the Asymmetric Clinical-Cost Loss (AC-HDL).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.evidential import (
    DirichletMetaClassifier,
    AsymmetricClinicalLoss,
    build_clinical_cost_matrix,
    CLASS_NAMES
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

NUM_CLASSES = 12
IN_FEATURES = 36  # 12 logits * 3 base models (ConvNeXt, DenseNet, EfficientNet)
EPOCHS = int(os.environ.get("EPOCHS", "15"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_evidential_classifier(num_samples: int = 1000):
    print(f"=== Training Evidential Meta-Classifier on {DEVICE} ===")
    print(f"Target Classes ({NUM_CLASSES}): {', '.join(CLASS_NAMES)}")

    model = DirichletMetaClassifier(in_features=IN_FEATURES, num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = AsymmetricClinicalLoss(num_classes=NUM_CLASSES, lambda_cost=1.0, lambda_kl=0.01).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Simulated realistic ensemble representations for demonstration/pre-training
    # (can be swapped with cached features from base model passes)
    torch.manual_seed(42)
    synthetic_targets = torch.randint(0, NUM_CLASSES, (num_samples,))
    # Add ground-truth class bias to simulate trained base model representations
    synthetic_inputs = torch.randn(num_samples, IN_FEATURES) * 0.5
    for i in range(num_samples):
        c = synthetic_targets[i].item()
        synthetic_inputs[i, c] += 3.0
        synthetic_inputs[i, c + 12] += 2.8
        synthetic_inputs[i, c + 24] += 2.5

    dataset = torch.utils.data.TensorDataset(synthetic_inputs, synthetic_targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        avg_vacuity = 0.0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            probs, vacuity, alpha = model(batch_x)
            loss = loss_fn(alpha, batch_y, epoch=epoch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(probs, dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            avg_vacuity += vacuity.sum().item()

        scheduler.step()
        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0
        epoch_vac = avg_vacuity / total
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Mean Vacuity: {epoch_vac:.4f}")

    out_path = os.path.join(MODELS_DIR, "evidential_meta_classifier.pth")
    torch.save(model.state_dict(), out_path)
    print(f"--> Saved trained Evidential Meta-Classifier to: {out_path}")
    return model


if __name__ == "__main__":
    train_evidential_classifier()
