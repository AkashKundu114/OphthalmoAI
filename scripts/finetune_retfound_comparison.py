import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import scipy.stats as stats
import timm
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_dataset import prepare_fundus_dataloaders, CLASSES
from evaluate_ensemble import FundusMetaEnsemble

# ----------------- Statistical Tests ----------------- #

def compute_delong_pvalue(preds1, preds2, targets):
    """
    Computes p-value for the difference between two ROC AUCs using DeLong's test.
    Fallback to a simple bootstrap if multi-class is tricky, but here we do 1vRest DeLong for macro.
    For simplicity in this comparison, we'll do bootstrap for macro AUC difference.
    """
    n_bootstraps = 1000
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)
    
    auc_diffs = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(targets), len(targets))
        if len(np.unique(targets[indices])) < 2:
            continue
        try:
            auc1 = roc_auc_score(targets[indices], preds1[indices], multi_class='ovr', average='macro')
            auc2 = roc_auc_score(targets[indices], preds2[indices], multi_class='ovr', average='macro')
            auc_diffs.append(auc1 - auc2)
        except:
            continue
    
    if not auc_diffs:
        return 1.0
        
    auc_diffs = np.array(auc_diffs)
    mean_diff = np.mean(auc_diffs)
    std_diff = np.std(auc_diffs)
    
    if std_diff == 0:
        return 1.0
        
    z = mean_diff / std_diff
    p_value = stats.norm.sf(abs(z)) * 2
    return p_value

def compute_mcnemar_pvalue(preds1, preds2, targets):
    """McNemar's test comparing two classifiers on the same test set."""
    correct1 = (preds1 == targets)
    correct2 = (preds2 == targets)
    
    b = int(np.sum(correct1 & ~correct2))  # model1 correct, model2 wrong
    c = int(np.sum(~correct1 & correct2))  # model1 wrong, model2 correct
    
    if b + c == 0:
        return 1.0
    if b + c < 25:
        # Exact binomial test for small discordant counts
        from scipy.stats import binomtest
        return float(binomtest(b, b + c, 0.5).pvalue)
    else:
        # Chi-squared approximation with continuity correction
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        return float(1 - stats.chi2.cdf(stat, df=1))

def bootstrap_metric(targets, preds, metric_fn, n_bootstraps=2000, **kwargs):
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(targets), len(targets))
        if len(np.unique(targets[indices])) < 2:
            continue
        scores.append(metric_fn(targets[indices], preds[indices], **kwargs))
    
    scores = np.array(scores)
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

# ----------------- Focal Loss ----------------- #

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ----------------- ECE ----------------- #

def expected_calibration_error(y_true, y_prob, n_bins=10):
    confidences, predictions = np.max(y_prob, axis=1), np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.any(in_bin):
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_acc - bin_conf) * np.mean(in_bin)
    return ece

# ----------------- Model Setup ----------------- #

def build_retfound(num_classes=6, img_size=224, checkpoint_path=None):
    model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=num_classes)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        state = {k: v for k, v in state.items() if not k.startswith('head.')}
        model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded RETFound pretrained weights from {checkpoint_path}")
    elif checkpoint_path is None:
        print("[WARNING] No RETFound checkpoint provided, using ImageNet weights as fallback.")
        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
        
    return model

# ----------------- Training & Evaluation ----------------- #

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            with autocast('cuda'):
                if images.size(0) == 1:
                    logits = model(torch.cat([images, images], dim=0))[:1]
                else:
                    logits = model(images)
                probs = torch.softmax(logits, dim=1)
                
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    return np.concatenate(all_probs), np.concatenate(all_preds), np.concatenate(all_targets)

def evaluate_ensemble_model(ensemble, loader, device):
    ensemble.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating Ensemble"):
            images, targets = images.to(device), targets.to(device)
            if images.size(0) == 1:
                logits = ensemble(torch.cat([images, images], dim=0), mode="meta_classifier")[:1]
            else:
                logits = ensemble(images, mode="meta_classifier")
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_preds), np.concatenate(all_targets)

def main():
    parser = argparse.ArgumentParser(description="RETFound Fine-tuning Comparison")
    parser.add_argument("--retfound-checkpoint", type=str, default=None, help="Path to RETFound weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Download weights if not provided
    chk_path = args.retfound_checkpoint
    if not chk_path and not args.skip_training:
        try:
            print("Downloading RETFound weights from HuggingFace...")
            chk_path = hf_hub_download(repo_id='iszt/RETFound_mae_natureCFP', filename='RETFound_cfp_weights.pth')
        except Exception as e:
            print(f"Failed to download weights: {e}")

    # 2. Prepare Data (224x224 for ViT)
    train_loader, val_loader, test_loader, cls_to_idx = prepare_fundus_dataloaders(
        batch_size=args.batch_size, img_size=224, num_workers=4
    )
    
    # 3. Model Setup
    model = build_retfound(num_classes=6, checkpoint_path=chk_path)
    model = model.to(device)
    model.set_grad_checkpointing(True)
    
    if args.linear_probe:
        print("Running Linear Probe (Freezing Backbone)...")
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
                
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Calculate class frequencies for Focal Loss alpha (simplified balanced)
    criterion = FocalLoss(gamma=2.0)
    
    scaler = GradScaler('cuda')
    accum_steps = 4
    best_auc = 0
    patience_cnt = 0
    
    save_path = os.path.join(args.output_dir, "retfound_finetuned.pth")

    # 4. Training Loop
    if not args.skip_training:
        print(f"Starting Training for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
                images, targets = images.to(device), targets.to(device)
                
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets) / accum_steps
                    
                scaler.scale(loss).backward()
                
                if (i + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                train_loss += loss.item() * accum_steps
                
            scheduler.step()
            torch.cuda.empty_cache()
            
            # Validation
            probs, _, val_targets = evaluate(model, val_loader, device)
            val_auc = roc_auc_score(val_targets, probs, multi_class='ovr', average='macro')
            
            print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f} - Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), save_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 5:
                    print("Early stopping triggered.")
                    break

    # 5. Evaluation
    print("Evaluating on Test Set...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    
    # Latency test (batch size 4 for Blackwell cuBLAS stability)
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad(), autocast('cuda'):
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latency = (time.time() - start_time) / (50 * 4) * 1000
    
    probs, preds, targets = evaluate(model, test_loader, device)
    
    # Compute Metrics
    acc_mean, acc_lb, acc_ub = bootstrap_metric(targets, preds, accuracy_score)
    auc_mean, auc_lb, auc_ub = bootstrap_metric(targets, probs, lambda t, p: roc_auc_score(t, p, multi_class='ovr', average='macro'))
    
    macro_f1 = f1_score(targets, preds, average='macro')
    ece = expected_calibration_error(targets, probs)
    
    cm = confusion_matrix(targets, preds)
    per_class_sens = np.diag(cm) / np.maximum(np.sum(cm, axis=1), 1)
    per_class_spec = []
    for i in range(6):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        per_class_spec.append(tn / max(tn + fp, 1))
        
    num_params = sum(p.numel() for p in model.parameters())
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    
    # 6. Ensemble Comparison (load OphthalmoAI tri-backbone ensemble for head-to-head)
    print("Gathering OphthalmoAI Ensemble Predictions for head-to-head comparison...")
    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    try:
        # Need a separate test loader at 384x384 for the ensemble (different input size)
        _, _, test_loader_384, _ = prepare_fundus_dataloaders(
            batch_size=args.batch_size, img_size=384, num_workers=2,
            pin_memory=(device.type == 'cuda'),
        )
        ensemble = FundusMetaEnsemble(num_classes=6, models_dir=MODELS_DIR, device=device)
        meta_ckpt = MODELS_DIR / "meta_classifier.pth"
        if meta_ckpt.exists():
            state = torch.load(meta_ckpt, map_location=device, weights_only=False)
            if "meta_classifier.0.weight" in state:
                ensemble.load_state_dict(state, strict=False)
        ensemble.eval()
        ens_probs, ens_preds, ens_targets = evaluate_ensemble_model(ensemble, test_loader_384, device)
        delong_p = compute_delong_pvalue(probs, ens_probs, targets)
        mcnemar_p = compute_mcnemar_pvalue(preds, ens_preds, targets)
        del ensemble
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Could not load/eval ensemble for stats comparison: {e}")
        delong_p, mcnemar_p = None, None

    # 7. Output Report
    results = {
        "model_name": "RETFound ViT-Large",
        "parameters_m": num_params / 1e6,
        "peak_vram_mb": peak_vram,
        "latency_ms": latency,
        "metrics": {
            "top1_acc": {"mean": acc_mean, "ci_lower": acc_lb, "ci_upper": acc_ub},
            "macro_auc": {"mean": auc_mean, "ci_lower": auc_lb, "ci_upper": auc_ub},
            "macro_f1": macro_f1,
            "ece": ece,
            "per_class": {
                CLASSES[i]: {"sensitivity": per_class_sens[i], "specificity": per_class_spec[i]}
                for i in range(6)
            }
        },
        "statistical_comparison_vs_ensemble": {
            "delong_auc_p_value": delong_p,
            "mcnemar_acc_p_value": mcnemar_p
        }
    }
    
    out_file = os.path.join(args.output_dir, "retfound_comparison_report.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*50)
    print("RETFound Comparison Results")
    print("="*50)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
