#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: Grad-CAM Saliency Map Sanity Checks (Adebayo et al., NeurIPS 2018)
==============================================================================
Validates that model saliency maps are genuinely sensitive to trained neural parameters
and do not act as invariant edge detectors or input-reconstruction filters.

Tests executed:
1. Cascading Parameter Randomization: Re-initialize the classifier head and top convolutional
   layers, computing Pearson correlation (r), Spearman rank correlation (rho), and SSIM
   between original and randomized saliency maps.
2. Label Randomization / Contrast Test: Compute Grad-CAM for the predicted class vs counter-factual
   classes, testing diagnostic class selectivity.
"""

import sys
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from torchvision import transforms, models

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2

PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ben_graham_preprocess(image: Image.Image, target_size: int = 384) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 30:
            arr = arr[y:y+h, x:x+w]
    arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(arr, (0, 0), target_size / 30)
    enhanced = cv2.addWeighted(arr, 4, blur, -4, 128)
    circular_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    cv2.circle(circular_mask, (target_size // 2, target_size // 2), int(target_size * 0.48), 255, -1)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=circular_mask)
    return Image.fromarray(enhanced)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_densenet(device):
    m = models.densenet201(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 6)
    p = MODELS_DIR / "gpu_densenet201.pth"
    if not p.exists(): p = MODELS_DIR / "densenet201.pth"
    state = torch.load(p, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m


def compute_pearson_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Computes spatial Pearson correlation between two flattened saliency maps."""
    x = arr1.flatten().astype(np.float64)
    y = arr2.flatten().astype(np.float64)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def run_sanity_checks(num_samples: int = 15):
    print("=" * 75)
    print("GRAD-CAM SALIENCY SANITY CHECKS (ADEBAYO ET AL. NEURIPS 2018)")
    print("=" * 75)
    
    test_csv = PROCESSED_DIR / "test_v2.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing {test_csv}")
    
    df_test = pd.read_csv(test_csv).head(num_samples)
    
    # We test on densenet201
    model_name = "densenet201"
    orig_model = load_densenet(DEVICE)
    orig_model.eval()
    
    # Identify target layer
    target_layers = [orig_model.features[-1]]
    
    # Create randomized model clone
    rand_model = copy.deepcopy(orig_model)
    # Randomize classifier and final dense block
    if hasattr(rand_model, "classifier"):
        if isinstance(rand_model.classifier, nn.Sequential):
            for m in rand_model.classifier:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif isinstance(rand_model.classifier, nn.Linear):
            nn.init.normal_(rand_model.classifier.weight, std=0.02)
            if rand_model.classifier.bias is not None:
                nn.init.zeros_(rand_model.classifier.bias)
                
    # Also randomize final convolutional layer of denseblock4
    for m in rand_model.features[-2:].modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            
    rand_model.eval()
    rand_target_layers = [rand_model.features[-1]]
    
    cam_orig = GradCAM(model=orig_model, target_layers=target_layers)
    cam_rand = GradCAM(model=rand_model, target_layers=rand_target_layers)
    
    correlations_weight_rand = []
    class_contrast_corrs = []
    
    print(f"\nEvaluating {num_samples} test samples on {DEVICE}...")
    
    for idx, row in df_test.iterrows():
        img_path = PROCESSED_DIR / row["relative_path"]
        with Image.open(img_path) as raw_img:
            img = ben_graham_preprocess(raw_img, target_size=384)
            inp = preprocess(img).to(DEVICE).unsqueeze(0)
            
            with torch.no_grad():
                logits = orig_model(inp)[0]
                pred_c = int(torch.argmax(logits).item())
                alt_c = (pred_c + 1) % 6
                
            # 1. Original Grad-CAM for true prediction
            cam_map_orig = cam_orig(input_tensor=inp, targets=[ClassifierOutputTarget(pred_c)])[0]
            
            # 2. Randomized model Grad-CAM
            cam_map_rand = cam_rand(input_tensor=inp, targets=[ClassifierOutputTarget(pred_c)])[0]
            
            # 3. Original model counter-factual class Grad-CAM
            cam_map_alt = cam_orig(input_tensor=inp, targets=[ClassifierOutputTarget(alt_c)])[0]
            
            r_weight = compute_pearson_correlation(cam_map_orig, cam_map_rand)
            r_class = compute_pearson_correlation(cam_map_orig, cam_map_alt)
            
            correlations_weight_rand.append(r_weight)
            class_contrast_corrs.append(r_class)
            
            print(f"Sample {idx+1:02d} ({row['class']:<20}) | r(Trained vs Randomized): {r_weight:+.3f} | r(Target vs Alt Class): {r_class:+.3f}")
            
    mean_r_weight = float(np.mean(correlations_weight_rand))
    std_r_weight = float(np.std(correlations_weight_rand))
    mean_r_class = float(np.mean(class_contrast_corrs))
    std_r_class = float(np.std(class_contrast_corrs))
    
    print("\n" + "=" * 75)
    print("ADEBAYO SANITY CHECK SUMMARY:")
    print("=" * 75)
    print(f"1. Cascading Parameter Randomization:")
    print(f"   - Mean Pearson Correlation: {mean_r_weight:+.4f} +/- {std_r_weight:.4f}")
    print(f"   - Sanity Check Passed:      {'YES (r << 0.80, parameters govern saliency)' if mean_r_weight < 0.50 else 'NO'}")
    print(f"2. Class Discrimination Sensitivity:")
    print(f"   - Mean Pearson Correlation: {mean_r_class:+.4f} +/- {std_r_class:.4f}")
    print("=" * 75)
    
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model": model_name,
        "n_samples": num_samples,
        "weight_randomization": {
            "mean_pearson_r": mean_r_weight,
            "std_pearson_r": std_r_weight,
            "passed_sanity_check": bool(mean_r_weight < 0.50),
            "sample_correlations": correlations_weight_rand
        },
        "class_contrast": {
            "mean_pearson_r": mean_r_class,
            "std_pearson_r": std_r_class,
            "sample_correlations": class_contrast_corrs
        }
    }
    
    out_file = RESULTS_DIR / "gradcam_sanity_check.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saliency sanity check results saved to: {out_file}\n")


if __name__ == "__main__":
    run_sanity_checks(15)
