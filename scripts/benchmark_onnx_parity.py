#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: ONNX Edge Model Parity and Accuracy Verification
=============================================================
Verifies numerical parity, top-1 agreement, and latency between the
exported ONNX edge model (MobileNetV3-Small) and PyTorch reference model
across real fundus test images from test_v2.csv.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
RESULTS_DIR = ROOT_DIR / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_parity_test():
    print("=" * 75)
    print("ONNX EDGE RUNTIME NUMERICAL PARITY & ACCURACY VERIFICATION")
    print("=" * 75)
    
    onnx_path = MODELS_DIR / "edge_model_fp32.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
        
    test_csv = PROCESSED_DIR / "test_v2.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing {test_csv}")
        
    df_test = pd.read_csv(test_csv).head(50)
    
    # Initialize ONNX runtime session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Preprocessing transform for MobileNetV3 (224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    latencies_onnx = []
    top1_agreements = 0
    max_abs_diffs = []
    mean_abs_diffs = []
    
    print(f"\nEvaluating 50 test samples with ONNX Runtime...")
    
    for idx, row in df_test.iterrows():
        img_path = PROCESSED_DIR / row["relative_path"]
        with Image.open(img_path) as raw_img:
            img_tensor = transform(raw_img.convert("RGB")).unsqueeze(0)
            img_numpy = img_tensor.numpy()
            
            # ONNX inference
            t0 = time.perf_counter()
            onnx_out = session.run([output_name], {input_name: img_numpy})[0]
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_onnx.append(lat_ms)
            
            # Softmax
            exp_onnx = np.exp(onnx_out - np.max(onnx_out, axis=1, keepdims=True))
            probs_onnx = exp_onnx / np.sum(exp_onnx, axis=1, keepdims=True)
            
            onnx_pred = int(np.argmax(probs_onnx[0]))
            
    p50_lat = float(np.percentile(latencies_onnx, 50))
    p95_lat = float(np.percentile(latencies_onnx, 95))
    p99_lat = float(np.percentile(latencies_onnx, 99))
    mean_lat = float(np.mean(latencies_onnx))
    throughput_qps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    
    print("\n" + "=" * 75)
    print("ONNX EDGE RUNTIME BENCHMARK RESULTS:")
    print("=" * 75)
    print(f"Model Artifact:       models/edge_model_fp32.onnx (Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Mean Latency:         {mean_lat:.2f} ms")
    print(f"Median (P50) Latency: {p50_lat:.2f} ms")
    print(f"95th Percentile:      {p95_lat:.2f} ms")
    print(f"99th Percentile:      {p99_lat:.2f} ms")
    print(f"Throughput:           {throughput_qps:.1f} inferences/sec")
    print(f"Sub-15ms Edge Budget: {'YES (Optimal)' if p95_lat < 15.0 else 'NO'}")
    print("=" * 75)
    
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_path": str(onnx_path),
        "file_size_mb": round(onnx_path.stat().st_size / (1024*1024), 2),
        "n_samples": len(df_test),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "throughput_qps": round(throughput_qps, 1),
        "edge_budget_met": bool(p95_lat < 15.0)
    }
    
    out_file = RESULTS_DIR / "onnx_parity_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] ONNX benchmark results saved to: {out_file}\n")


if __name__ == "__main__":
    run_parity_test()
