#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: CPU Architecture Latency & Throughput Benchmark Suite
==================================================================
Empirically benchmarks inference latency, batch throughput, and memory footprint
across 6 architectures on the 32-thread AMD Ryzen CPU:
1. ResNet-50 (Standard Residual Baseline)
2. EfficientNet-B4 (Balanced Compound Scaling)
3. ConvNeXt-Small (Modern Pure-CNN)
4. DenseNet-201 (Dense Feature Connectivity)
5. EfficientNet-V2-M (Progressive Neural Architecture Search)
6. RetinalMetaEnsemble (Tri-Backbone Meta-Classifier)
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "dataset" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from scripts.train_ensemble import build_models
from scripts.train_model import build_backbone

# Configure multi-threaded CPU execution
num_cores = min(os.cpu_count() or 16, 16)
torch.set_num_threads(num_cores)

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def benchmark_model(name: str, model: nn.Module, img_size: int = 384, warmup: int = 5, reps: int = 20, batch_size: int = 16):
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)
    
    # 1. Single-image Latency (Batch Size = 1)
    dummy_single = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_single)
            
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(dummy_single)
        t_single = (time.perf_counter() - t0) / reps
        single_lat_ms = round(t_single * 1000.0, 2)
        single_fps = round(1.0 / t_single, 2)

    # 2. Batch Throughput (Batch Size = 16)
    dummy_batch = torch.randn(batch_size, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(max(2, warmup // 2)):
            _ = model(dummy_batch)
            
        t0 = time.perf_counter()
        for _ in range(max(5, reps // 2)):
            _ = model(dummy_batch)
        t_batch = (time.perf_counter() - t0) / max(5, reps // 2)
        batch_lat_ms = round(t_batch * 1000.0, 2)
        batch_throughput_fps = round((batch_size) / t_batch, 2)

    params_m = round(count_parameters(model), 2)

    return {
        "architecture": name,
        "parameters_m": params_m,
        "single_latency_ms": single_lat_ms,
        "single_throughput_fps": single_fps,
        "batch_16_latency_ms": batch_lat_ms,
        "batch_16_throughput_fps": batch_throughput_fps
    }

def run_cpu_comparison():
    print("=" * 85)
    print(f" OPHTHALMOAI: ARCHITECTURE LATENCY & THROUGHPUT BENCHMARK (CPU: {num_cores} THREADS)")
    print("=" * 85)
    
    architectures = [
        ("ResNet-50", lambda: build_backbone("resnet50")),
        ("EfficientNet-B4", lambda: build_backbone("efficientnet_b4")),
        ("ConvNeXt-Small", lambda: build_backbone("convnext_small")),
        ("DenseNet-201", lambda: build_backbone("densenet201")),
        ("EfficientNet-V2-M", lambda: build_backbone("efficientnet_v2_m")),
        ("RetinalMetaEnsemble (Tri-Backbone)", lambda: build_models(torch.device("cpu")))
    ]

    results = []
    print(f"{'Architecture':<34} {'Params (M)':<12} {'Latency (ms)':<14} {'Batch Latency':<16} {'Throughput (img/s)'}")
    print("-" * 85)

    for name, builder in architectures:
        try:
            m = builder()
            res = benchmark_model(name, m, warmup=3, reps=10, batch_size=16)
            results.append(res)
            print(f"{res['architecture']:<34} {res['parameters_m']:<12.1f} {res['single_latency_ms']:<14.1f} {res['batch_16_latency_ms']:<16.1f} {res['batch_16_throughput_fps']}")
        except Exception as e:
            print(f"[WARN] Error benchmarking {name}: {e}")

    out_file = MODELS_DIR / "cpu_architecture_benchmark_comparison.json"
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cpu_threads": num_cores,
            "device": "AMD Ryzen 9 (32-thread logical)",
            "results": results
        }, f, indent=2)
    print("=" * 85)
    print(f"[OK] Saved CPU architecture benchmark comparison to: {out_file}\n")

if __name__ == "__main__":
    run_cpu_comparison()
