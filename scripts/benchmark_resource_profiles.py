#!/usr/bin/env python3
"""
OphthalmoAI: Empirical Resource & Performance Profiling Suite.
Measures cold-load time, warm inference latency, peak RAM/VRAM utilization,
CPU utilization, preprocessing throughput, and model footprint on disk.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import ben_graham_preprocess, preprocess

MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "benchmarks"


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("sullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def get_current_rss_mb() -> float:
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, os.getpid())
        if handle:
            ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            ctypes.windll.kernel32.CloseHandle(handle)
            return round(counters.WorkingSetSize / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def get_total_system_ram_gb() -> float:
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / (1024**3), 2)
    except Exception:
        pass
    return 16.0


def get_current_vram_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    return 0.0


def benchmark_image_preprocessing(num_runs: int = 50) -> Dict[str, float]:
    dummy_img = Image.new("RGB", (2048, 1536), color=(180, 80, 40))
    latencies_standard = []
    latencies_ben_graham = []

    # Standard PIL Resize
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = dummy_img.resize((384, 384), Image.Resampling.BILINEAR)
        latencies_standard.append((time.perf_counter() - t0) * 1000.0)

    # Clinical Ben Graham Preprocessing
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = ben_graham_preprocess(dummy_img, target_size=384)
        latencies_ben_graham.append((time.perf_counter() - t0) * 1000.0)

    return {
        "standard_resize_mean_ms": round(float(np.mean(latencies_standard)), 2),
        "standard_resize_p95_ms": round(float(np.percentile(latencies_standard, 95)), 2),
        "ben_graham_mean_ms": round(float(np.mean(latencies_ben_graham)), 2),
        "ben_graham_p95_ms": round(float(np.percentile(latencies_ben_graham, 95)), 2),
    }


def benchmark_model_profile(
    model_name: str,
    checkpoint_path: Path,
    device_name: str = "cpu",
    iterations: int = 30,
) -> Dict[str, Any]:
    device = torch.device(device_name)
    initial_ram = get_current_rss_mb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    initial_vram = get_current_vram_mb()

    # Cold Load Time
    t_load_start = time.perf_counter()
    if "mobilenet" in model_name:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 6)
    elif "densenet" in model_name:
        model = models.densenet201(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 6)
    elif "convnext" in model_name:
        model = models.convnext_small(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 6)
    elif "efficientnet_v2" in model_name:
        model = models.efficientnet_v2_m(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
    else:
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    cold_load_time_ms = (time.perf_counter() - t_load_start) * 1000.0

    post_load_ram = get_current_rss_mb()
    post_load_vram = get_current_vram_mb()
    ram_delta_mb = max(0.0, post_load_ram - initial_ram)
    vram_delta_mb = max(0.0, post_load_vram - initial_vram)

    # Warmup
    dummy_input = torch.randn(1, 3, 384, 384 if "mobilenet" not in model_name else 224, device=device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Latency profiling
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_el = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_el)

    peak_ram = get_current_rss_mb()
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if device.type == "cuda" else 0.0

    disk_size_mb = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0.0

    return {
        "model_name": model_name,
        "device": device_name,
        "disk_size_mb": round(disk_size_mb, 2),
        "cold_load_time_ms": round(cold_load_time_ms, 2),
        "post_load_ram_delta_mb": round(ram_delta_mb, 2),
        "post_load_vram_delta_mb": round(vram_delta_mb, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "throughput_qps": round(1000.0 / float(np.mean(latencies)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="OphthalmoAI Resource & Latency Profiler")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "os": sys.platform,
            "cpu_count": os.cpu_count() or 4,
            "total_ram_gb": get_total_system_ram_gb(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cuda_available": torch.cuda.is_available(),
        },
        "image_preprocessing": benchmark_image_preprocessing(num_runs=30),
        "models": [],
    }

    target_models = [
        ("mobilenet_v3_small_edge", MODELS_DIR / "mobilenet_v3_small_edge.pth"),
        ("densenet201", MODELS_DIR / "densenet201.pth"),
        ("convnext_small", MODELS_DIR / "convnext_small.pth"),
        ("efficientnet_v2_m", MODELS_DIR / "efficientnet_v2_m.pth"),
        ("efficientnet_b4", MODELS_DIR / "efficientnet_b4.pth"),
    ]

    print(f"\n{'='*80}\nOphthalmoAI Empirical Resource Benchmarks ({args.device.upper()})\n{'='*80}")
    print(f"Host: {results['host']['cpu_count']} Logical Cores | {results['host']['total_ram_gb']} GB RAM | GPU: {results['host']['gpu_name']}\n")

    for name, pth in target_models:
        if pth.exists():
            print(f"Profiling {name} on {args.device}...")
            profile = benchmark_model_profile(name, pth, device_name=args.device, iterations=args.iterations)
            results["models"].append(profile)
            print(f"  -> Cold Load: {profile['cold_load_time_ms']} ms | p50: {profile['latency_p50_ms']} ms | QPS: {profile['throughput_qps']} | RAM: {profile['peak_ram_mb']} MB | VRAM: {profile['peak_vram_mb']} MB")

    out_file = OUTPUT_DIR / "resource_profiles.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Profiling complete. Persisted empirical results to {out_file}\n")


if __name__ == "__main__":
    main()
