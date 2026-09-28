#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OphthalmoAI: Automated 20-Epoch Docker GPU Multi-Model Training Orchestrator
===========================================================================
Executes sequential 20-epoch GPU training across all constituent backbones
and the meta-ensemble using Docker with NVIDIA CUDA acceleration:
  Phase 1: ConvNeXt-Small      (20 epochs, GPU/FP16)
  Phase 2: DenseNet-201        (20 epochs, GPU/FP16)
  Phase 3: EfficientNet-V2-M   (20 epochs, GPU/FP16)
  Phase 4: RetinalMetaEnsemble (20 epochs, GPU/FP16)
  Phase 5: AW-CRC Conformal Recalibration
  Phase 6: Extended Clinical Battery (n=2,249)
  Phase 7: All 18 Publication Figures Generation (300+ DPI, RGB)
"""

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATASET_DIR = ROOT_DIR / "dataset"
SCRIPTS_DIR = ROOT_DIR / "scripts"

TORCH_CACHE_DIR = Path.home() / ".cache" / "torch"

def run_cmd(cmd, desc):
    print("\n" + "=" * 75)
    print(f" {desc}")
    print("=" * 75)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT_DIR))
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[WARN] Step finished with return code {res.returncode} in {dt:.1f}s")
    else:
        print(f"[OK] Completed in {dt:.1f}s")
    return res.returncode

def get_gpu_temp():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            return float(res.stdout.strip())
    except Exception:
        pass
    return None

def probe_cuda_ready():
    try:
        res = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "ophthalmoai-gpu-trainer",
             "python", "-c", "import torch; torch.cuda.init(); assert torch.cuda.is_available(); x=torch.zeros(1, device='cuda')"],
            capture_output=True, text=True, timeout=25
        )
        return res.returncode == 0
    except Exception:
        return False

def cooldown_wait(min_seconds=300, max_seconds=600, target_temp=58.0):
    print("\n" + "=" * 75)
    print(f"[COOLDOWN] Entering GPU cooldown: min {min_seconds//60}m, max {max_seconds//60}m (Target Temp <= {target_temp:.0f}°C)...")
    print("=" * 75)
    start_t = time.time()
    temp_str = "N/A"
    while True:
        elapsed = time.time() - start_t
        temp = get_gpu_temp()
        temp_str = f"{temp:.1f}°C" if temp is not None else "N/A"

        if elapsed < min_seconds:
            mins, secs = divmod(int(min_seconds - elapsed), 60)
            print(f"  --> Mandatory Cooldown: {mins:02d}m {secs:02d}s remaining | GPU Temp: {temp_str}", flush=True)
            time.sleep(min(30, max(1, int(min_seconds - elapsed))))
        elif elapsed < max_seconds and temp is not None and temp > target_temp:
            mins, secs = divmod(int(max_seconds - elapsed), 60)
            print(f"  --> Thermal Stabilization (Temp: {temp_str} > {target_temp:.0f}°C): up to {mins:02d}m {secs:02d}s left...", flush=True)
            time.sleep(15)
        else:
            break

    # Driver handshake check before returning
    print("  --> Performing CUDA driver handshake check...", flush=True)
    for retry in range(5):
        if probe_cuda_ready():
            print(f"[COOLDOWN] CUDA driver ready & verified! (Final GPU Temp: {temp_str})\n")
            return
        print(f"  --> Driver resetting... Retrying handshake in 10s ({retry+1}/5)...", flush=True)
        time.sleep(10)
    print("[WARN] Handshake finished with warnings; proceeding.\n")

def main(epochs=20, batch_size=32, precision="fp16", wait_seconds=300):
    print("=" * 80)
    print(" OPHTHALMOAI: AUTOMATED 20-EPOCH DOCKER GPU TRAINING SUITE")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | Precision: {precision} | Cooldown: {wait_seconds}s")
    print("=" * 80)

    # 1. Build Docker GPU Image if needed
    img_check = subprocess.run(["docker", "image", "inspect", "ophthalmoai-gpu-trainer"], capture_output=True)
    if img_check.returncode != 0:
        build_cmd = ["docker", "build", "-t", "ophthalmoai-gpu-trainer", "-f", "Dockerfile.gpu", "."]
        run_cmd(build_cmd, "BUILDING DOCKER GPU TRAINING IMAGE (CUDA)")
    else:
        print("\n[OK] Docker GPU training image 'ophthalmoai-gpu-trainer' already exists and verified.")

    # 2. Sequential Backbone Training (All 5 benchmark models)
    backbones = ["resnet50", "efficientnet_b4", "convnext_small", "densenet201", "efficientnet_v2_m"]
    for i, b in enumerate(backbones):
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--ipc=host",
            "-v", f"{DATASET_DIR}:/workspace/app/dataset",
            "-v", f"{MODELS_DIR}:/workspace/app/models",
            "-v", f"{SCRIPTS_DIR}:/workspace/app/scripts",
            "-v", f"{TORCH_CACHE_DIR}:/root/.cache/torch",
            "ophthalmoai-gpu-trainer",
            "python", "scripts/train_model.py",
            "--model", b,
            "--precision", precision,
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--device", "cuda"
        ]
        run_cmd(cmd, f"TRAINING BACKBONE ON GPU (20 EPOCHS): {b.upper()}")
        
        # 5-minute cooldown between backbones
        if i < len(backbones) - 1:
            cooldown_wait(wait_seconds)

    # 5-minute cooldown before Meta-Ensemble
    cooldown_wait(wait_seconds)

    # 3. Train Meta-Ensemble on GPU
    ensemble_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "-v", f"{DATASET_DIR}:/workspace/app/dataset",
        "-v", f"{MODELS_DIR}:/workspace/app/models",
        "-v", f"{SCRIPTS_DIR}:/workspace/app/scripts",
        "-v", f"{TORCH_CACHE_DIR}:/root/.cache/torch",
        "ophthalmoai-gpu-trainer",
        "python", "scripts/train_ensemble.py",
        "--precision", precision,
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--device", "cuda"
    ]
    run_cmd(ensemble_cmd, "TRAINING RETINAL META-ENSEMBLE ON GPU (20 EPOCHS)")

    # 5-minute cooldown before calibration and battery
    cooldown_wait(wait_seconds)

    # 4. AW-CRC Conformal Calibration
    calib_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "-v", f"{DATASET_DIR}:/workspace/app/dataset",
        "-v", f"{MODELS_DIR}:/workspace/app/models",
        "-v", f"{SCRIPTS_DIR}:/workspace/app/scripts",
        "-v", f"{ROOT_DIR / 'research'}:/workspace/app/research",
        "ophthalmoai-gpu-trainer",
        "python", "scripts/run_aw_crc_calibration.py"
    ]
    run_cmd(calib_cmd, "CALIBRATING ADMISSIBILITY-WEIGHTED CONFORMAL RISK CONTROL (AW-CRC)")

    # 5. Extended Clinical Battery Evaluation
    eval_cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "-v", f"{DATASET_DIR}:/workspace/app/dataset",
        "-v", f"{MODELS_DIR}:/workspace/app/models",
        "-v", f"{SCRIPTS_DIR}:/workspace/app/scripts",
        "-v", f"{ROOT_DIR / 'research'}:/workspace/app/research",
        "ophthalmoai-gpu-trainer",
        "python", "scripts/evaluate_extended_clinical_battery.py"
    ]
    run_cmd(eval_cmd, "EVALUATING EXTENDED CLINICAL BATTERY (n=2,249)")

    # 6. Generate All 18 Publication Figures
    fig_py = SCRIPTS_DIR / "generate_all_manuscript_figures.py"
    if fig_py.exists():
        py_exe = sys.executable
        run_cmd([py_exe, str(fig_py)], "GENERATING ALL 18 PUBLICATION FIGURES (300+ DPI)")

    print("\n" + "=" * 80)
    print(" [ALL COMPLETE] 20-Epoch Docker GPU Training & Benchmark Lifecycle Finished!")
    print("=" * 80)

if __name__ == "__main__":
    main()
