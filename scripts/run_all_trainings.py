"""
Master Benchmark & Training Orchestration Suite for Retinal Disease Models.
=============================================================================
Allows systematic execution and comparison across:
- Models: convnext_small, densenet201, efficientnet_v2_m, efficientnet_b4, resnet50
- Precisions: fp32, fp16, bf16
- Batch sizes: 16, 32, 64
- Hardware devices: cpu, cuda

Outputs consolidated benchmark tables in console and json.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
PYTHON_EXE = sys.executable

MODELS = ["efficientnet_b4", "convnext_small", "densenet201", "efficientnet_v2_m", "resnet50"]
PRECISIONS = ["fp16", "bf16", "fp32"]
BATCH_SIZES = [16, 32, 64]
DEVICES = ["cuda", "cpu"]

def run_single(model, precision, batch_size, device, epochs=5, lr=3e-4):
    print(f"\n======================================================================")
    print(f"LAUNCHING BENCHMARK: {model.upper()} | {precision.upper()} | BS={batch_size} | {device.upper()}")
    print(f"======================================================================")

    cmd = [
        PYTHON_EXE,
        str(SCRIPTS_DIR / "train_model.py"),
        "--model", model,
        "--precision", precision,
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--device", device,
        "--lr", str(lr)
    ]

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=False)
    elapsed = time.time() - t0

    return {
        "model": model,
        "precision": precision,
        "batch_size": batch_size,
        "device": device,
        "epochs": epochs,
        "status": "SUCCESS" if res.returncode == 0 else "FAILED",
        "duration_sec": round(elapsed, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="Master Training & Benchmark Orchestrator")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["smoke", "precision_sweep", "batch_sweep", "model_sweep", "interactive", "custom"],
                        help="Benchmark orchestration profile")
    parser.add_argument("--model", type=str, default="efficientnet_b4", choices=MODELS + ["all"])
    parser.add_argument("--precision", type=str, default="fp16", choices=PRECISIONS + ["all"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    results = []

    if args.mode == "smoke":
        print("[SMOKE TEST] Running 1 epoch of ResNet50 (CPU) and EfficientNet-B4 (GPU/FP16)...")
        results.append(run_single("resnet50", "fp32", 16, "cpu", epochs=1))
        results.append(run_single("efficientnet_b4", "fp16", 16, "cuda", epochs=1))

    elif args.mode == "precision_sweep":
        print(f"[PRECISION SWEEP] Comparing FP32 vs FP16 vs BF16 on {args.model}...")
        for prec in ["fp32", "fp16", "bf16"]:
            results.append(run_single(args.model, prec, args.batch_size, args.device, epochs=args.epochs))

    elif args.mode == "batch_sweep":
        print(f"[BATCH SIZE SWEEP] Comparing batch sizes 16, 32, 64 on {args.model} ({args.precision})...")
        for bs in [16, 32, 64]:
            results.append(run_single(args.model, args.precision, bs, args.device, epochs=args.epochs))

    elif args.mode == "model_sweep":
        print(f"[MODEL SWEEP] Comparing architectures at {args.precision} precision...")
        for m in MODELS:
            results.append(run_single(m, args.precision, args.batch_size, args.device, epochs=args.epochs))

    else:
        target_models = MODELS if args.model == "all" else [args.model]
        target_precs = PRECISIONS if args.precision == "all" else [args.precision]
        for m in target_models:
            for p in target_precs:
                results.append(run_single(m, p, args.batch_size, args.device, epochs=args.epochs))

    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("BENCHMARK ORCHESTRATION RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    out_file = ROOT_DIR / "dataset" / "logs" / "benchmark_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to: {out_file}")

if __name__ == "__main__":
    main()
