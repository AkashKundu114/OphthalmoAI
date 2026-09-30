#!/usr/bin/env python3
"""
OphthalmoAI: Standalone ONNX Micro-Benchmarking Suite.

Measures real inference latency percentiles (p50, p95, p99), throughput (QPS),
and speedup vs PyTorch eager inference using time.perf_counter().
If the ONNX model is missing, clearly flags all output and persisted JSON
as SYNTHETIC ESTIMATES to guarantee benchmark integrity.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.onnx_inference import (
    DEFAULT_BENCHMARK_RESULTS_PATH,
    DEFAULT_ONNX_MODEL_PATH,
    SYNTHETIC_WARNING_MESSAGE,
    LatencyBenchmarkSuite,
    ONNXInferenceEngine,
    get_hardware_info,
    save_benchmark_results,
)


def print_banner(title: str, character: str = "=") -> None:
    line = character * 86
    print(f"\n{line}")
    print(f"{title.center(86)}")
    print(f"{line}")


def print_results_table(results: dict) -> None:
    pt = results.get("pytorch_eager", {})
    ort = results.get("onnx_runtime", {})
    fp16 = results.get("onnx_quantized_fp16", {})
    summary = results.get("summary", {})
    is_synth = results.get("is_synthetic", False)

    ort_mode = "SYNTHETIC ESTIMATE" if ort.get("is_synthetic", is_synth) else "MEASURED REAL"
    pt_mode = "BASELINE REFERENCE" if pt.get("is_synthetic", False) else "MEASURED REAL"
    fp16_mode = "SYNTHETIC ESTIMATE" if fp16.get("is_synthetic", True) else "MEASURED REAL"

    header_row = (
        f"| {'Inference Engine':<30} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'p99 (ms)':<9} "
        f"| {'Mean (ms)':<9} | {'QPS':<7} | {'Speedup':<8} | {'Mode':<18} |"
    )
    divider = "+" + "+".join(["-" * 32, "-" * 11, "-" * 11, "-" * 11, "-" * 11, "-" * 9, "-" * 10, "-" * 20]) + "+"

    print(divider)
    print(header_row)
    print(divider)

    pt_row = (
        f"| {'PyTorch Eager (FP32)':<30} | {pt.get('p50_latency_ms', 0.0):>9.2f} "
        f"| {pt.get('p95_latency_ms', 0.0):>9.2f} | {pt.get('p99_latency_ms', 0.0):>9.2f} "
        f"| {pt.get('mean_latency_ms', 0.0):>9.2f} | {pt.get('throughput_qps', 0.0):>7.1f} "
        f"| {'1.00x':<8} | {pt_mode:<18} |"
    )
    print(pt_row)

    ort_speedup = f"{summary.get('onnx_speedup_factor', 1.0):.2f}x"
    ort_row = (
        f"| {'ONNX Runtime (Graph Opt)':<30} | {ort.get('p50_latency_ms', 0.0):>9.2f} "
        f"| {ort.get('p95_latency_ms', 0.0):>9.2f} | {ort.get('p99_latency_ms', 0.0):>9.2f} "
        f"| {ort.get('mean_latency_ms', 0.0):>9.2f} | {ort.get('throughput_qps', 0.0):>7.1f} "
        f"| {ort_speedup:<8} | {ort_mode:<18} |"
    )
    print(ort_row)

    fp16_speedup = f"{summary.get('quantized_speedup_factor', 1.0):.2f}x"
    fp16_row = (
        f"| {'ONNX Runtime (FP16 Est)':<30} | {fp16.get('p50_latency_ms', 0.0):>9.2f} "
        f"| {fp16.get('p95_latency_ms', 0.0):>9.2f} | {fp16.get('p99_latency_ms', 0.0):>9.2f} "
        f"| {fp16.get('mean_latency_ms', 0.0):>9.2f} | {fp16.get('throughput_qps', 0.0):>7.1f} "
        f"| {fp16_speedup:<8} | {fp16_mode:<18} |"
    )
    print(fp16_row)
    print(divider)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OphthalmoAI: ONNX Runtime Low-Latency Benchmarking Suite"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_ONNX_MODEL_PATH),
        help="Path to the target ONNX model artifact (default: models/surface.onnx)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of initial unmeasured warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measured benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_BENCHMARK_RESULTS_PATH),
        help="Destination JSON filepath for benchmark output",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Forces execution in synthetic estimation mode for testing and calibration",
    )

    args = parser.parse_args()

    print_banner("OPHTHALMOAI ONNX RUNTIME MICRO-BENCHMARKING SUITE")

    # Display host hardware metadata
    hw = get_hardware_info()
    print("Host Environment Metadata:")
    print(f"  - Platform:           {hw['platform']}")
    print(f"  - CPU Processor:      {hw['processor']}")
    print(f"  - CPU Cores:          {hw['cpu_count']}")
    print(f"  - Python Version:     {hw['python_version']}")
    print(f"  - PyTorch Version:    {hw.get('torch_version', 'N/A')}")
    print(f"  - ONNX Runtime:       {hw.get('ort_version', 'N/A')}")
    print(f"  - CUDA Available:     {hw.get('cuda_available', False)}")
    if hw.get("gpu_device_name"):
        print(f"  - GPU Device:         {hw['gpu_device_name']}")

    model_path = Path(args.model_path)
    onnx_engine = None

    if not args.force_synthetic and model_path.exists():
        print(f"\n[INFO] Loading ONNX model from: {model_path}")
        onnx_engine = ONNXInferenceEngine(str(model_path))
        if onnx_engine.is_available:
            print(f"[OK] ONNX Session initialized successfully using: {onnx_engine.execution_provider}")
            print(f"[OK] Input Name:  {onnx_engine.input_name}")
            print(f"[OK] Output Name: {onnx_engine.output_name}")
        else:
            print(f"[WARN] Failed to initialize ONNX session ({onnx_engine.init_error}). Falling back to synthetic.")
            onnx_engine = None
    else:
        if args.force_synthetic:
            print("\n[INFO] Force-synthetic flag specified by user.")
        else:
            print(f"\n[INFO] Target ONNX model not found at: {model_path}")

    if onnx_engine and onnx_engine.is_available:
        print_banner("REAL BENCHMARK EXECUTION (MEASURED PERF_COUNTER)", character="=")
        print(f"Configuration: {args.warmup} warmup iterations + {args.iterations} measured iterations")
    else:
        print_banner("SYNTHETIC ESTIMATE MODE (ESTIMATED METRICS)", character="*")
        print(f"  WARNING: {SYNTHETIC_WARNING_MESSAGE}\n")

    # Execute comprehensive benchmark
    results = LatencyBenchmarkSuite.run_comprehensive_benchmark(
        onnx_engine=onnx_engine,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        save_results=True,
        output_path=args.output,
    )

    # In synthetic mode, ensure results are saved to output path explicitly
    if results.get("is_synthetic"):
        save_benchmark_results(results, args.output)

    print("\nBenchmark Summary Results:")
    print_results_table(results)

    print(f"\nResults successfully saved to: {args.output}")
    print(f"Benchmark Mode: {results.get('mode')} (is_synthetic={results.get('is_synthetic')})\n")


if __name__ == "__main__":
    main()
