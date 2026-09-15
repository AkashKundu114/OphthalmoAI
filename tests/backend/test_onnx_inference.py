"""
Tests for backend/onnx_inference.py (ONNX Runtime and Micro-Benchmarking Suite).
"""

import numpy as np
import pytest

from backend.onnx_inference import (
    LatencyBenchmarkSuite,
    ONNXInferenceEngine,
    get_cached_benchmark_results,
)


def test_latency_benchmark_suite_basic():
    def mock_fn(x):
        return x * 2.0

    res = LatencyBenchmarkSuite.benchmark_numpy_or_torch(
        mock_fn, np.ones((10, 10)), warmup_runs=2, test_runs=5
    )
    assert "mean_latency_ms" in res
    assert "p50_latency_ms" in res
    assert "p95_latency_ms" in res
    assert "p99_latency_ms" in res
    assert "throughput_qps" in res
    assert res["throughput_qps"] > 0


def test_comprehensive_benchmark_generation():
    results = LatencyBenchmarkSuite.run_comprehensive_benchmark(iterations=5)
    assert "pytorch_eager" in results
    assert "onnx_runtime" in results
    assert "onnx_quantized_fp16" in results
    assert "summary" in results

    summary = results["summary"]
    assert "onnx_speedup_factor" in summary
    assert "quantized_speedup_factor" in summary
    assert summary["onnx_speedup_factor"] > 1.0


def test_get_cached_benchmark_results():
    data1 = get_cached_benchmark_results()
    data2 = get_cached_benchmark_results()
    assert data1 is data2
    assert "summary" in data1


def test_onnx_engine_missing_file_raises():
    engine = ONNXInferenceEngine(model_path="non_existent_path.onnx")
    with pytest.raises(RuntimeError):
        engine.run(np.zeros((1, 3, 224, 224), dtype=np.float32))
