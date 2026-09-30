import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.onnx_inference import (
    BenchmarkMetrics,
    BenchmarkResult,
    LatencyBenchmarkSuite,
    ONNXInferenceEngine,
    SYNTHETIC_WARNING_MESSAGE,
    get_cached_benchmark_results,
    save_benchmark_results,
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
    assert res["is_synthetic"] is False
    assert res["mode"] == "REAL BENCHMARK"


def test_comprehensive_benchmark_generation_missing_model_is_synthetic():
    """Verifies that missing ONNX model produces explicitly labeled synthetic estimates."""
    results = LatencyBenchmarkSuite.run_comprehensive_benchmark(onnx_engine=None, iterations=5, save_results=False)
    assert "pytorch_eager" in results
    assert "onnx_runtime" in results
    assert "onnx_quantized_fp16" in results
    assert "summary" in results

    # Required: results must NEVER be presented as real benchmarks
    assert results["is_synthetic"] is True
    assert results["synthetic"] is True
    assert results["mode"] == "SYNTHETIC MODE"
    assert SYNTHETIC_WARNING_MESSAGE in results["warning"]

    assert results["onnx_runtime"]["is_synthetic"] is True
    assert results["onnx_runtime"]["mode"] == "SYNTHETIC MODE"
    assert SYNTHETIC_WARNING_MESSAGE in results["onnx_runtime"]["warning"]

    summary = results["summary"]
    assert summary["is_synthetic"] is True
    assert summary["mode"] == "SYNTHETIC MODE"
    assert summary["onnx_speedup_factor"] > 1.0


def test_comprehensive_benchmark_real_model_computes_actual_metrics(tmp_path):
    """Verifies that when ONNX model is available, real time.perf_counter() benchmarks are computed and saved."""
    engine = ONNXInferenceEngine()
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.get_outputs.return_value = [MagicMock(name="output")]

    call_count = 0

    def fake_run(output_names, input_feed):
        nonlocal call_count
        call_count += 1
        return [np.zeros((1, 6), dtype=np.float32)]

    mock_session.run.side_effect = fake_run
    engine.session = mock_session
    engine.input_name = "input"
    engine.output_name = "output"
    engine.model_path = "models/surface.onnx"
    engine.execution_provider = "CPUExecutionProvider"
    engine.is_synthetic = False

    out_file = tmp_path / "test_benchmark_results.json"

    results = LatencyBenchmarkSuite.run_comprehensive_benchmark(
        onnx_engine=engine,
        iterations=10,
        warmup_iterations=3,
        save_results=True,
        output_path=out_file,
    )

    # Real benchmark assertions
    assert results["is_synthetic"] is False
    assert results["synthetic"] is False
    assert results["mode"] == "REAL BENCHMARK"
    assert results["warning"] is None

    ort_res = results["onnx_runtime"]
    assert ort_res["is_synthetic"] is False
    assert ort_res["mode"] == "REAL BENCHMARK"
    assert ort_res["warning"] is None
    assert ort_res["p50_latency_ms"] >= 0.0
    assert ort_res["throughput_qps"] > 0

    assert results["summary"]["is_synthetic"] is False
    assert results["summary"]["mode"] == "REAL BENCHMARK"

    # Verify warmup (3) + test runs (10) were executed
    assert call_count >= 13

    # Verify JSON persistence
    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["is_synthetic"] is False
    assert saved_data["mode"] == "REAL BENCHMARK"
    assert saved_data["hardware_info"]["cpu_count"] > 0
    assert saved_data["model_path"] == "models/surface.onnx"


def test_dataclasses_have_is_synthetic_field():
    """Verifies that benchmark result dataclasses have the is_synthetic boolean field."""
    metrics = BenchmarkMetrics(
        mean_latency_ms=10.0,
        std_latency_ms=1.0,
        p50_latency_ms=9.8,
        p95_latency_ms=11.2,
        p99_latency_ms=12.0,
        min_latency_ms=9.0,
        max_latency_ms=13.0,
        throughput_qps=100.0,
        is_synthetic=False,
    )
    assert hasattr(metrics, "is_synthetic")
    assert metrics.is_synthetic is False
    m_dict = metrics.to_dict()
    assert "is_synthetic" in m_dict
    assert m_dict["is_synthetic"] is False

    report = BenchmarkResult(
        timestamp=123456.0,
        timestamp_iso="2026-09-30T00:00:00Z",
        input_shape=[1, 3, 380, 380],
        iterations=10,
        ort_available=True,
        torch_available=True,
        is_synthetic=True,
        synthetic=True,
        mode="SYNTHETIC MODE",
        warning=SYNTHETIC_WARNING_MESSAGE,
        hardware_info={},
        model_path=None,
        pytorch_eager={},
        onnx_runtime={},
        onnx_quantized_fp16={},
        summary={},
    )
    assert hasattr(report, "is_synthetic")
    assert report.is_synthetic is True
    r_dict = report.to_dict()
    assert r_dict["is_synthetic"] is True
    assert r_dict["synthetic"] is True


def test_get_cached_benchmark_results():
    data1 = get_cached_benchmark_results(force_refresh=True)
    data2 = get_cached_benchmark_results(force_refresh=False)
    assert data1 is data2
    assert "summary" in data1
    assert "is_synthetic" in data1


def test_onnx_engine_missing_file_raises():
    engine = ONNXInferenceEngine(model_path="non_existent_path.onnx")
    assert engine.is_available is False
    assert engine.is_synthetic is True
    with pytest.raises(RuntimeError) as exc_info:
        engine.run(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert "unavailable" in str(exc_info.value).lower()

