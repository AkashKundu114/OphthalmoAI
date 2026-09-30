"""
Integration tests for the new low-latency ONNX benchmarking and async screening API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-not-for-prod")
    monkeypatch.setenv("FORCE_CPU", "true")

    with TestClient(main_module.app) as c:
        yield c


def test_benchmarks_inference_endpoint(client):
    res = client.get("/api/v1/benchmarks/inference")
    assert res.status_code == 200
    data = res.json()
    assert "pytorch_eager" in data
    assert "onnx_runtime" in data
    assert "summary" in data
    assert data["summary"]["onnx_speedup_factor"] > 1.0

    # Required: verify is_synthetic flag and mode propagation
    assert "is_synthetic" in data
    assert data["is_synthetic"] is True
    assert data["synthetic"] is True
    assert data["mode"] == "SYNTHETIC MODE"
    assert "warning" in data
    assert "ONNX model not found" in data["warning"]
    assert data["summary"]["is_synthetic"] is True


def test_benchmarks_run_endpoint(client):
    res = client.post("/api/v1/benchmarks/run?iterations=5")
    assert res.status_code == 200
    data = res.json()
    assert "pytorch_eager" in data
    assert "summary" in data

    # Required: verify is_synthetic flag in live run endpoint
    assert "is_synthetic" in data
    assert data["is_synthetic"] is True
    assert data["synthetic"] is True
    assert data["mode"] == "SYNTHETIC MODE"
    assert data["summary"]["is_synthetic"] is True


def test_benchmarks_api_propagates_synthetic_and_real_flags(client, monkeypatch):
    """Verifies that API endpoint correctly propagates real vs synthetic flags."""
    # 1. Real benchmark propagation
    def fake_real_benchmark(*args, **kwargs):
        return {
            "is_synthetic": False,
            "synthetic": False,
            "mode": "REAL BENCHMARK",
            "warning": None,
            "summary": {"is_synthetic": False, "mode": "REAL BENCHMARK", "onnx_speedup_factor": 2.45},
            "pytorch_eager": {"p50_latency_ms": 150.0},
            "onnx_runtime": {"p50_latency_ms": 61.2, "is_synthetic": False},
        }

    monkeypatch.setattr(
        main_module.LatencyBenchmarkSuite, "run_comprehensive_benchmark", fake_real_benchmark
    )
    res_real = client.post("/api/v1/benchmarks/run?iterations=5")
    assert res_real.status_code == 200
    d_real = res_real.json()
    assert d_real["is_synthetic"] is False
    assert d_real["synthetic"] is False
    assert d_real["mode"] == "REAL BENCHMARK"
    assert d_real["warning"] is None

    # 2. Synthetic benchmark propagation
    def fake_synthetic_benchmark(*args, **kwargs):
        return {
            "is_synthetic": True,
            "synthetic": True,
            "mode": "SYNTHETIC MODE",
            "warning": "ONNX model not found - using synthetic estimates. These are NOT real benchmarks.",
            "summary": {"is_synthetic": True, "mode": "SYNTHETIC MODE", "onnx_speedup_factor": 2.15},
            "pytorch_eager": {"p50_latency_ms": 181.0},
            "onnx_runtime": {"p50_latency_ms": 84.19, "is_synthetic": True},
        }

    monkeypatch.setattr(
        main_module.LatencyBenchmarkSuite, "run_comprehensive_benchmark", fake_synthetic_benchmark
    )
    res_synth = client.post("/api/v1/benchmarks/run?iterations=5")
    assert res_synth.status_code == 200
    d_synth = res_synth.json()
    assert d_synth["is_synthetic"] is True
    assert d_synth["synthetic"] is True
    assert d_synth["mode"] == "SYNTHETIC MODE"
    assert "ONNX model not found" in d_synth["warning"]


def test_job_status_404(client):
    res = client.get("/api/v1/jobs/job_nonexistent999")
    assert res.status_code == 404


def test_screen_async_validation_rejection(client):
    # Missing form fields and invalid content type
    res = client.post(
        "/api/v1/screen/async",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        data={"pain": "No", "vision": "Normal", "itch": "No"},
    )
    assert res.status_code in (415, 422)
