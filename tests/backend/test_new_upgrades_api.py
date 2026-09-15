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


def test_benchmarks_run_endpoint(client):
    res = client.post("/api/v1/benchmarks/run?iterations=5")
    assert res.status_code == 200
    data = res.json()
    assert "pytorch_eager" in data
    assert "summary" in data


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
