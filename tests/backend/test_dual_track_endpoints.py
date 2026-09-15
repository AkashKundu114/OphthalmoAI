"""
Integration tests for Dual-Track Upgrade Endpoints in FastAPI main application:
- CBMIR Similar Cases Retrieval
- Demographic Fairness Audit
- Prometheus /metrics Exporter
- OpenTelemetry Distributed Tracing
- Multi-Tenant Clinic Management
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_similar_cases_retrieval_endpoint():
    payload = {
        "probabilities": {
            "Glaucoma": 0.88,
            "Normal": 0.12,
        },
        "top_k": 2,
    }
    resp = client.post("/api/v1/cases/similar", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
    top_match = data["results"][0]
    assert "case_id" in top_match
    assert "diagnosis" in top_match
    assert "similarity_score" in top_match
    assert "outcome_12mo" in top_match
    assert top_match["diagnosis"] == "Glaucoma"


def test_fairness_audit_endpoint():
    resp = client.get("/api/v1/audit/fairness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fairness_certified"] is True
    assert "compliance_standards" in data
    assert "cohort_breakdowns" in data
    assert "age_cohorts" in data["cohort_breakdowns"]


def test_prometheus_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "ophthalmoai_inference_requests_total" in resp.text
    assert "ophthalmoai_gpu_memory_bytes" in resp.text


def test_opentelemetry_traces_endpoint():
    resp = client.get("/api/v1/traces/recent?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "traces" in data
    assert isinstance(data["traces"], list)


def test_current_tenant_endpoint():
    resp = client.get("/api/v1/tenants/current")
    assert resp.status_code == 200
    data = resp.json()
    assert "tenant_id" in data
    assert "name" in data
    assert "tier" in data
