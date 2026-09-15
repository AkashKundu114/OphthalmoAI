"""
Tests for Prometheus Telemetry & OpenTelemetry-Compatible Distributed Tracing.
"""

import pytest
from backend.metrics import PrometheusMetricsCollector
from backend.tracing import (
    TraceSpan,
    TracingBuffer,
    generate_span_id,
    generate_trace_id,
    trace_pipeline_span,
    tracer_buffer,
)


def test_prometheus_metrics_generation():
    collector = PrometheusMetricsCollector()
    collector.record_inference("resnet50_onnx", 0.014, success=True)
    collector.record_inference("resnet50_fp32", 0.052, success=True)
    collector.record_inference("resnet50_onnx", 0.088, success=False)
    collector.record_domain_shift("reinhard_corrected")
    collector.set_active_tenants(6)

    prom_text = collector.generate_prometheus_text()
    assert "ophthalmoai_inference_requests_total" in prom_text
    assert 'model="resnet50_onnx"' in prom_text
    assert "ophthalmoai_inference_duration_seconds" in prom_text
    assert "ophthalmoai_gpu_memory_bytes" in prom_text
    assert "ophthalmoai_domain_shift_detections_total" in prom_text
    assert "ophthalmoai_active_tenants_total 6" in prom_text


def test_opentelemetry_tracing_context_manager():
    trace_id = generate_trace_id()
    assert len(trace_id) == 32
    span_id = generate_span_id()
    assert len(span_id) == 16

    with trace_pipeline_span("test_inference_pipeline", trace_id=trace_id) as span:
        span.set_attribute("model.name", "resnet50_onnx")
        span.set_attribute("batch_size", 1)

    assert span.duration_ms > 0
    assert span.status == "OK"

    recent = tracer_buffer.get_recent_traces(limit=5)
    assert len(recent) > 0
    latest = recent[0]
    assert latest["name"] == "test_inference_pipeline"
    assert latest["attributes"]["model.name"] == "resnet50_onnx"


def test_tracing_buffer_exception_handling():
    with pytest.raises(ValueError):
        with trace_pipeline_span("failing_pipeline_stage") as span:
            raise ValueError("Test forced pipeline failure")

    recent = tracer_buffer.get_recent_traces(limit=5)
    matching = [t for t in recent if t["name"] == "failing_pipeline_stage"]
    assert len(matching) > 0
    assert matching[0]["status"] == "ERROR"
    assert "Test forced pipeline failure" in matching[0]["error_message"]
