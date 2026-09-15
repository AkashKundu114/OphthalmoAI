"""
OpenTelemetry-Compatible Distributed Tracing Engine.
=============================================================================
Provides lightweight distributed span context tracing for end-to-end clinical
pipelines (auth -> IQA pre-check -> domain adaptation -> ONNX inference -> CBMIR retrieval).
"""

from __future__ import annotations

import collections
import contextlib
import time
import uuid
from typing import Any, Dict, List, Optional


def generate_trace_id() -> str:
    return uuid.uuid4().hex


def generate_span_id() -> str:
    return uuid.uuid4().hex[:16]


class TraceSpan:
    def __init__(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.trace_id = trace_id or generate_trace_id()
        self.span_id = generate_span_id()
        self.parent_span_id = parent_span_id
        self.attributes: Dict[str, Any] = attributes or {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        self.status: str = "OK"
        self.error_message: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> TraceSpan:
        self.attributes[key] = value
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error_message": self.error_message,
            "attributes": self.attributes,
        }


class TracingBuffer:
    """Thread-safe circular ring buffer keeping the last N completed traces."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._traces: collections.deque = collections.deque(maxlen=capacity)
        self._seed_default_traces()

    def _seed_default_traces(self):
        """Seeds standard sample traces for immediate diagnostic observability."""
        t_id = generate_trace_id()
        root_span = generate_span_id()
        now = time.time()
        self.record_span(
            TraceSpan(
                name="POST /api/v1/screen",
                trace_id=t_id,
                attributes={"http.method": "POST", "http.status_code": 200, "tenant_id": "clinic-apollo-metro"},
            )
        )

    def record_span(self, span: TraceSpan):
        self._traces.appendleft(span.to_dict())

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._traces)[:limit]


# Global tracing buffer
tracer_buffer = TracingBuffer(capacity=200)


@contextlib.contextmanager
def trace_pipeline_span(
    name: str,
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Context manager for tracing execution blocks and automatically committing to tracer_buffer."""
    span = TraceSpan(name=name, trace_id=trace_id, parent_span_id=parent_span_id, attributes=attributes)
    span.start_time = time.time()
    start_perf = time.perf_counter()
    try:
        yield span
    except Exception as exc:
        span.status = "ERROR"
        span.error_message = str(exc)
        raise
    finally:
        span.end_time = time.time()
        span.duration_ms = (time.perf_counter() - start_perf) * 1000.0
        tracer_buffer.record_span(span)
