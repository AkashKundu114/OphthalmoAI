"""
System Health, Telemetry, Benchmarking, Multitenancy & PACS Administration.

Endpoints:
- GET /: Deployment environment status and chat backend metadata
- GET /health, GET /api/health: Lightweight liveness probe
- GET /ready, GET /api/ready: Model readiness probe
- GET /metrics: Real-time Prometheus exposition telemetry
- GET /api/v1/benchmarks/inference, POST /api/v1/benchmarks/run: ONNX vs PyTorch latency benchmarks
- POST /api/v1/cases/similar: Content-Based Medical Image Retrieval (CBMIR)
- GET /api/v1/audit/fairness: Demographic slice parity and equalized odds audit
- GET /api/v1/traces/recent: Distributed tracing buffer
- POST /api/v1/tenants, GET /api/v1/tenants/current: Clinic multi-tenancy onboarding
- POST /admin/pacs-import: Simulated PACS DICOM ingestion
- GET /admin/triage-queue: Pending clinician sign-off queue
- POST /admin/federated/sync: Multi-hospital federated weight aggregation
- POST /admin/synthetic/generate: Synthetic retinal fundus case generator
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth import get_current_user, require_role
from backend.config import DEVICE, GEMINI_MODEL, MONOLITHIC_CLASSES, _ENV
from backend.db import ScanResult, Tenant, User, get_db
from backend.fairness_audit import get_fairness_audit_report
from backend.logging_config import get_logger
from backend.metrics import metrics_collector
from backend.onnx_inference import LatencyBenchmarkSuite, get_cached_benchmark_results
from backend.tenancy import apply_tenant_filter, create_new_tenant, ensure_default_tenant, get_current_tenant_id
from backend.tracing import tracer_buffer
from backend.vector_search import vector_index

logger = get_logger("ophthalmoai.system_routes")
router = APIRouter(tags=["System"])


class SimilarCasesRequest(BaseModel):
    probabilities: Optional[Dict[str, float]] = None
    embedding: Optional[List[float]] = None
    top_k: int = 3


class TenantCreateRequest(BaseModel):
    name: str
    slug: str
    tier: str = "hospital_standard"


@router.get("/")
def read_root():
    import backend.main as bm

    monolithic_model = getattr(bm, "MONOLITHIC_MODEL", None)
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    ollama_url = os.getenv("OLLAMA_URL", "").strip()
    chat_backend = (
        f"Google Gemini ({GEMINI_MODEL})"
        if gemini_key
        else f"Ollama ({os.getenv('OLLAMA_MODEL', 'llama3.2:3b')})"
        if ollama_url
        else "Not Configured"
    )
    return {
        "status": "OphthalmoAI System Ready",
        "device": str(DEVICE),
        "model_loaded": monolithic_model is not None,
        "classes_count": len(MONOLITHIC_CLASSES),
        "chat_backend": chat_backend,
        "version": "2.1.0",
    }


@router.get("/health")
@router.get("/api/health")
def health_check():
    """Liveness probe reporting device status."""
    return {"ok": True, "device": str(DEVICE)}


@router.get("/ready")
@router.get("/api/ready")
def readiness_check(response: Response):
    """Readiness probe reporting model availability."""
    import backend.main as bm

    monolithic_model = getattr(bm, "MONOLITHIC_MODEL", None)
    ready = monolithic_model is not None
    if not ready:
        response.status_code = 503
    return {"ok": ready, "model_loaded": ready}


@router.get("/metrics")
async def get_prometheus_metrics():
    """Exposes real-time Prometheus telemetry in standard text format."""
    return Response(
        content=metrics_collector.generate_prometheus_text(),
        media_type="text/plain; version=0.0.4",
    )


@router.get("/api/v1/benchmarks/inference")
async def get_inference_benchmarks():
    """Returns cached inference latency and throughput benchmarks."""
    import backend.main as bm

    benchmark_fn = getattr(bm, "get_cached_benchmark_results", get_cached_benchmark_results)
    return benchmark_fn()


@router.post("/api/v1/benchmarks/run")
async def run_live_inference_benchmark(iterations: int = 15):
    """Executes a multi-iteration micro-benchmarking sweep across available backbones."""
    import backend.main as bm

    benchmark_suite = getattr(bm, "LatencyBenchmarkSuite", LatencyBenchmarkSuite)
    return benchmark_suite.run_comprehensive_benchmark(iterations=max(3, min(100, iterations)))


@router.post("/api/v1/cases/similar")
async def find_similar_reference_cases(payload: SimilarCasesRequest):
    """Content-Based Medical Image Retrieval (CBMIR) over historical clinical repository."""
    if payload.embedding:
        query_vec = np.array(payload.embedding, dtype=np.float32)
    elif payload.probabilities:
        query_vec = vector_index.extract_embedding_from_probabilities(payload.probabilities)
    else:
        raise HTTPException(status_code=400, detail="Either 'probabilities' or 'embedding' must be provided.")

    results = vector_index.query_similar_cases(query_vec, top_k=payload.top_k)
    return {
        "count": len(results),
        "results": results,
    }


@router.get("/api/v1/audit/fairness")
async def get_fairness_audit(force_refresh: bool = False):
    """Demographic Fairness, Bias Auditing & Slice Disparity Evaluation."""
    report = get_fairness_audit_report(force_refresh=force_refresh)
    return report


@router.get("/api/v1/traces/recent")
async def get_recent_traces(limit: int = 20):
    """OpenTelemetry-compatible distributed tracing endpoint."""
    return {
        "traces": tracer_buffer.get_recent_traces(limit=limit),
    }


@router.post("/api/v1/tenants")
async def create_tenant_endpoint(
    payload: TenantCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Admin-only endpoint to onboard a new hospital/clinic tenant."""
    tenant = create_new_tenant(db, name=payload.name, slug=payload.slug, tier=payload.tier)
    return {
        "id": tenant.id,
        "name": tenant.name,
        "slug": tenant.slug,
        "tier": tenant.tier,
        "created_at": tenant.created_at.isoformat() if tenant.created_at else None,
    }


@router.get("/api/v1/tenants/current")
async def get_current_tenant_info(
    request: Request,
    db: Session = Depends(get_db),
    tenant_id: str = Depends(get_current_tenant_id),
):
    """Returns organizational context and active tier for current tenant."""
    tenant = db.query(Tenant).filter((Tenant.id == tenant_id) | (Tenant.slug == tenant_id)).first()
    if not tenant:
        tenant = ensure_default_tenant(db)
    return {
        "tenant_id": tenant.id,
        "name": tenant.name,
        "slug": tenant.slug,
        "tier": tenant.tier,
        "is_active": tenant.is_active,
    }


@router.post("/admin/pacs-import")
def pacs_import_dicom(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Simulates DICOM ingestion from hospital PACS and syncs to EHR."""
    from backend.pacs_middleware import parse_simulated_dicom

    mock_bytes = b"MOCK-DICOM-IMAGE-PIXELS"
    dicom_meta = parse_simulated_dicom(mock_bytes)

    scan = ScanResult(
        user_id=current_user.id if current_user else None,
        diagnosis="Normal",
        confidence=98.5,
        group_name="Adnexal",
        probabilities={"Normal": 0.985, "Conjunctivitis": 0.015},
        calibrated=True,
        uncertainty=0.02,
        requires_human_review=False,
        icd10_code="Z01.00",
        snomed_code="165070006",
        urgency="none",
        urgency_rank=0,
        dicom_patient_id=dicom_meta["dicom_patient_id"],
        dicom_study_uid=dicom_meta["dicom_study_uid"],
        dicom_series_uid=dicom_meta["dicom_series_uid"],
        spatial_description="No abnormalities noted in optic disc arcade.",
        sign_off_status="signed_off",
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "status": "imported",
        "scan_id": scan.id,
        "dicom_metadata": dicom_meta,
        "message": "Successfully fetched and parsed DICOM file from PACS. Synced to EHR provider.",
    }


@router.get("/admin/triage-queue")
def get_triage_queue(db: Session = Depends(get_db)):
    """Returns list of pending scans awaiting clinician sign-off."""
    pending = db.query(ScanResult).filter(ScanResult.sign_off_status == "pending").all()
    return {
        "queue_length": len(pending),
        "items": [
            {
                "scan_id": s.id,
                "user_id": s.user_id,
                "diagnosis": s.diagnosis,
                "confidence": s.confidence,
                "uncertainty": s.uncertainty,
                "reasons": s.review_reasons,
                "timestamp": s.created_at.isoformat() if s.created_at else None,
            }
            for s in pending
        ],
    }


@router.post("/admin/federated/sync")
def sync_federated_nodes():
    """Triggers federated weight aggregation round across participating clinical nodes."""
    from backend.federated import FederatedServer

    server = FederatedServer()
    result = server.trigger_aggregation_round()
    return result


@router.post("/admin/synthetic/generate")
def generate_synthetic_case(condition: str, severity: str = Form(default="moderate")):
    """Generates synthetic retinal fundus scan and attention map for clinician simulation training."""
    from backend.synthetic_generator import generate_synthetic_fundus

    img_url, heatmap_url = generate_synthetic_fundus(condition, severity)
    return {
        "condition": condition,
        "severity": severity,
        "synthetic_image_url": img_url,
        "synthetic_gradcam_url": heatmap_url,
        "disclaimer": "This image is synthetic and generated for educational and simulator training purposes.",
    }
