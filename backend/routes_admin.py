
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import require_role
from .db import AuditLog, ClinicianOverride, ModelVersion, ScanResult, User
from .db_async import get_async_db
from .logging_config import get_logger
from .model_registry import list_versions, set_active

logger = get_logger("routes_admin")

router = APIRouter()

VALID_VERDICTS = {"agree", "disagree", "inconclusive", "insufficient_image_quality"}


class OverrideRequest(BaseModel):
    verdict: str
    corrected_diagnosis: Optional[str] = None
    corrected_icd10: Optional[str] = None
    notes: Optional[str] = None


class ActivateModelRequest(BaseModel):
    version_id: str





@router.post("/scans/{scan_id}/override", status_code=201)
async def create_scan_override(
    scan_id: str,
    payload: OverrideRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(require_role("clinician", "admin")),
):
    if payload.verdict not in VALID_VERDICTS:
        raise HTTPException(
            422,
            detail=(
                f"verdict must be one of {sorted(VALID_VERDICTS)}, got '{payload.verdict}'."
            ),
        )
    if payload.verdict == "disagree" and not payload.corrected_diagnosis:
        raise HTTPException(
            422,
            detail="corrected_diagnosis is required when verdict is 'disagree'.",
        )

    scan_result = await db.execute(select(ScanResult).where(ScanResult.id == scan_id))
    scan = scan_result.scalar_one_or_none()
    if not scan:
        raise HTTPException(404, detail=f"Scan '{scan_id}' not found.")

    existing_result = await db.execute(
        select(ClinicianOverride).where(ClinicianOverride.scan_id == scan_id)
    )
    if existing_result.scalar_one_or_none():


        raise HTTPException(
            409,
            detail=(
                f"Scan '{scan_id}' already has a recorded override. Overrides are "
                "append-only and cannot be edited or replaced (see CLINICAL_SAFETY.md §4)."
            ),
        )

    override = ClinicianOverride(
        scan_id=scan_id,
        clinician_id=current_user.id,
        verdict=payload.verdict,
        corrected_diagnosis=payload.corrected_diagnosis,
        corrected_icd10=payload.corrected_icd10,
        notes=payload.notes,
    )
    db.add(override)
    await db.commit()
    await db.refresh(override)

    logger.info(
        "scan_override.created",
        scan_id=scan_id,
        clinician_id=current_user.id,
        verdict=payload.verdict,
    )

    return {
        "id": override.id,
        "scan_id": override.scan_id,
        "clinician_id": override.clinician_id,
        "verdict": override.verdict,
        "corrected_diagnosis": override.corrected_diagnosis,
        "corrected_icd10": override.corrected_icd10,
        "notes": override.notes,
        "created_at": override.created_at,
    }





@router.get("/admin/audit-logs")
async def list_audit_logs(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(require_role("admin")),
    action: Optional[str] = Query(default=None, description="Filter by action, e.g. 'predict', 'login'"),
    user_id: Optional[str] = Query(default=None),
    success: Optional[bool] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    stmt = select(AuditLog)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)
    if success is not None:
        stmt = stmt.where(AuditLog.success == success)
    stmt = stmt.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "entries": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "action": r.action,
                "resource_id": r.resource_id,
                "resource_type": r.resource_type,
                "ip_address": r.ip_address,
                "success": r.success,
                "error_detail": r.error_detail,
                "metadata": r.metadata_,
                "timestamp": r.timestamp,
            }
            for r in rows
        ],
    }





@router.post("/admin/model-registry/activate")
async def activate_model_version(
    payload: ActivateModelRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(require_role("admin")),
):






    target_result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == payload.version_id)
    )
    target = target_result.scalar_one_or_none()
    if not target:
        raise HTTPException(404, detail=f"ModelVersion id='{payload.version_id}' not found.")

    others_result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.group_key == target.group_key,
            ModelVersion.id != target.id,
        )
    )
    for other in others_result.scalars().all():
        other.active = False
    target.active = True
    await db.commit()
    await db.refresh(target)

    logger.info(
        "model_registry.activated",
        group=target.group_key,
        version=target.version_tag,
        id=target.id,
        activated_by=current_user.id,
    )

    return {
        "id": target.id,
        "group_key": target.group_key,
        "version_tag": target.version_tag,
        "architecture": target.architecture,
        "weights_path": target.weights_path,
        "active": target.active,
        "calibration_temperature": target.calibration_temperature,


        "warning": (
            "This updates the model registry record only. It does NOT hot-swap the model "
            "weights currently loaded in this running process's memory. A process restart "
            "is required for this activation to actually affect inference. Treat this as "
            "staging the next deployment, not an instantaneous production change."
        ),
    }


@router.get("/admin/hitl/discrepancies")
async def get_hitl_discrepancy_analytics(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(require_role("clinician", "admin")),
):
    """
    Computes Human-in-the-Loop (HITL) concordance/discordance analytics between
    clinician overrides and ensemble AI predictions.
    """
    stmt = select(ClinicianOverride, ScanResult).join(
        ScanResult, ClinicianOverride.scan_id == ScanResult.id, isouter=True
    )
    result = await db.execute(stmt)
    pairs = result.all()

    total_reviews = len(pairs)
    agreed = sum(1 for ov, _ in pairs if ov.verdict == "agree")
    disagreed = sum(1 for ov, _ in pairs if ov.verdict == "disagree")
    inconclusive = sum(1 for ov, _ in pairs if ov.verdict in ("inconclusive", "insufficient_image_quality"))

    evaluable = agreed + disagreed
    concordance_rate = round(agreed / evaluable, 4) if evaluable > 0 else 1.0
    discordance_rate = round(disagreed / evaluable, 4) if evaluable > 0 else 0.0

    confusion_map: Dict[str, int] = {}
    for ov, scan in pairs:
        if ov.verdict == "disagree" and scan and ov.corrected_diagnosis:
            key = f"{scan.diagnosis} -> {ov.corrected_diagnosis}"
            confusion_map[key] = confusion_map.get(key, 0) + 1

    confusion_pairs = [
        {"ai_diagnosis": k.split(" -> ")[0], "clinician_diagnosis": k.split(" -> ")[1], "count": v}
        for k, v in confusion_map.items()
    ]

    return {
        "total_reviews": total_reviews,
        "agreed_count": agreed,
        "disagreed_count": disagreed,
        "inconclusive_count": inconclusive,
        "concordance_rate": concordance_rate,
        "discordance_rate": discordance_rate,
        "confusion_pairs": confusion_pairs,
    }


@router.get("/admin/hitl/active-learning")
async def get_active_learning_candidates(
    min_confidence: float = Query(default=0.0, ge=0.0, le=100.0),
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(require_role("clinician", "admin")),
):
    """
    Extracts high-value discrepancy cases (AI vs. Doctor) prioritized for
    active learning retraining candidate pools.
    """
    stmt = (
        select(ClinicianOverride, ScanResult)
        .join(ScanResult, ClinicianOverride.scan_id == ScanResult.id)
        .where(ClinicianOverride.verdict == "disagree")
    )
    result = await db.execute(stmt)
    rows = result.all()

    candidates = []
    for ov, scan in rows:
        conf = float(scan.confidence) if scan and scan.confidence is not None else 0.0
        if conf >= min_confidence:
            candidates.append({
                "scan_id": ov.scan_id,
                "ai_diagnosis": scan.diagnosis if scan else "Unknown",
                "ai_confidence": conf,
                "clinician_corrected_diagnosis": ov.corrected_diagnosis,
                "clinician_notes": ov.notes,
                "created_at": ov.created_at.isoformat() if hasattr(ov.created_at, "isoformat") else str(ov.created_at),
            })

    candidates.sort(key=lambda c: c["ai_confidence"], reverse=True)
    return {
        "candidate_count": len(candidates),
        "min_confidence_filter": min_confidence,
        "candidates": candidates[:limit],
    }

