
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
