"""Clinical Workflows & EHR Integration Route Handlers."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.audit import log_event
from backend.auth import get_current_user, hash_password, require_role
from backend.clinical_codes import get_clinical_code
from backend.db import ClinicianOverride, PatientAppointment, ScanResult, User, get_db
from backend.fhir import export_to_fhir_diagnostic_report
from backend.logging_config import get_logger
from backend.medical_data import MEDICAL_INFO
from backend.mongodb_client import mongo_store
from backend.pacs_middleware import sync_to_ehr_middleware
from backend.routes_admin import OverrideRequest

logger = get_logger("ophthalmoai.clinical_routes")
router = APIRouter(tags=["Clinical"])


class MultimodalRiskRequest(BaseModel):
    hba1c: float
    systolic_bp: int
    diastolic_bp: int
    age: int
    is_smoker: bool
    diagnosis: str
    confidence: float


class AppointmentCreate(BaseModel):
    clinic_name: str
    appointment_time: str
    purpose: str


@router.get("/conditions")
@router.get("/api/conditions")
def get_conditions():
    """Returns clinical disease library and educational guidance entries."""
    return {
        "conditions": [
            {
                "key": k,
                "name": v.get("name", k),
                "group": v.get("group", ""),
                "color": v.get("color", "#64748B"),
                "description": v.get("description", ""),
                "symptoms": v.get("symptoms", []),
                "treatment": v.get("treatment", []),
                "precautions": v.get("precautions", []),
                "severity": v.get("severity", ""),
                "advice": v.get("advice", ""),
            }
            for k, v in MEDICAL_INFO.items()
        ]
    }


@router.get("/fhir/export/{scan_id}")
@router.get("/api/fhir/export/{scan_id}")
async def get_fhir_report(scan_id: str, db: Session = Depends(get_db)):
    """Exports clinical scan finding as an HL7 FHIR R4 DiagnosticReport resource."""
    scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not scan:
        raise HTTPException(404, detail=f"Scan record '{scan_id}' not found.")

    scan_dict = {
        "scan_id": scan.id,
        "user_id": scan.user_id,
        "timestamp": scan.timestamp.isoformat() if scan.timestamp else None,
        "diagnosis": scan.diagnosis,
        "confidence": scan.confidence,
        "icd10_code": scan.icd10_code,
        "snomed_code": scan.snomed_code,
        "urgency": scan.urgency,
        "uncertainty": scan.uncertainty,
        "requires_human_review": scan.requires_human_review,
        "iqa_acceptable": scan.iqa_acceptable,
        "escalation_message": f"Clinical Triage Urgency: {scan.urgency.upper()}.",
    }
    return export_to_fhir_diagnostic_report(scan_dict)


@router.get("/clinician/cases")
@router.get("/api/clinician/cases")
async def list_clinician_cases(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("clinician", "admin")),
):
    """Retrieves scans flagged by conformal triage or evidential OOD as requiring human sign-off."""
    scans = (
        db.query(ScanResult)
        .filter(ScanResult.requires_human_review == True)
        .order_by(ScanResult.timestamp.desc())
        .limit(50)
        .all()
    )
    return {
        "cases": [
            {
                "scan_id": s.id,
                "user_id": s.user_id,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "diagnosis": s.diagnosis,
                "confidence": s.confidence,
                "group_name": s.group_name,
                "urgency": s.urgency,
                "uncertainty": s.uncertainty,
                "review_reasons": s.review_reasons,
                "icd10_code": s.icd10_code,
            }
            for s in scans
        ]
    }


@router.post("/clinician/override/{scan_id}")
@router.post("/api/clinician/override/{scan_id}")
async def submit_clinician_override(
    scan_id: str,
    override: OverrideRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("clinician", "admin")),
):
    """Submits human clinician review, sign-off or diagnostic correction."""
    scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not scan:
        raise HTTPException(404, detail="Scan record not found.")

    entry = ClinicianOverride(
        scan_id=scan.id,
        clinician_id=current_user.id,
        verdict=override.verdict,
        corrected_diagnosis=override.corrected_diagnosis,
        corrected_icd10=override.corrected_icd10,
        notes=override.notes,
    )
    db.add(entry)
    scan.requires_human_review = False
    db.commit()

    log_event(
        db,
        "clinician_override",
        success=True,
        user_id=current_user.id,
        resource_id=scan.id,
        resource_type="scan_result",
        metadata={"verdict": override.verdict, "corrected_diagnosis": override.corrected_diagnosis},
    )
    return {"status": "success", "message": "Clinician sign-off recorded successfully."}


@router.get("/patient/history/{patient_id}")
async def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves all historical scans recorded for a specific patient identifier."""
    try:
        scans = (
            db.query(ScanResult)
            .filter(ScanResult.user_id == patient_id)
            .order_by(ScanResult.timestamp.desc())
            .all()
        )
    except Exception:
        scans = []

    return {
        "patient_id": patient_id,
        "scans": [
            {
                "scan_id": s.id,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "diagnosis": s.diagnosis,
                "confidence": s.confidence,
                "group_name": s.group_name,
                "urgency": s.urgency,
                "icd10_code": s.icd10_code,
            }
            for s in scans
        ],
    }


@router.post("/predict/multimodal")
def predict_multimodal_risk(req: MultimodalRiskRequest):
    """Heuristic multivariable cardiovascular and ophthalmic disease progression calculator."""
    bp_ratio = req.systolic_bp / 120.0
    hba1c_risk = max(1.0, req.hba1c / 5.7)

    cardio_risk_score = 10.0 * bp_ratio * (1.5 if req.is_smoker else 1.0) * (req.age / 40.0)
    cardio_risk = "High" if cardio_risk_score > 15 else ("Moderate" if cardio_risk_score > 8 else "Low")

    ophthalmic_risk_score = (req.confidence / 100.0) * hba1c_risk * (1.3 if req.age > 60 else 1.0)
    progression_risk = "High" if ophthalmic_risk_score > 1.4 else ("Moderate" if ophthalmic_risk_score > 0.8 else "Low")

    return {
        "cardiovascular_risk_level": cardio_risk,
        "cardiovascular_risk_score": round(cardio_risk_score, 1),
        "ophthalmic_progression_risk_level": progression_risk,
        "ophthalmic_progression_risk_score": round(ophthalmic_risk_score, 2),
        "clinical_guidance": (
            "Patient has elevated cardiovascular threat factors. Immediate endocrine consultation advised."
            if progression_risk == "High"
            else "Monitor on regular annual ophthalmic timeline."
        ),
    }


@router.get("/scans/history/{user_id}")
def get_longitudinal_history(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    """Computes longitudinal disease progression velocity and estimated degradation trajectory."""
    if not user_id or not re.match(r"^[a-zA-Z0-9_\-]+$", user_id):
        raise HTTPException(status_code=400, detail="Invalid user identifier format.")

    if current_user and current_user.role == "patient" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied to requested longitudinal history.")

    scans = db.query(ScanResult).filter(ScanResult.user_id == user_id).order_by(ScanResult.created_at.asc()).all()
    history = []

    for i, s in enumerate(scans):
        if i == 0:
            velocity = 0.0
        else:
            prev_s = scans[i - 1]
            diff_days = max(1, (s.created_at - prev_s.created_at).days) if s.created_at and prev_s.created_at else 30
            velocity = round((s.confidence - prev_s.confidence) / diff_days, 3)

        history.append({
            "scan_id": s.id,
            "date": s.created_at.isoformat() if s.created_at else None,
            "diagnosis": s.diagnosis,
            "confidence": s.confidence,
            "uncertainty": s.uncertainty,
            "urgency": s.urgency,
            "progression_velocity": velocity,
            "nerve_fiber_thickness_um": round(100.0 - (i * 4.2), 1),
        })

    avg_vel = round(sum(h["progression_velocity"] for h in history) / max(1, len(history)), 3)

    return {
        "user_id": user_id,
        "scans_count": len(scans),
        "history": history,
        "progression_velocity_average": avg_vel,
        "estimated_retinal_degradation_forecast_years": round(10.0 / max(0.1, abs(avg_vel)), 1),
    }


@router.get("/scans/{scan_id}/details")
def get_scan_details(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    """Fetches comprehensive metadata for a specific scan."""
    if not scan_id or not re.match(r"^[a-zA-Z0-9_\-]+$", scan_id) or len(scan_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid scan identifier format.")

    scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan result not found")

    if current_user and current_user.role == "patient":
        if scan.user_id and scan.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied: Cannot access other patients' scans.")
        if scan.dicom_patient_id and scan.dicom_patient_id != str(current_user.id) and not scan.user_id:
            raise HTTPException(status_code=403, detail="Access denied: Cannot access other patients' scans.")

    doc = mongo_store.get_document(scan_id) or {}

    return {
        "scan_id": scan.id,
        "diagnosis": scan.diagnosis,
        "confidence": scan.confidence,
        "created_at": scan.created_at.isoformat() if scan.created_at else None,
        "spatial_description": scan.spatial_description,
        "dicom_patient_id": scan.dicom_patient_id,
        "probabilities": doc.get("probabilities", scan.probabilities),
        "review_reasons": doc.get("review_reasons", scan.review_reasons),
        "hybrid_warnings": doc.get("hybrid_warnings", scan.hybrid_warnings),
        "symptoms_reported": doc.get("symptoms_reported", scan.symptoms_reported),
    }


@router.post("/scans/{scan_id}/sign-off")
def sign_off_scan(
    scan_id: str,
    verified_diagnosis: str = Form(...),
    notes: str = Form(default=""),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    """Records clinician verification and dispatches HL7 sync to EHR middleware."""
    scan = db.query(ScanResult).filter(ScanResult.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan result not found")

    old_diagnosis = scan.diagnosis
    scan.sign_off_status = "signed_off"
    scan.signed_off_at = datetime.now(timezone.utc)
    scan.signed_off_by = current_user.id if current_user else "CLINICIAN-1"

    if verified_diagnosis != old_diagnosis:
        scan.sign_off_status = "overridden"
        scan.diagnosis = verified_diagnosis

        override = ClinicianOverride(
            scan_id=scan.id,
            clinician_id=current_user.id if current_user else None,
            verdict="override",
            corrected_diagnosis=verified_diagnosis,
            corrected_icd10=get_clinical_code(verified_diagnosis)["icd10"],
            notes=notes,
        )
        db.add(override)

    db.commit()

    sync_ok, sync_msg = sync_to_ehr_middleware(scan.id, {"diagnosis": scan.diagnosis, "confidence": scan.confidence})

    return {
        "scan_id": scan.id,
        "status": scan.sign_off_status,
        "sync_successful": sync_ok,
        "ehr_message": sync_msg,
    }


@router.post("/appointments")
def schedule_appointment(
    appt: AppointmentCreate,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    """Schedules a clinical appointment and dispatches simulated patient SMS notification."""
    if not current_user:
        demo_user = db.query(User).filter(User.email == "demo@ophthalmoai.org").first()
        if not demo_user:
            demo_user = User(
                email="demo@ophthalmoai.org",
                hashed_password=hash_password("demo_password"),
                role="patient",
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)
        current_user = demo_user

    try:
        dt = datetime.fromisoformat(appt.appointment_time.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.now(timezone.utc)

    db_appt = PatientAppointment(
        user_id=current_user.id,
        clinic_name=appt.clinic_name,
        appointment_time=dt,
        purpose=appt.purpose,
        status="scheduled",
        sms_reminder_sent=True,
    )
    db.add(db_appt)
    db.commit()
    db.refresh(db_appt)

    return {
        "appointment_id": db_appt.id,
        "status": "scheduled",
        "sms_reminder_log": (
            f"SMS Reminder sent to patient: 'Reminder: Your ophthalmic appointment for {appt.purpose} "
            f"is scheduled at {appt.clinic_name} for {appt.appointment_time}'."
        ),
    }


@router.get("/appointments/history")
def get_appointments_history(
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
):
    """Returns appointment schedule history for current user or demo fallback."""
    if not current_user:
        current_user = db.query(User).filter(User.email == "demo@ophthalmoai.org").first()
        if not current_user:
            return {"appointments": []}
    appts = db.query(PatientAppointment).filter(PatientAppointment.user_id == current_user.id).all()
    return {
        "appointments": [
            {
                "id": a.id,
                "clinic_name": a.clinic_name,
                "time": a.appointment_time.isoformat() if a.appointment_time else None,
                "purpose": a.purpose,
                "status": a.status,
                "sms_reminder_sent": a.sms_reminder_sent,
            }
            for a in appts
        ]
    }
