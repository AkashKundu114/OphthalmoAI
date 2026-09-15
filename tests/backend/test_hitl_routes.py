"""
Tests for Human-in-the-Loop (HITL) analytics & active learning endpoints.
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from backend.auth import hash_password
from backend.db import ClinicianOverride, ModelVersion, ScanResult, SessionLocal, User
import backend.main as main_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_hitl.db"
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-not-for-prod")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("FORCE_CPU", "true")

    with TestClient(main_module.app) as c:
        yield c


def _seed_clinician_and_scans():
    db = SessionLocal()
    clinician = User(
        email=f"clin-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password=hash_password("Sup3rSecure!Pass"),
        role="clinician",
        is_active=True,
    )
    scan1 = ScanResult(
        diagnosis="Glaucoma",
        confidence=91.5,
        group_name="Posterior Segment",
        probabilities={"Glaucoma": 0.915, "Normal": 0.085},
        calibrated=True,
        calibration_temperature=1.2,
        uncertainty=0.04,
        requires_human_review=False,
        review_reasons=[],
        icd10_code="H40.9",
        snomed_code="23986001",
        urgency="urgent",
        urgency_rank=3,
        hybrid_warnings=[],
        hybrid_warnings_structured=[],
        iqa_acceptable=True,
        iqa_warnings=[],
        symptoms_reported={},
    )
    db.add_all([clinician, scan1])
    db.commit()
    db.refresh(clinician)
    db.refresh(scan1)

    override = ClinicianOverride(
        scan_id=scan1.id,
        clinician_id=clinician.id,
        verdict="disagree",
        corrected_diagnosis="Normal",
        corrected_icd10="Z01.00",
        notes="Optic disc cup-to-disc ratio is physiologically large but neuroretinal rim is intact.",
    )
    db.add(override)
    db.commit()

    meta = {
        "clinician_email": clinician.email,
        "scan_id": scan1.id,
    }
    db.close()
    return meta


def _login(c, email, password):
    r = c.post("/auth/token", data={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_hitl_analytics_and_active_learning(client):
    meta = _seed_clinician_and_scans()

    # 1. Unauthenticated should return 401
    res = client.get("/admin/hitl/discrepancies")
    assert res.status_code in (401, 403)

    # 2. Authenticate as clinician
    token = _login(client, meta["clinician_email"], "Sup3rSecure!Pass")
    headers = {"Authorization": f"Bearer {token}"}

    res = client.get("/admin/hitl/discrepancies", headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["total_reviews"] >= 1
    assert data["disagreed_count"] >= 1
    assert "concordance_rate" in data
    assert "discordance_rate" in data
    assert len(data["confusion_pairs"]) >= 1

    # 3. Query active learning candidates
    res_al = client.get("/admin/hitl/active-learning?min_confidence=80.0", headers=headers)
    assert res_al.status_code == 200, res_al.text
    data_al = res_al.json()
    assert data_al["candidate_count"] >= 1
    assert any(c["scan_id"] == meta["scan_id"] for c in data_al["candidates"])
