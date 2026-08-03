"""
Tests for the three endpoints added in backend/routes_admin.py:

    POST /scans/{scan_id}/override
    GET  /admin/audit-logs
    POST /admin/model-registry/activate

Uses FastAPI's TestClient against a real SQLite DB (per-test tmp file, like the rest of
this test suite), not mocks - these endpoints are thin wrappers around real DB writes and
role checks, so the thing worth verifying is that the writes/permissions actually happen,
not that functions get called with the right arguments.

Run with:
    pytest tests/backend/test_routes_admin.py -v
"""
import os
import uuid

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test_routes_admin.db"
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-not-for-prod")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
    monkeypatch.setenv("FORCE_CPU", "true")

    # Reload backend.main fresh so it picks up the monkeypatched env vars - importing it
    # at module level would bind DATABASE_URL etc. before the fixture runs.
    import importlib
    import backend.db as db_module
    import backend.db_async as db_async_module
    importlib.reload(db_module)
    importlib.reload(db_async_module)
    import backend.main as main_module
    importlib.reload(main_module)

    from fastapi.testclient import TestClient
    with TestClient(main_module.app) as c:
        yield c, main_module, db_module


def _seed_users_and_scan(db_module):
    from backend.auth import hash_password
    from backend.db import SessionLocal, User, ScanResult, ModelVersion

    db = SessionLocal()
    clinician = User(email=f"clin-{uuid.uuid4().hex[:8]}@example.com",
                      hashed_password=hash_password("Sup3rSecure!Pass"),
                      role="clinician", is_active=True)
    patient = User(email=f"pat-{uuid.uuid4().hex[:8]}@example.com",
                    hashed_password=hash_password("Sup3rSecure!Pass2"),
                    role="patient", is_active=True)
    admin = User(email=f"admin-{uuid.uuid4().hex[:8]}@example.com",
                 hashed_password=hash_password("Sup3rSecure!Pass3"),
                 role="admin", is_active=True)
    scan = ScanResult(diagnosis="Uveitis", confidence=62.5, group_name="Anterior Segment Pathology",
                       probabilities={"Cataract": 0.375, "Uveitis": 0.625}, calibrated=True,
                       calibration_temperature=1.3, uncertainty=0.09, requires_human_review=True,
                       review_reasons=["low confidence"], icd10_code="H20.9", snomed_code="128473001",
                       urgency="urgent", urgency_rank=3, hybrid_warnings=[], hybrid_warnings_structured=[],
                       iqa_acceptable=True, iqa_warnings=[], symptoms_reported={})
    mv1 = ModelVersion(group_key="anterior", version_tag="v1", architecture="EfficientNet-B4",
                        weights_path="/models/anterior_v1.pth", val_accuracy=0.88, active=True)
    mv2 = ModelVersion(group_key="anterior", version_tag="v2", architecture="EfficientNet-B4",
                        weights_path="/models/anterior_v2.pth", val_accuracy=0.91, active=False)
    db.add_all([clinician, patient, admin, scan, mv1, mv2])
    db.commit()
    for obj in (clinician, patient, admin, scan, mv1, mv2):
        db.refresh(obj)
    ids = dict(clinician_email=clinician.email, patient_email=patient.email, admin_email=admin.email,
               scan_id=scan.id, mv1_id=mv1.id, mv2_id=mv2.id)
    db.close()
    return ids


def _login(client, email, password):
    r = client.post("/auth/token", data={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


class TestScanOverride:

    def test_clinician_can_create_override(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        token = _login(c, ids["clinician_email"], "Sup3rSecure!Pass")

        r = c.post(f"/scans/{ids['scan_id']}/override",
                   json={"verdict": "disagree", "corrected_diagnosis": "Cataract"},
                   headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["verdict"] == "disagree"
        assert body["corrected_diagnosis"] == "Cataract"
        assert body["scan_id"] == ids["scan_id"]

    def test_disagree_without_corrected_diagnosis_rejected(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        token = _login(c, ids["clinician_email"], "Sup3rSecure!Pass")

        r = c.post(f"/scans/{ids['scan_id']}/override",
                   json={"verdict": "disagree"},
                   headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 422

    def test_invalid_verdict_rejected(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        token = _login(c, ids["clinician_email"], "Sup3rSecure!Pass")

        r = c.post(f"/scans/{ids['scan_id']}/override",
                   json={"verdict": "not_a_real_verdict"},
                   headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 422

    def test_duplicate_override_rejected_409(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        token = _login(c, ids["clinician_email"], "Sup3rSecure!Pass")

        r1 = c.post(f"/scans/{ids['scan_id']}/override", json={"verdict": "agree"},
                    headers={"Authorization": f"Bearer {token}"})
        assert r1.status_code == 201

        r2 = c.post(f"/scans/{ids['scan_id']}/override", json={"verdict": "agree"},
                    headers={"Authorization": f"Bearer {token}"})
        assert r2.status_code == 409

    def test_unknown_scan_404(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        token = _login(c, ids["clinician_email"], "Sup3rSecure!Pass")

        r = c.post("/scans/nonexistent-scan-id/override", json={"verdict": "agree"},
                   headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 404

    def test_patient_forbidden(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        token = _login(c, ids["patient_email"], "Sup3rSecure!Pass2")

        r = c.post(f"/scans/{ids['scan_id']}/override", json={"verdict": "agree"},
                   headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_no_auth_401(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)

        r = c.post(f"/scans/{ids['scan_id']}/override", json={"verdict": "agree"})
        assert r.status_code == 401


class TestAdminAuditLogs:

    def test_admin_can_list(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        # Generate at least one audit event via a real login first.
        _login(c, ids["clinician_email"], "Sup3rSecure!Pass")
        admin_token = _login(c, ids["admin_email"], "Sup3rSecure!Pass3")

        r = c.get("/admin/audit-logs", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] >= 1
        assert any(e["action"] == "login" for e in body["entries"])

    def test_clinician_forbidden(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        token = _login(c, ids["clinician_email"], "Sup3rSecure!Pass")

        r = c.get("/admin/audit-logs", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_filter_by_action(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        _login(c, ids["clinician_email"], "Sup3rSecure!Pass")
        admin_token = _login(c, ids["admin_email"], "Sup3rSecure!Pass3")

        r = c.get("/admin/audit-logs?action=login", headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        assert all(e["action"] == "login" for e in r.json()["entries"])


class TestAdminModelRegistryActivate:

    def test_admin_can_activate(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        admin_token = _login(c, ids["admin_email"], "Sup3rSecure!Pass3")

        r = c.post("/admin/model-registry/activate", json={"version_id": ids["mv2_id"]},
                   headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 200
        body = r.json()
        assert body["active"] is True
        assert body["id"] == ids["mv2_id"]
        assert "does NOT hot-swap" in body["warning"]

    def test_activating_new_version_deactivates_old(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        admin_token = _login(c, ids["admin_email"], "Sup3rSecure!Pass3")

        c.post("/admin/model-registry/activate", json={"version_id": ids["mv2_id"]},
               headers={"Authorization": f"Bearer {admin_token}"})

        from backend.db import SessionLocal, ModelVersion
        db = SessionLocal()
        mv1 = db.query(ModelVersion).filter(ModelVersion.id == ids["mv1_id"]).first()
        assert mv1.active is False
        db.close()

    def test_unknown_version_404(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        admin_token = _login(c, ids["admin_email"], "Sup3rSecure!Pass3")

        r = c.post("/admin/model-registry/activate", json={"version_id": "does-not-exist"},
                   headers={"Authorization": f"Bearer {admin_token}"})
        assert r.status_code == 404

    def test_clinician_forbidden(self, client):
        c, main_module, db_module = client
        ids = _seed_users_and_scan(db_module)
        token = _login(c, ids["clinician_email"], "Sup3rSecure!Pass")

        r = c.post("/admin/model-registry/activate", json={"version_id": ids["mv2_id"]},
                   headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403
