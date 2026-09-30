from datetime import date, datetime, timedelta, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.analytics import TenantDailyMetric, compute_daily_metrics, _is_normal_diagnosis
from backend.auth import get_current_user
from backend.db import Base, ScanResult, Tenant, User, get_db
from backend.main import app
from backend.tenancy import get_current_tenant_id

from sqlalchemy.pool import StaticPool

@pytest.fixture
def analytics_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Create tenant
    tenant = Tenant(id="tenant-test-1", name="Eye Clinic 1", slug="eye-clinic-1")
    db.add(tenant)
    db.commit()

    yield db
    db.close()

def test_is_normal_diagnosis():
    assert _is_normal_diagnosis("Normal") is True
    assert _is_normal_diagnosis("Healthy fundus") is True
    assert _is_normal_diagnosis("No diabetic retinopathy") is True
    assert _is_normal_diagnosis("Diabetic Retinopathy") is False
    assert _is_normal_diagnosis(None) is False

def test_tenant_daily_metric_to_dict():
    metric = TenantDailyMetric(
        id=1,
        tenant_id="tenant-test-1",
        date=date(2026, 9, 30),
        total_screenings=10,
        total_patients=8,
        avg_confidence=0.88,
        avg_inference_time_ms=95.5,
        diagnoses_by_class={"Normal": 8, "DR": 2},
        high_risk_count=1,
        edge_screenings=3,
        server_screenings=7,
        false_positive_rate=0.0,
        model_agreement_rate=1.0,
    )
    d = metric.to_dict()
    assert d["tenant_id"] == "tenant-test-1"
    assert d["total_screenings"] == 10
    assert d["total_patients"] == 8
    assert d["avg_confidence"] == 0.88

def test_compute_daily_metrics(analytics_db):
    target_d = date(2026, 9, 30)
    target_dt = datetime(2026, 9, 30, 10, 0, 0, tzinfo=timezone.utc)

    # Insert a scan
    scan1 = ScanResult(
        id="scan-1",
        tenant_id="tenant-test-1",
        diagnosis="Normal",
        confidence=95.0,
        dicom_patient_id="patient-1",
        created_at=target_dt,
    )
    scan2 = ScanResult(
        id="scan-2",
        tenant_id="tenant-test-1",
        diagnosis="Diabetic Retinopathy",
        confidence=0.92,
        dicom_patient_id="patient-2",
        router_group_idx=0,
        created_at=target_dt,
    )
    analytics_db.add_all([scan1, scan2])
    analytics_db.commit()

    metric = compute_daily_metrics(
        tenant_id="tenant-test-1",
        target_date=target_d,
        db=analytics_db,
    )
    assert metric.total_screenings == 2
    assert metric.total_patients == 2
    assert metric.edge_screenings == 1
    assert metric.server_screenings == 1
    assert metric.high_risk_count == 1  # scan2 DR with confidence 0.92

def test_analytics_api_endpoints(analytics_db):
    test_user = User(
        id="user-1",
        email="clinician@example.com",
        role="clinician",
        tenant_id="tenant-test-1",
    )

    # Prepopulate daily metric
    metric = TenantDailyMetric(
        tenant_id="tenant-test-1",
        date=date.today(),
        total_screenings=25,
        total_patients=20,
        avg_confidence=0.89,
        avg_inference_time_ms=88.0,
        diagnoses_by_class={"Normal": 15, "Diabetic Retinopathy": 10},
        high_risk_count=4,
        edge_screenings=5,
        server_screenings=20,
        false_positive_rate=0.05,
        model_agreement_rate=0.95,
    )
    analytics_db.add(metric)
    analytics_db.commit()

    def override_get_db():
        yield analytics_db

    def override_current_user():
        return test_user

    def override_tenant_id():
        return "tenant-test-1"

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_current_tenant_id] = override_tenant_id

    client = TestClient(app)

    try:
        # Dashboard endpoint
        resp = client.get("/api/analytics/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "tenant-test-1"
        assert "summary" in data

        # Trends endpoint
        resp_trends = client.get("/api/analytics/trends?metric=screenings&period=7d")
        assert resp_trends.status_code == 200
        trends_data = resp_trends.json()
        assert "data_points" in trends_data

        # Anomalies endpoint
        resp_anom = client.get("/api/analytics/anomalies")
        assert resp_anom.status_code == 200
        assert "anomalies" in resp_anom.json()

        # Comparison endpoint: clinician should be forbidden (403)
        resp_comp_clin = client.get("/api/analytics/comparison")
        assert resp_comp_clin.status_code == 403

        # Switch to admin user: comparison should succeed (200)
        admin_user = User(
            id="admin-1",
            email="admin@example.com",
            role="admin",
            tenant_id="tenant-test-1",
        )
        app.dependency_overrides[get_current_user] = lambda: admin_user
        resp_comp_admin = client.get("/api/analytics/comparison")
        assert resp_comp_admin.status_code == 200
        assert "tenants" in resp_comp_admin.json()
        assert "platform_summary" in resp_comp_admin.json()

        # Reports generate endpoint
        resp_report = client.post(
            "/api/analytics/generate-report",
            json={"period": "30d", "format": "json"}
        )
        assert resp_report.status_code == 200
        report_data = resp_report.json()
        assert report_data["status"] == "success"
        assert "report" in report_data
    finally:
        app.dependency_overrides.clear()
