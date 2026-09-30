"""
Comprehensive Test Suite for Multi-Tenant Analytics & Anomaly Detection.

Validates ad-tech telemetry patterns:
- Time-series aggregation & KPIs
- Statistical anomaly detection (Z-score volume, MA quality, SLA breach, Chi-square distribution shift)
- Multi-tenant data isolation & IDOR prevention
- KPI dashboard, trend charting, anomaly alerts, admin comparisons, and report generation.
"""

from __future__ import annotations

from datetime import date, datetime as dt, timedelta, timezone
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.analytics import TenantDailyMetric, compute_daily_metrics
from backend.auth import create_access_token, get_current_user
from backend.db import (
    AuditLog, Base, ClinicianOverride, ModelVersion, ScanResult,
    Tenant, User, get_db,
)
from backend.main import app
from backend.screening_anomaly_detector import ScreeningAnomaly, ScreeningAnomalyDetector
from backend.tenancy import DEFAULT_TENANT_ID, get_current_tenant_id


@pytest.fixture
def analytics_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()

    def _get_test_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_test_db
    yield db
    app.dependency_overrides.pop(get_db, None)
    db.close()


@pytest.fixture
def test_tenants(analytics_db):
    t_a = Tenant(
        id="tenant-alpha",
        name="Alpha Eye Clinic",
        slug="alpha-eye",
        tier="hospital_standard",
        is_active=True,
    )
    t_b = Tenant(
        id="tenant-beta",
        name="Beta Eye Center",
        slug="beta-eye",
        tier="specialty_clinic",
        is_active=True,
    )
    analytics_db.add_all([t_a, t_b])
    analytics_db.commit()
    return t_a, t_b


# =========================================================================
# 1. Daily Metric Aggregation
# =========================================================================

def test_daily_metric_aggregation_produces_correct_counts(analytics_db, test_tenants):
    t_a, _ = test_tenants
    today = dt.now(timezone.utc).date()
    target_dt = dt(today.year, today.month, today.day, 10, 30, tzinfo=timezone.utc)

    # 4 Scans for Tenant A across 2 distinct patients
    s1 = ScanResult(
        tenant_id=t_a.id,
        user_id="patient-1",
        diagnosis="Diabetic Retinopathy",
        confidence=94.5,
        created_at=target_dt,
    )
    s2 = ScanResult(
        tenant_id=t_a.id,
        user_id="patient-1",
        diagnosis="Diabetic Retinopathy",
        confidence=0.88,
        created_at=target_dt,
    )
    s3 = ScanResult(
        tenant_id=t_a.id,
        user_id="patient-2",
        diagnosis="Normal",
        confidence=99.0,
        created_at=target_dt,
    )
    s4 = ScanResult(
        tenant_id=t_a.id,
        user_id="patient-2",
        diagnosis="Glaucoma",
        confidence=0.85,
        created_at=target_dt,
    )

    analytics_db.add_all([s1, s2, s3, s4])
    analytics_db.commit()

    # Aggregate
    metric = compute_daily_metrics(t_a.id, today, db=analytics_db)

    assert metric.total_screenings == 4
    assert metric.total_patients == 2
    assert metric.diagnoses_by_class.get("Diabetic Retinopathy") == 2
    assert metric.diagnoses_by_class.get("Normal") == 1
    assert metric.diagnoses_by_class.get("Glaucoma") == 1
    assert metric.high_risk_count == 3  # 2 DR + 1 Glaucoma with confidence > 0.8
    assert metric.avg_confidence > 0.85


# =========================================================================
# 2. Z-Score Volume Anomaly Detection
# =========================================================================

def test_z_score_anomaly_detection_catches_volume_spike():
    detector = ScreeningAnomalyDetector()
    history = [100.0, 104.0, 98.0, 102.0, 101.0, 99.0, 103.0]  # mean ~101, std ~2.16
    current = 260.0  # Massive spike (Z > 70)

    anomaly = detector.detect_volume_anomaly("tenant-test", current, history)
    assert anomaly is not None
    assert anomaly.anomaly_type == "volume"
    assert anomaly.severity == "critical"
    assert anomaly.current_value == 260.0
    assert "Similar to detecting sudden impression drops in campaign monitoring" in anomaly.ad_tech_parallel


def test_z_score_does_not_flag_normal_daily_variance():
    detector = ScreeningAnomalyDetector()
    history = [100.0, 104.0, 98.0, 102.0, 101.0, 99.0, 103.0]
    current = 104.0  # Within ~1.4σ

    anomaly = detector.detect_volume_anomaly("tenant-test", current, history)
    assert anomaly is None


# =========================================================================
# 3. Quality Anomaly Detection (Moving Average Drop)
# =========================================================================

def test_quality_anomaly_detection_catches_confidence_drop():
    detector = ScreeningAnomalyDetector()
    # 3-day history with ~95% confidence
    history = [0.95, 0.94, 0.96]
    current = 0.72  # ~24% decline (bad camera optics)

    anomaly = detector.detect_quality_anomaly("tenant-test", current, history)
    assert anomaly is not None
    assert anomaly.anomaly_type == "quality"
    assert anomaly.severity == "critical"
    assert "Similar to detecting CTR decline from ad fatigue" in anomaly.ad_tech_parallel


def test_quality_anomaly_does_not_flag_normal_fluctuation():
    detector = ScreeningAnomalyDetector()
    history = [0.95, 0.94, 0.96]
    current = 0.93  # ~2% fluctuation, well below 10% threshold

    anomaly = detector.detect_quality_anomaly("tenant-test", current, history)
    assert anomaly is None


# =========================================================================
# 4. SLA Breach Detection
# =========================================================================

def test_sla_breach_detection_flags_slow_inference_times():
    detector = ScreeningAnomalyDetector(latency_sla_ms=200.0)

    # Compliant latency
    assert detector.detect_sla_breach("tenant-test", 85.0) is None

    # Breached latency
    breach = detector.detect_sla_breach("tenant-test", 275.0)
    assert breach is not None
    assert breach.anomaly_type == "sla_breach"
    assert breach.current_value == 275.0
    assert breach.baseline_value == 200.0
    assert "Similar to bid response latency exceeding RTB timeout" in breach.ad_tech_parallel


# =========================================================================
# 5. Distribution Shift Detection (Chi-Square)
# =========================================================================

def test_distribution_shift_detection_catches_diagnosis_imbalance():
    detector = ScreeningAnomalyDetector()

    # Historical baseline: 60% Normal, 20% DR, 10% Glaucoma, 10% Cataract
    hist_day = {"Normal": 60, "Diabetic Retinopathy": 20, "Glaucoma": 10, "Cataract": 10}
    history = [hist_day] * 10

    # Today's surge: 80% DR, 10% Normal, 5% Glaucoma, 5% Cataract
    current = {"Diabetic Retinopathy": 80, "Normal": 10, "Glaucoma": 5, "Cataract": 5}

    shift = detector.detect_distribution_shift("tenant-test", current, history)
    assert shift is not None
    assert shift.anomaly_type == "distribution_shift"
    assert "Similar to audience composition drift in programmatic targeting" in shift.ad_tech_parallel


# =========================================================================
# 6. Tenant KPI Dashboard API & Isolation
# =========================================================================

def test_kpi_dashboard_api_returns_correct_tenant_data(analytics_db, test_tenants):
    t_a, _ = test_tenants
    today = dt.now(timezone.utc).date()

    # Seed 7 days of metrics for Tenant A
    for i in range(7):
        d = today - timedelta(days=i)
        m = TenantDailyMetric(
            tenant_id=t_a.id,
            date=d,
            total_screenings=120,
            total_patients=110,
            avg_confidence=0.95,
            avg_inference_time_ms=80.0,
            diagnoses_by_class={"Normal": 80, "Diabetic Retinopathy": 40},
            high_risk_count=25,
        )
        analytics_db.add(m)
    analytics_db.commit()

    app.dependency_overrides[get_current_tenant_id] = lambda: t_a.id

    client = TestClient(app)
    try:
        response = client.get("/api/analytics/dashboard")
        assert response.status_code == 200
        data = response.json()

        assert data["tenant_id"] == t_a.id
        assert data["summary"]["total_screenings_7d"] == 7 * 120
        assert data["summary"]["avg_confidence"] > 90.0
        assert len(data["diagnosis_distribution"]) >= 2
        assert "ad_tech_summary" in data
    finally:
        app.dependency_overrides.pop(get_current_tenant_id, None)


def test_tenant_isolation_tenant_a_cannot_see_tenant_b(analytics_db, test_tenants):
    t_a, t_b = test_tenants
    today = dt.now(timezone.utc).date()

    # Tenant A has 50 scans, Tenant B has 500 scans
    m_a = TenantDailyMetric(
        tenant_id=t_a.id,
        date=today,
        total_screenings=50,
        total_patients=45,
        avg_confidence=0.92,
        avg_inference_time_ms=75.0,
        diagnoses_by_class={"Normal": 50},
    )
    m_b = TenantDailyMetric(
        tenant_id=t_b.id,
        date=today,
        total_screenings=500,
        total_patients=450,
        avg_confidence=0.98,
        avg_inference_time_ms=90.0,
        diagnoses_by_class={"Normal": 500},
    )
    analytics_db.add_all([m_a, m_b])
    analytics_db.commit()

    # User bound to Tenant A
    user_a = User(id="user-a", email="clinician@alpha.org", role="clinician", tenant_id=t_a.id)
    app.dependency_overrides[get_current_user] = lambda: user_a

    client = TestClient(app)
    try:
        # Request without header defaults to user's JWT tenant
        res_legit = client.get("/api/analytics/dashboard")
        assert res_legit.status_code == 200
        assert res_legit.json()["tenant_id"] == t_a.id
        assert res_legit.json()["summary"]["total_screenings_30d"] == 50

        # Attempt to spoof Tenant B via X-Tenant-ID header -> Must be 403 Forbidden
        res_spoof = client.get("/api/analytics/dashboard", headers={"X-Tenant-ID": t_b.id})
        assert res_spoof.status_code == 403
        assert "Tenant access denied" in res_spoof.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_current_user, None)


# =========================================================================
# 7. Time-Series Trends Endpoint
# =========================================================================

def test_trend_endpoint_returns_correct_timeseries_shape(analytics_db, test_tenants):
    t_a, _ = test_tenants
    today = dt.now(timezone.utc).date()

    for i in range(10):
        d = today - timedelta(days=9 - i)
        m = TenantDailyMetric(
            tenant_id=t_a.id,
            date=d,
            total_screenings=100 + i * 5,
            avg_confidence=0.94,
            avg_inference_time_ms=85.0,
            high_risk_count=15,
        )
        analytics_db.add(m)
    analytics_db.commit()

    app.dependency_overrides[get_current_tenant_id] = lambda: t_a.id

    client = TestClient(app)
    try:
        res = client.get("/api/analytics/trends?metric=screenings&period=30d")
        assert res.status_code == 200
        data = res.json()

        assert data["metric"] == "screenings"
        assert data["period"] == "30d"
        assert len(data["data_points"]) == 10
        first_pt = data["data_points"][0]
        assert "date" in first_pt
        assert "value" in first_pt
        assert "baseline" in first_pt
        assert "is_anomaly" in first_pt
    finally:
        app.dependency_overrides.pop(get_current_tenant_id, None)


def test_trend_endpoint_supports_different_metrics_and_periods(analytics_db, test_tenants):
    t_a, _ = test_tenants
    app.dependency_overrides[get_current_tenant_id] = lambda: t_a.id

    client = TestClient(app)
    try:
        for m in ("screenings", "confidence", "inference_time", "high_risk_count"):
            for p in ("7d", "30d", "90d"):
                res = client.get(f"/api/analytics/trends?metric={m}&period={p}")
                assert res.status_code == 200
                assert res.json()["metric"] == m
    finally:
        app.dependency_overrides.pop(get_current_tenant_id, None)


# =========================================================================
# 8. Anomaly Endpoint & Ad-Tech Parallel Field
# =========================================================================

def test_anomaly_endpoint_includes_ad_tech_parallel_field(analytics_db, test_tenants):
    t_a, _ = test_tenants
    today = dt.now(timezone.utc).date()

    # Seed baseline 7 days + 1 SLA breach day (>200ms)
    for i in range(7):
        analytics_db.add(TenantDailyMetric(
            tenant_id=t_a.id,
            date=today - timedelta(days=7 - i),
            total_screenings=100,
            avg_confidence=0.95,
            avg_inference_time_ms=80.0,
            diagnoses_by_class={"Normal": 100},
        ))
    # Breached today
    analytics_db.add(TenantDailyMetric(
        tenant_id=t_a.id,
        date=today,
        total_screenings=100,
        avg_confidence=0.95,
        avg_inference_time_ms=280.0,  # Breach!
        diagnoses_by_class={"Normal": 100},
    ))
    analytics_db.commit()

    app.dependency_overrides[get_current_tenant_id] = lambda: t_a.id

    client = TestClient(app)
    try:
        res = client.get("/api/analytics/anomalies")
        assert res.status_code == 200
        anomalies = res.json()["anomalies"]
        assert len(anomalies) >= 1

        sla_anom = next(a for a in anomalies if a["anomaly_type"] == "sla_breach")
        assert "ad_tech_parallel" in sla_anom
        assert "RTB timeout" in sla_anom["ad_tech_parallel"]
        assert "recommendation" in sla_anom
    finally:
        app.dependency_overrides.pop(get_current_tenant_id, None)


# =========================================================================
# 9. Cross-Tenant Admin Comparison
# =========================================================================

def test_cross_tenant_comparison_requires_admin(analytics_db, test_tenants):
    # Regular clinician
    app.dependency_overrides[get_current_user] = lambda: User(role="clinician")

    client = TestClient(app)
    try:
        res = client.get("/api/analytics/comparison")
        assert res.status_code == 403
        assert "Admin role required" in res.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_cross_tenant_comparison_admin_success(analytics_db, test_tenants):
    t_a, t_b = test_tenants
    today = dt.now(timezone.utc).date()

    analytics_db.add(TenantDailyMetric(
        tenant_id=t_a.id,
        date=today,
        total_screenings=150,
        avg_confidence=0.94,
        avg_inference_time_ms=85.0,
        high_risk_count=20,
    ))
    analytics_db.add(TenantDailyMetric(
        tenant_id=t_b.id,
        date=today,
        total_screenings=220,
        avg_confidence=0.96,
        avg_inference_time_ms=78.0,
        high_risk_count=35,
    ))
    analytics_db.commit()

    app.dependency_overrides[get_current_user] = lambda: User(role="admin")

    client = TestClient(app)
    try:
        res = client.get("/api/analytics/comparison")
        assert res.status_code == 200
        body = res.json()
        assert "platform_summary" in body
        assert len(body["tenants"]) >= 2
    finally:
        app.dependency_overrides.pop(get_current_user, None)


# =========================================================================
# 10. Report Generation Endpoint
# =========================================================================

def test_report_generation_produces_valid_output(analytics_db, test_tenants):
    t_a, _ = test_tenants
    today = dt.now(timezone.utc).date()

    analytics_db.add(TenantDailyMetric(
        tenant_id=t_a.id,
        date=today,
        total_screenings=130,
        avg_confidence=0.95,
        avg_inference_time_ms=82.0,
        diagnoses_by_class={"Normal": 100, "DR": 30},
        high_risk_count=18,
    ))
    analytics_db.commit()

    app.dependency_overrides[get_current_tenant_id] = lambda: t_a.id

    client = TestClient(app)
    try:
        res = client.post(
            "/api/analytics/generate-report",
            json={"period": "30d", "format": "json"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        rep = data["report"]
        assert rep["tenant_id"] == t_a.id
        assert "executive_summary" in rep
        assert rep["executive_summary"]["total_screenings"] == 130
        assert "ad_tech_executive_notes" in rep
        assert len(rep["ad_tech_executive_notes"]) >= 3
    finally:
        app.dependency_overrides.pop(get_current_tenant_id, None)
