"""
Tests for Multi-Tenant Isolation, Clinic RBAC & Row-Level Data Partitioning.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.auth import ROLE_HIERARCHY
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.auth import ROLE_HIERARCHY, get_current_user
from backend.db import Base, ScanResult, Tenant, User
from backend.main import app
from backend.tenancy import (
    DEFAULT_TENANT_ID,
    apply_tenant_filter,
    create_new_tenant,
    ensure_default_tenant,
    get_current_tenant_id,
)
from backend.validators import validate_role_claim


@pytest.fixture
def mem_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()


def test_rbac_technician_role_hierarchy():
    # Verify role hierarchy ordering
    assert ROLE_HIERARCHY["patient"] == 0
    assert ROLE_HIERARCHY["technician"] == 1
    assert ROLE_HIERARCHY["clinician"] == 2
    assert ROLE_HIERARCHY["admin"] == 3

    valid, role = validate_role_claim("technician")
    assert valid is True
    assert role == "technician"


def test_tenant_creation_and_row_level_isolation(mem_db):
    # Create two distinct clinic tenants
    clinic_a = create_new_tenant(mem_db, name="Apex Retina Center", slug="apex-retina")
    clinic_b = create_new_tenant(mem_db, name="St. Jude Eye Clinic", slug="st-jude-eye")

    # Add scan records for each clinic
    scan_a = ScanResult(
        diagnosis="Diabetic Retinopathy",
        confidence=0.94,
        tenant_id=clinic_a.id,
    )
    scan_b = ScanResult(
        diagnosis="Glaucoma",
        confidence=0.88,
        tenant_id=clinic_b.id,
    )
    mem_db.add_all([scan_a, scan_b])
    mem_db.commit()

    # Query scoped to Clinic A
    query_a = apply_tenant_filter(mem_db.query(ScanResult), ScanResult, clinic_a.id).all()
    assert len(query_a) == 1
    assert query_a[0].diagnosis == "Diabetic Retinopathy"

    # Query scoped to Clinic B
    query_b = apply_tenant_filter(mem_db.query(ScanResult), ScanResult, clinic_b.id).all()
    assert len(query_b) == 1
    assert query_b[0].diagnosis == "Glaucoma"


def test_tenant_id_resolution():
    # 1. Platform admin and superadmin can override tenant context via header
    admin_user = User(email="admin@clinic.org", role="platform_admin")
    admin_user.tenant_id = "default-org"
    assert get_current_tenant_id(x_tenant_id="custom-clinic-1", current_user=admin_user) == "custom-clinic-1"

    super_user = User(email="super@clinic.org", role="superadmin")
    super_user.tenant_id = "default-org"
    assert get_current_tenant_id(x_tenant_id="custom-clinic-2", current_user=super_user) == "custom-clinic-2"

    # 2. Regular user ALWAYS uses JWT tenant and ignores matching header
    dummy_user = User(email="tech@clinic.org", role="technician")
    dummy_user.tenant_id = "user-org-99"
    assert get_current_tenant_id(x_tenant_id=None, current_user=dummy_user) == "user-org-99"
    assert get_current_tenant_id(x_tenant_id="user-org-99", current_user=dummy_user) == "user-org-99"


def test_regular_user_cannot_access_other_tenant_via_header_spoofing(mem_db):
    """
    IDOR Prevention Test:
    Verifies that a regular user (clinician/technician) CANNOT access another clinic's
    data by spoofing the X-Tenant-ID header.
    """
    clinic_a = create_new_tenant(mem_db, name="Clinic Alpha", slug="clinic-alpha")
    clinic_b = create_new_tenant(mem_db, name="Clinic Beta", slug="clinic-beta")

    scan_a = ScanResult(diagnosis="Normal", confidence=0.99, tenant_id=clinic_a.id)
    scan_b = ScanResult(diagnosis="Severe DR", confidence=0.92, tenant_id=clinic_b.id)
    mem_db.add_all([scan_a, scan_b])
    mem_db.commit()

    regular_user = User(id="user-123", email="doc@clinic-a.com", role="clinician")
    regular_user.tenant_id = clinic_a.id

    # Attempting to spoof clinic_b via X-Tenant-ID must fail with HTTP 403
    with pytest.raises(HTTPException) as exc_info:
        get_current_tenant_id(x_tenant_id=clinic_b.id, current_user=regular_user)
    assert exc_info.value.status_code == 403
    assert "Tenant access denied" in exc_info.value.detail

    # When request header is not spoofed, user can only access their own tenant
    legit_tenant_id = get_current_tenant_id(x_tenant_id=None, current_user=regular_user)
    assert legit_tenant_id == clinic_a.id

    scans = apply_tenant_filter(mem_db.query(ScanResult), ScanResult, legit_tenant_id).all()
    assert len(scans) == 1
    assert scans[0].diagnosis == "Normal"
    assert scans[0].tenant_id == clinic_a.id


def test_platform_admin_can_use_x_tenant_id_to_access_other_tenants(mem_db):
    """
    Verifies that users with 'platform_admin' or 'superadmin' roles CAN use
    X-Tenant-ID to switch clinic tenant context.
    """
    clinic_a = create_new_tenant(mem_db, name="Regional Hospital A", slug="reg-hosp-a")
    clinic_b = create_new_tenant(mem_db, name="Regional Hospital B", slug="reg-hosp-b")

    scan_b = ScanResult(diagnosis="Cataract", confidence=0.85, tenant_id=clinic_b.id)
    mem_db.add(scan_b)
    mem_db.commit()

    admin = User(id="admin-999", email="admin@platform.net", role="platform_admin")
    admin.tenant_id = clinic_a.id

    # Platform admin overrides tenant to clinic_b
    resolved_tenant = get_current_tenant_id(x_tenant_id=clinic_b.id, current_user=admin)
    assert resolved_tenant == clinic_b.id

    scans = apply_tenant_filter(mem_db.query(ScanResult), ScanResult, resolved_tenant).all()
    assert len(scans) == 1
    assert scans[0].tenant_id == clinic_b.id
    assert scans[0].diagnosis == "Cataract"

    # Superadmin also permitted
    superadmin = User(id="super-111", email="super@platform.net", role="superadmin")
    superadmin.tenant_id = clinic_a.id
    assert get_current_tenant_id(x_tenant_id=clinic_b.id, current_user=superadmin) == clinic_b.id


def test_mismatched_x_tenant_id_from_regular_user_returns_403():
    """
    Verifies that a mismatched X-Tenant-ID header from a non-admin user
    is rejected with HTTP 403 Forbidden.
    """
    regular_user = User(id="user-456", email="tech@metro.org", role="technician")
    regular_user.tenant_id = "tenant-metro"

    # Mismatched tenant header
    with pytest.raises(HTTPException) as exc_info:
        get_current_tenant_id(x_tenant_id="tenant-victim", current_user=regular_user)
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Tenant access denied"


def test_api_endpoint_idor_prevention():
    """
    FastAPI endpoint integration test:
    Validates that /api/v1/tenants/current enforces IDOR protections over HTTP.
    """
    from backend.db import SessionLocal

    db = SessionLocal()
    tenant_a = db.query(Tenant).filter(Tenant.slug == "clinic-alpha").first()
    if not tenant_a:
        tenant_a = Tenant(name="Clinic Alpha", slug="clinic-alpha", tier="hospital_standard", is_active=True)
        db.add(tenant_a)
    tenant_b = db.query(Tenant).filter(Tenant.slug == "clinic-beta").first()
    if not tenant_b:
        tenant_b = Tenant(name="Clinic Beta", slug="clinic-beta", tier="hospital_standard", is_active=True)
        db.add(tenant_b)
    db.commit()
    db.refresh(tenant_a)
    db.refresh(tenant_b)
    db.close()

    test_client = TestClient(app)

    reg_user = User(id="user-789", email="clinician@eye.org", role="clinician")
    reg_user.tenant_id = tenant_a.id

    admin_user = User(id="admin-001", email="platform@eye.org", role="platform_admin")
    admin_user.tenant_id = "default-metro-eye-hospital"

    try:
        # 1. Regular user trying to spoof tenant -> 403 Forbidden
        app.dependency_overrides[get_current_user] = lambda: reg_user
        res = test_client.get("/api/v1/tenants/current", headers={"X-Tenant-ID": tenant_b.id})
        assert res.status_code == 403
        assert res.json()["detail"] == "Tenant access denied"

        # 2. Regular user without header -> 200 OK with their own tenant
        res = test_client.get("/api/v1/tenants/current")
        assert res.status_code == 200
        assert res.json()["tenant_id"] == tenant_a.id

        # 3. Platform admin with header -> 200 OK with overridden tenant
        app.dependency_overrides[get_current_user] = lambda: admin_user
        res = test_client.get("/api/v1/tenants/current", headers={"X-Tenant-ID": tenant_b.id})
        assert res.status_code == 200
        assert res.json()["tenant_id"] == tenant_b.id

        # 4. Unauthenticated user trying to spoof tenant -> 403 Forbidden
        app.dependency_overrides[get_current_user] = lambda: None
        res = test_client.get("/api/v1/tenants/current", headers={"X-Tenant-ID": tenant_b.id})
        assert res.status_code == 403
        assert res.json()["detail"] == "Tenant access denied"

    finally:
        app.dependency_overrides.pop(get_current_user, None)

