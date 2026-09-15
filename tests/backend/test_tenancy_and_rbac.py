"""
Tests for Multi-Tenant Isolation, Clinic RBAC & Row-Level Data Partitioning.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.auth import ROLE_HIERARCHY
from backend.db import Base, ScanResult, Tenant, User
from backend.tenancy import (
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
    # 1. From header
    assert get_current_tenant_id(x_tenant_id="custom-clinic-1") == "custom-clinic-1"

    # 2. From authenticated user
    dummy_user = User(email="tech@clinic.org", role="technician")
    dummy_user.tenant_id = "user-org-99"
    assert get_current_tenant_id(x_tenant_id=None, current_user=dummy_user) == "user-org-99"
