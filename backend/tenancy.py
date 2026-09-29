"""
Multi-Tenant Isolation, Clinic RBAC, and Row-Level Data Partitioning.

Enforces strict tenant isolation across hospital organizations to prevent
cross-clinic data leakage in clinical records, screening results, and audits.
Supports PostgreSQL, MS SQL Server, and SQLite dialect schemas.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple
from fastapi import Header, HTTPException, status
from sqlalchemy import inspect
from sqlalchemy.orm import Query, Session

from .db import Tenant, User


DEFAULT_TENANT_ID = "default-metro-eye-hospital"
DEFAULT_TENANT_SLUG = "metro-eye-hospital"

# Tenant ID/Slug format: 3 to 64 alphanumeric chars, underscores, or hyphens.
_TENANT_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{3,64}$")
_UNSATISFIABLE_TENANT_SENTINEL = "__TENANT_ACCESS_DENIED__"


def validate_tenant_identifier(identifier: Optional[str]) -> Tuple[bool, str]:
    """
    Validates and sanitizes a tenant ID or tenant slug.
    Rejects malformed characters, whitespace-only strings, and potential injection vectors.

    Returns:
        Tuple of (is_valid, sanitized_string_or_error_message).
    """
    if identifier is None:
        return False, "Tenant identifier cannot be None."
    
    cleaned = identifier.strip()
    if not cleaned:
        return False, "Tenant identifier cannot be empty."

    if len(cleaned) < 3 or len(cleaned) > 64:
        return False, f"Tenant identifier length must be between 3 and 64 characters (got {len(cleaned)})."

    if not _TENANT_IDENTIFIER_PATTERN.match(cleaned):
        return False, "Tenant identifier contains invalid characters. Only alphanumeric, hyphens, and underscores allowed."

    return True, cleaned


def get_current_tenant_id(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    current_user: Optional[User] = None,
) -> str:
    """
    Extracts and sanitizes tenant ID from request header, authenticated user,
    or falls back to system default.

    Raises:
        HTTPException(400): If explicitly supplied X-Tenant-ID header violates format bounds.
    """
    if x_tenant_id is not None and x_tenant_id.strip():
        is_valid, sanitized_or_err = validate_tenant_identifier(x_tenant_id)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid X-Tenant-ID header: {sanitized_or_err}",
            )
        return sanitized_or_err

    if current_user and getattr(current_user, "tenant_id", None):
        user_tenant = str(current_user.tenant_id).strip()
        is_valid, sanitized = validate_tenant_identifier(user_tenant)
        if is_valid:
            return sanitized

    return DEFAULT_TENANT_ID


def ensure_default_tenant(db: Session) -> Tenant:
    """Ensures the baseline clinical tenant exists in the persistence layer."""
    tenant = db.query(Tenant).filter(Tenant.slug == DEFAULT_TENANT_SLUG).first()
    if not tenant:
        tenant = Tenant(
            id=DEFAULT_TENANT_ID,
            name="Metro Eye Institute & Research Hospital",
            slug=DEFAULT_TENANT_SLUG,
            tier="tertiary_care",
            is_active=True,
        )
        db.add(tenant)
        try:
            db.commit()
            db.refresh(tenant)
        except Exception:
            db.rollback()
            tenant = db.query(Tenant).filter(Tenant.slug == DEFAULT_TENANT_SLUG).first()
    return tenant


def apply_tenant_filter(query: Query, model: Any, tenant_id: Optional[str]) -> Query:
    """
    Applies strict row-level security isolation to an SQLAlchemy query.

    Ensures parameterized filtering compatible with PostgreSQL, MS SQL Server, and SQLite.
    If tenant_id is missing or invalid, an unsatisfiable condition is bound to avoid
    accidental global data disclosure.
    """
    if not hasattr(model, "tenant_id"):
        return query

    if not tenant_id:
        return query.filter(model.tenant_id == _UNSATISFIABLE_TENANT_SENTINEL)

    is_valid, sanitized_tenant = validate_tenant_identifier(tenant_id)
    if not is_valid:
        return query.filter(model.tenant_id == _UNSATISFIABLE_TENANT_SENTINEL)

    return query.filter(model.tenant_id == sanitized_tenant)


def create_new_tenant(
    db: Session,
    name: str,
    slug: str,
    tier: str = "hospital_standard",
) -> Tenant:
    """Registers a new clinic tenant after validating input constraints."""
    clean_name = name.strip()
    if not clean_name or len(clean_name) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant name must be between 1 and 255 characters.",
        )

    is_valid_slug, sanitized_slug = validate_tenant_identifier(slug)
    if not is_valid_slug:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tenant slug: {sanitized_slug}",
        )

    existing = db.query(Tenant).filter(Tenant.slug == sanitized_slug).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tenant with slug '{sanitized_slug}' already registered.",
        )

    tenant = Tenant(name=clean_name, slug=sanitized_slug, tier=tier.strip(), is_active=True)
    db.add(tenant)
    try:
        db.commit()
        db.refresh(tenant)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tenant: {exc}",
        )
    return tenant
