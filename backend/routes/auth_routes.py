"""Authentication & Access Control Route Handlers."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import (
    authenticate_user,
    create_access_token,
    hash_password,
    oauth2_scheme,
    require_role,
    revoke_token,
)
from backend.config import AUTH_RATE_LIMIT
from backend.db import AuditLog, User
from backend.db_async import get_async_db
from backend.logging_config import get_logger
from backend.security import anonymise_ip, make_rate_limit_decorator
from backend.validators import validate_email, validate_password_strength

logger = get_logger("ophthalmoai.auth_routes")
router = APIRouter(prefix="/auth", tags=["Authentication"])

_auth_limit = make_rate_limit_decorator(AUTH_RATE_LIMIT)


def _client_ip(request: Request) -> str | None:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else None


async def _log_audit_async(
    db: AsyncSession,
    action: str,
    success: bool = True,
    user_id: str | None = None,
    ip_address: str | None = None,
    error_detail: str | None = None,
) -> None:
    log_fn = logger.info if success else logger.warning
    log_fn("audit.event", action=action, success=success, user_id=user_id, ip=ip_address)
    try:
        entry = AuditLog(
            action=action,
            success=success,
            user_id=user_id,
            ip_address=ip_address,
            error_detail=error_detail,
        )
        db.add(entry)
        await db.commit()
    except Exception as exc:
        logger.error("audit.db_write_failed", action=action, error=str(exc))
        await db.rollback()


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: str


@router.post("/register", response_model=TokenResponse, status_code=201)
@_auth_limit
async def register(
    request: Request,
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_async_db),
):
    email_ok, email_or_err = validate_email(payload.email)
    if not email_ok:
        raise HTTPException(422, detail=email_or_err)

    pw_ok, pw_err = validate_password_strength(payload.password)
    if not pw_ok:
        raise HTTPException(422, detail=pw_err)

    email = email_or_err
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(409, detail="An account with this email already exists.")

    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role="patient",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(subject=user.id, role=user.role)
    await _log_audit_async(
        db,
        "register",
        success=True,
        user_id=user.id,
        ip_address=anonymise_ip(_client_ip(request)),
    )
    return TokenResponse(access_token=token, role=user.role, user_id=user.id)


@router.post("/token", response_model=TokenResponse)
@_auth_limit
async def login(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    email: str = Form(...),
    password: str = Form(...),
):
    client_ip = anonymise_ip(_client_ip(request))

    user, error = await authenticate_user(db, email, password)
    if error or not user:
        await _log_audit_async(
            db,
            "login",
            success=False,
            ip_address=client_ip,
            error_detail=error or "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error or "Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    token = create_access_token(subject=user.id, role=user.role)
    await _log_audit_async(db, "login", success=True, user_id=user.id, ip_address=client_ip)
    return TokenResponse(access_token=token, role=user.role, user_id=user.id)


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
):
    if token:
        revoke_token(token)
    return Response(status_code=204)


@router.get("/me")
async def get_me(current_user: User = Depends(require_role("patient", "clinician", "admin"))):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "created_at": current_user.created_at,
    }
