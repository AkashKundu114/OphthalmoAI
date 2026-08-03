"""
Async database engine/session, additive to backend/db.py.

backend/db.py's engine, SessionLocal, and get_db() are synchronous SQLAlchemy and are left
untouched here — audit.py, model_registry.py, and every other module that already depends on
the sync session keep working exactly as before. This module exists solely so that
backend/auth.py's request-hot-path functions (get_current_user, require_role, authenticate_user)
can use a real async engine instead of blocking the event loop on every login/token check.

Driver note: DATABASE_URL as written in db.py ("sqlite:///./ophthalmoai.db",
"postgresql://...") is a *sync* DBAPI URL. Async SQLAlchemy needs an async driver:
  - sqlite:///           -> sqlite+aiosqlite:///
  - postgresql://        -> postgresql+asyncpg://
_to_async_url() below does that translation automatically so operators don't need to
maintain two separate DATABASE_URL-shaped env vars. Requires `aiosqlite` (dev) and/or
`asyncpg` (prod) to be installed - neither is currently in backend/requirements.txt,
see the note added there.
"""
from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .db import DATABASE_URL


def _to_async_url(sync_url: str) -> str:
    if sync_url.startswith("sqlite:///"):
        return sync_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if sync_url.startswith("postgresql+psycopg2://"):
        return sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    # Already async, or a driver we don't special-case — pass through unchanged.
    return sync_url


ASYNC_DATABASE_URL = os.getenv("ASYNC_DATABASE_URL", _to_async_url(DATABASE_URL))

_connect_args = {"check_same_thread": False} if ASYNC_DATABASE_URL.startswith("sqlite") else {}

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    connect_args=_connect_args,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
