
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
