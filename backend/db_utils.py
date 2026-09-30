from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)


def get_database_type(database_url: Optional[str] = None) -> str:
    """Return 'sqlite', 'postgresql', or 'mssql' based on the database connection string."""
    url = database_url
    if url is None:
        url = os.getenv("DATABASE_URL", "sqlite:///./ophthalmoai.db")

    url_clean = str(url).strip().lower()
    if url_clean.startswith("sqlite") or url_clean == "sqlite":
        return "sqlite"
    elif url_clean.startswith("postgresql") or url_clean.startswith("postgres"):
        return "postgresql"
    elif url_clean.startswith("mssql") or url_clean.startswith("sqlserver"):
        return "mssql"
    elif "postgres" in url_clean:
        return "postgresql"
    elif "mssql" in url_clean:
        return "mssql"
    elif "sqlite" in url_clean:
        return "sqlite"
    return "sqlite"


def verify_database_connection(engine_or_url: Optional[Union[Engine, str]] = None) -> Dict[str, Any]:
    """Ping the database and log the backend type and version.

    Returns:
        Dict[str, Any] with 'status', 'backend_type', 'version', and 'url'.
    """
    from .db import create_db_engine, engine as default_engine

    if engine_or_url is None:
        target_engine = default_engine
    elif isinstance(engine_or_url, str):
        target_engine = create_db_engine(engine_or_url)
    else:
        target_engine = engine_or_url

    backend_type = get_database_type(str(target_engine.url))

    try:
        with target_engine.connect() as conn:
            if backend_type == "sqlite":
                version_query = text("SELECT sqlite_version()")
            elif backend_type == "postgresql":
                version_query = text("SELECT version()")
            elif backend_type == "mssql":
                version_query = text("SELECT @@VERSION")
            else:
                version_query = text("SELECT 1")

            result = conn.execute(version_query).scalar()
            version_str = str(result) if result is not None else "unknown"

            logger.info(
                "Database connection verified successfully. Backend: %s, Version: %s",
                backend_type,
                version_str,
            )
            return {
                "status": "connected",
                "backend_type": backend_type,
                "version": version_str,
                "url": str(target_engine.url),
            }
    except Exception as exc:
        logger.error(
            "Database connection verification failed for backend '%s': %s",
            backend_type,
            str(exc),
        )
        return {
            "status": "error",
            "backend_type": backend_type,
            "error": str(exc),
            "url": str(target_engine.url),
        }
