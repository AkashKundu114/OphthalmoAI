from __future__ import annotations

import logging
import os
import sys
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool, QueuePool, StaticPool

from backend.db import create_db_engine
from backend.db_utils import get_database_type, verify_database_connection


def test_get_database_type_variants():
    """Verify get_database_type accurately categorizes connection string dialects."""
    assert get_database_type("sqlite:///./ophthalmoai.db") == "sqlite"
    assert get_database_type("sqlite:///:memory:") == "sqlite"
    assert get_database_type("sqlite") == "sqlite"

    assert get_database_type("postgresql://ophthalmo:password@localhost:5432/ophthalmoai") == "postgresql"
    assert get_database_type("postgresql+psycopg://ophthalmo:password@localhost:5432/ophthalmoai") == "postgresql"
    assert get_database_type("postgresql+psycopg2://user:pass@host:5432/db") == "postgresql"
    assert get_database_type("postgres://user:pass@host:5432/db") == "postgresql"

    assert (
        get_database_type(
            "mssql+pyodbc://sa:password@localhost/ophthalmoai?driver=ODBC+Driver+18+for+SQL+Server"
        )
        == "mssql"
    )
    assert get_database_type("mssql+pymssql://user:pass@host/db") == "mssql"
    assert get_database_type("mssql://user:pass@host/db") == "mssql"


def test_sqlite_engine_creation_and_warning(caplog):
    """Verify SQLite engine has no pooling (NullPool) and emits production warning."""
    with caplog.at_level(logging.WARNING):
        eng = create_db_engine("sqlite:///./test_ophthalmo.db")

    assert eng.dialect.name == "sqlite"
    assert isinstance(eng.pool, NullPool)
    assert "WARNING: Using SQLite for development. Set DATABASE_URL for production." in caplog.text


def test_sqlite_memory_engine_uses_static_pool():
    """Verify in-memory SQLite uses StaticPool to preserve session state across calls."""
    eng = create_db_engine("sqlite:///:memory:")
    assert eng.dialect.name == "sqlite"
    assert isinstance(eng.pool, StaticPool)


def test_postgresql_engine_configuration():
    """Verify PostgreSQL engine is configured with pool_size=10, max_overflow=20, pool_pre_ping=True."""
    eng = create_db_engine("postgresql+psycopg://ophthalmo:password@localhost:5432/ophthalmoai")

    assert eng.dialect.name == "postgresql"
    assert isinstance(eng.pool, QueuePool)
    assert eng.pool.size() == 10
    assert eng.pool._max_overflow == 20
    assert eng.pool._pre_ping is True


def test_postgresql_url_auto_normalization_psycopg3():
    """Verify postgresql:// automatically normalizes to postgresql+psycopg:// when psycopg 3 is present."""
    eng = create_db_engine("postgresql://ophthalmo:password@localhost:5432/ophthalmoai")

    assert eng.dialect.name == "postgresql"
    assert eng.driver == "psycopg"
    assert eng.pool.size() == 10
    assert eng.pool._max_overflow == 20
    assert eng.pool._pre_ping is True


def test_mssql_engine_configuration():
    """Verify MS SQL Server engine is configured with pool_size=10, max_overflow=20, pool_pre_ping=True."""
    # Ensure pyodbc can be resolved by SQLAlchemy dialect
    if "pyodbc" not in sys.modules:
        mock_pyodbc = MagicMock()
        mock_pyodbc.version = "5.0.0"
        mock_pyodbc.paramstyle = "qmark"
        sys.modules["pyodbc"] = mock_pyodbc

    conn_str = "mssql+pyodbc://sa:password@localhost/ophthalmoai?driver=ODBC+Driver+18+for+SQL+Server"
    eng = create_db_engine(conn_str)

    assert eng.dialect.name == "mssql"
    assert isinstance(eng.pool, QueuePool)
    assert eng.pool.size() == 10
    assert eng.pool._max_overflow == 20
    assert eng.pool._pre_ping is True


def test_verify_database_connection_sqlite(caplog):
    """Verify verify_database_connection pings SQLite and logs backend type and version."""
    mem_engine = create_engine("sqlite:///:memory:")
    with caplog.at_level(logging.INFO):
        res = verify_database_connection(mem_engine)

    assert res["status"] == "connected"
    assert res["backend_type"] == "sqlite"
    assert "version" in res and len(res["version"]) > 0
    assert "Database connection verified successfully" in caplog.text


def test_verify_database_connection_failure():
    """Verify verify_database_connection handles unreachable engines gracefully."""
    mock_engine = MagicMock()
    mock_engine.url = "postgresql+psycopg://baduser:badpass@127.0.0.1:5432/baddb"
    mock_engine.connect.side_effect = ConnectionRefusedError("Database host unreachable")

    res = verify_database_connection(mock_engine)

    assert res["status"] == "error"
    assert res["backend_type"] == "postgresql"
    assert "Database host unreachable" in res["error"]
