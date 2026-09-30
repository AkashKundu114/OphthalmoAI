import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import Base, DATABASE_URL
from backend.db_utils import get_database_type

config = context.config

# Read DATABASE_URL from environment variable with fallback to backend.db.DATABASE_URL
database_url = os.getenv("DATABASE_URL", DATABASE_URL)

# Normalize postgresql:// to postgresql+psycopg:// if psycopg3 is installed without psycopg2
if database_url.startswith("postgresql://") or database_url.startswith("postgres://"):
    try:
        import psycopg  # noqa: F401
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
            database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
    except ImportError:
        pass

config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    is_sqlite = get_database_type(url) == "sqlite"
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=is_sqlite,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    url = config.get_main_option("sqlalchemy.url")
    is_sqlite = get_database_type(url) == "sqlite"

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=is_sqlite,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
