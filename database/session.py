
"""
Kerala Flood Relief Coordination Agent
File: backend/database/session.py

Async SQLAlchemy 2.x database session management.

Features:
- PostgreSQL with asyncpg
- Async database engine
- Connection pooling
- Async session factory
- FastAPI dependency injection
- Automatic transaction rollback
- Explicit transaction management
- Database health checks
- Connection retry support
- Database initialization
- Graceful engine shutdown
- Alembic-compatible metadata
- SQLAlchemy ORM integration

Dependencies:
    pip install sqlalchemy>=2.0 asyncpg pydantic-settings

Environment:
    DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/kerala_flood_ai

IMPORTANT:
- Use Alembic migrations in production.
- Do not use create_all() as a production migration system.
- Never expose database credentials in logs.
- Never share one AsyncSession between concurrent tasks.
"""


import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, AsyncIterator
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import (
    DBAPIError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.database.base import (
    Base,
    get_metadata,
    import_all_models,
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DatabaseSettings:
    database_url: str

    echo_sql: bool = False

    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 1800

    connect_timeout: int = 10

    health_check_timeout: float = 5.0

    retry_attempts: int = 3
    retry_delay_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is required"
            )

        return cls(
            database_url=database_url,
            echo_sql=os.getenv(
                "DATABASE_ECHO",
                "false",
            ).lower() == "true",
            pool_size=int(
                os.getenv("DB_POOL_SIZE", "10")
            ),
            max_overflow=int(
                os.getenv("DB_MAX_OVERFLOW", "20")
            ),
            pool_timeout=int(
                os.getenv("DB_POOL_TIMEOUT", "30")
            ),
            pool_recycle=int(
                os.getenv("DB_POOL_RECYCLE", "1800")
            ),
            connect_timeout=int(
                os.getenv("DB_CONNECT_TIMEOUT", "10")
            ),
            health_check_timeout=float(
                os.getenv("DB_HEALTH_TIMEOUT", "5")
            ),
            retry_attempts=int(
                os.getenv("DB_RETRY_ATTEMPTS", "3")
            ),
            retry_delay_seconds=float(
                os.getenv("DB_RETRY_DELAY", "2")
            ),
        )

    def __post_init__(self) -> None:
        if self.pool_size < 1:
            raise ValueError(
                "pool_size must be at least 1"
            )

        if self.max_overflow < 0:
            raise ValueError(
                "max_overflow cannot be negative"
            )

        if self.pool_timeout <= 0:
            raise ValueError(
                "pool_timeout must be positive"
            )

        if self.connect_timeout <= 0:
            raise ValueError(
                "connect_timeout must be positive"
            )

        if self.retry_attempts < 1:
            raise ValueError(
                "retry_attempts must be at least 1"
            )

        if self.retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds cannot be negative"
            )


# ============================================================
# DATABASE URL NORMALIZATION
# ============================================================

def normalize_database_url(url: str) -> str:
    """
    Normalize PostgreSQL connection URLs for asyncpg.

    Examples:
        postgres://... -> postgresql+asyncpg://...
        postgresql://... -> postgresql+asyncpg://...
    """

    if url.startswith("postgres://"):
        return url.replace(
            "postgres://",
            "postgresql+asyncpg://",
            1,
        )

    if url.startswith("postgresql://"):
        return url.replace(
            "postgresql://",
            "postgresql+asyncpg://",
            1,
        )

    return url


def validate_database_url(url: str) -> str:
    normalized = normalize_database_url(url)

    parsed = make_url(normalized)

    if parsed.drivername != "postgresql+asyncpg":
        raise ValueError(
            "Only PostgreSQL with asyncpg is supported"
        )

    if not parsed.database:
        raise ValueError(
            "Database name is missing"
        )

    return normalized


# ============================================================
# DATABASE SETTINGS
# ============================================================

database_settings = DatabaseSettings.from_env()

DATABASE_URL = validate_database_url(
    database_settings.database_url
)


# ============================================================
# ASYNC DATABASE ENGINE
# ============================================================

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=database_settings.echo_sql,
    pool_size=database_settings.pool_size,
    max_overflow=database_settings.max_overflow,
    pool_timeout=database_settings.pool_timeout,
    pool_recycle=database_settings.pool_recycle,
    pool_pre_ping=True,
    connect_args={
        "timeout": database_settings.connect_timeout,
        "command_timeout": 60,
        "server_settings": {
            "application_name": "kerala_flood_relief_ai",
            "timezone": "UTC",
        },
    },
)


# ============================================================
# ASYNC SESSION FACTORY
# ============================================================

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,
)


# ============================================================
# FASTAPI DATABASE DEPENDENCY
# ============================================================

async def get_db() -> AsyncGenerator[
    AsyncSession,
    None,
]:
    """
    FastAPI database dependency.

    Example:

        @router.get("/districts")
        async def districts(
            db: AsyncSession = Depends(get_db)
        ):
            ...

    Each request receives an independent session.

    Successful read/write operations are committed when
    the dependency completes normally.

    Exceptions trigger rollback.

    IMPORTANT:
        Do not send a successful response for a write
        before its transaction has committed.
        For critical writes, explicitly commit in the
        service layer before returning success.
    """

    async with AsyncSessionLocal() as session:
        try:
            yield session

            if session.in_transaction():
                await session.commit()

        except Exception:
            if session.in_transaction():
                await session.rollback()

            raise


# ============================================================
# READ-ONLY SESSION DEPENDENCY
# ============================================================

async def get_readonly_db() -> AsyncGenerator[
    AsyncSession,
    None,
]:
    """
    Read-only session dependency.

    PostgreSQL transaction is set to READ ONLY.
    No application writes are committed.
    """

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                text("SET TRANSACTION READ ONLY")
            )

            yield session

        finally:
            if session.in_transaction():
                await session.rollback()


# ============================================================
# EXPLICIT SESSION CONTEXT MANAGER
# ============================================================

@asynccontextmanager
async def get_session() -> AsyncIterator[
    AsyncSession
]:
    """
    Use outside FastAPI dependency injection.

    Example:

        async with get_session() as db:
            result = await db.execute(...)
    """

    async with AsyncSessionLocal() as session:
        try:
            yield session

        except Exception:
            if session.in_transaction():
                await session.rollback()

            raise

        finally:
            if session.in_transaction():
                await session.rollback()


# ============================================================
# TRANSACTION CONTEXT MANAGER
# ============================================================

@asynccontextmanager
async def transaction() -> AsyncIterator[
    AsyncSession
]:
    """
    Automatically commit or rollback one transaction.

    Example:

        async with transaction() as db:
            db.add(new_incident)
            db.add(new_audit_record)

    Both operations commit together.
    """

    async with AsyncSessionLocal() as session:
        async with session.begin():
            yield session


# ============================================================
# TRANSACTION HELPER
# ============================================================

async def commit_transaction(
    session: AsyncSession,
) -> None:
    """
    Commit an existing session transaction.

    Roll back if commit fails.
    """

    try:
        await session.commit()

    except Exception:
        await session.rollback()
        logger.exception(
            "Database transaction commit failed"
        )
        raise


async def rollback_transaction(
    session: AsyncSession,
) -> None:
    """
    Roll back an active transaction.
    """

    if session.in_transaction():
        await session.rollback()


# ============================================================
# DATABASE CONNECTION TEST
# ============================================================

async def check_database_connection() -> bool:
    """
    Return True if PostgreSQL responds successfully.

    Returns False on connection or query failures.
    """

    try:
        async with asyncio.timeout(
            database_settings.health_check_timeout
        ):
            async with engine.connect() as connection:
                result = await connection.execute(
                    text("SELECT 1")
                )

                return result.scalar_one() == 1

    except (
        SQLAlchemyError,
        TimeoutError,
        OSError,
    ):
        logger.warning(
            "Database health check failed"
        )

        return False


# ============================================================
# DATABASE HEALTH DETAILS
# ============================================================

async def get_database_health() -> dict[str, Any]:
    """
    Return safe database health information.

    Never returns database passwords or connection URLs.
    """

    checked_at = datetime.now(timezone.utc)

    try:
        async with asyncio.timeout(
            database_settings.health_check_timeout
        ):
            async with engine.connect() as connection:
                result = await connection.execute(
                    text(
                        """
                        SELECT
                            current_database() AS database_name,
                            current_timestamp AS server_time,
                            version() AS server_version
                        """
                    )
                )

                row = result.mappings().one()

                return {
                    "status": "healthy",
                    "connected": True,
                    "database": row["database_name"],
                    "server_time": (
                        row["server_time"].isoformat()
                    ),
                    "server_version": row[
                        "server_version"
                    ].split(",")[0],
                    "checked_at": checked_at.isoformat(),
                }

    except (
        SQLAlchemyError,
        TimeoutError,
        OSError,
    ):
        logger.warning(
            "Database health details unavailable"
        )

        return {
            "status": "unhealthy",
            "connected": False,
            "database": None,
            "server_time": None,
            "server_version": None,
            "checked_at": checked_at.isoformat(),
        }


# ============================================================
# CONNECTION RETRY
# ============================================================

async def wait_for_database() -> bool:
    """
    Wait for PostgreSQL to become reachable.

    Intended for startup, not individual API requests.
    """

    attempts = database_settings.retry_attempts

    for attempt in range(1, attempts + 1):
        if await check_database_connection():
            logger.info(
                "Database connection established"
            )
            return True

        if attempt < attempts:
            logger.warning(
                "Database unavailable; retry %s/%s",
                attempt,
                attempts,
            )

            await asyncio.sleep(
                database_settings.retry_delay_seconds
            )

    logger.error(
        "Database unavailable after %s attempts",
        attempts,
    )

    return False


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

async def init_db(
    *,
    create_tables: bool = False,
) -> None:
    """
    Initialize database connectivity.

    By default, only checks connectivity.

    create_tables=True is intended for local development
    and isolated tests.

    Production schema changes must use Alembic.
    """

    connected = await wait_for_database()

    if not connected:
        raise RuntimeError(
            "Cannot initialize application: "
            "PostgreSQL is unavailable"
        )

    if create_tables:
        environment = os.getenv(
            "APP_ENV",
            "development",
        ).lower()

        if environment in {
            "production",
            "prod",
        }:
            raise RuntimeError(
                "Automatic table creation is disabled "
                "in production. Use Alembic migrations."
            )

        import_all_models()

        async with engine.begin() as connection:
            await connection.run_sync(
                Base.metadata.create_all
            )

        logger.info(
            "Development database tables initialized"
        )


# ============================================================
# DATABASE SHUTDOWN
# ============================================================

async def close_db() -> None:
    """
    Dispose of all pooled database connections.

    Call during FastAPI application shutdown.
    """

    await engine.dispose()

    logger.info(
        "Database engine disposed"
    )


# ============================================================
# DATABASE LIFESPAN
# ============================================================

@asynccontextmanager
async def database_lifespan(
    app: Any,
) -> AsyncIterator[None]:
    """
    Optional FastAPI lifespan context manager.

    Example:

        app = FastAPI(
            lifespan=database_lifespan
        )

    If the application already has a lifespan manager,
    call init_db() and close_db() inside that manager
    instead of replacing it.
    """

    await init_db(
        create_tables=False
    )

    try:
        yield

    finally:
        await close_db()


# ============================================================
# DATABASE INFORMATION
# ============================================================

def get_database_info() -> dict[str, Any]:
    """
    Return non-sensitive database configuration.
    """

    parsed = make_url(DATABASE_URL)

    return {
        "driver": parsed.drivername,
        "host": parsed.host,
        "port": parsed.port or 5432,
        "database": parsed.database,
        "pool_size": database_settings.pool_size,
        "max_overflow": database_settings.max_overflow,
        "pool_timeout": database_settings.pool_timeout,
        "pool_recycle": database_settings.pool_recycle,
    }


# ============================================================
# CONNECTION POOL INFORMATION
# ============================================================

def get_pool_status() -> dict[str, Any]:
    """
    Return connection pool statistics.

    Values describe this application process only.
    """

    pool = engine.sync_engine.pool

    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "status": pool.status(),
    }


# ============================================================
# SIMPLE DATABASE QUERY
# ============================================================

async def execute_scalar(
    query: str,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """
    Execute a trusted SQL query and return one scalar.

    Only use static, developer-controlled SQL.

    Never interpolate untrusted user input into query text.
    Pass user values through bound parameters.
    """

    async with get_session() as session:
        result = await session.execute(
            text(query),
            parameters or {},
        )

        return result.scalar_one_or_none()


# ============================================================
# ALEMBIC METADATA
# ============================================================

def get_target_metadata():
    """
    Return ORM metadata for Alembic.

    All model modules must be importable and registered.
    """

    import_all_models()

    return get_metadata()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "DatabaseSettings",
    "database_settings",
    "DATABASE_URL",
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "get_readonly_db",
    "get_session",
    "transaction",
    "commit_transaction",
    "rollback_transaction",
    "check_database_connection",
    "get_database_health",
    "wait_for_database",
    "init_db",
    "close_db",
    "database_lifespan",
    "get_database_info",
    "get_pool_status",
    "execute_scalar",
    "get_target_metadata",
    "normalize_database_url",
    "validate_database_url",
]
