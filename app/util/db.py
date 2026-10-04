import asyncio
import sqlite3
from collections.abc import Callable
from typing import cast

from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, text

from app.internal.env_settings import Settings
from app.util.log import logger

db = Settings().db
if db.use_postgres:
    engine = create_engine(
        f"postgresql://{db.postgres_user}:{db.postgres_password}@{db.postgres_host}:{db.postgres_port}/{db.postgres_db}?sslmode={db.postgres_ssl_mode}"
    )
else:
    sqlite_path = Settings().get_sqlite_path()
    engine = create_engine(f"sqlite+pysqlite:///{sqlite_path}")

    @event.listens_for(engine, "connect")
    def _sqlite_on_connect(  # pyright: ignore[reportUnusedFunction]
        dbapi_connection: sqlite3.Connection, _connection_record: object
    ):
        # WAL lets readers proceed while another session holds the write lock,
        # which is common since requests share the database across awaits.
        _ = dbapi_connection.execute("PRAGMA journal_mode=WAL")


def get_session():
    with Session(engine) as session:
        if not Settings().db.use_postgres:
            session.execute(text("PRAGMA foreign_keys=ON"))
        yield session


_DEFAULT_BUSY_TIMEOUT_MS = 5000  # Python's sqlite3 default
_RETRY_BUSY_TIMEOUT_MS = 100


def _is_locked_error(e: OperationalError) -> bool:
    return "database is locked" in str(e.orig)


async def commit_with_retry(
    session: Session,
    apply: Callable[[], None],
    attempts: int = 5,
    delay: float = 1.0,
) -> bool:
    """
    Run `apply` to stage changes on the session and commit them, retrying if
    SQLite reports the database is locked.

    SQLite's busy timeout blocks the thread, and since the lock is usually held
    by another coroutine on this same event loop, blocking can never let it
    finish. Sleeping with asyncio yields to that coroutine instead.

    A rollback discards the staged changes, so `apply` is run again before
    every attempt. Returns False if every attempt failed.
    """
    use_sqlite = not Settings().db.use_postgres
    for attempt in range(1, attempts + 1):
        raw: sqlite3.Connection | None = None
        try:
            if use_sqlite:
                # Fail fast instead of blocking the event loop in a busy wait.
                # Restored in `finally` on the same DBAPI connection, since the
                # session may have released it back to the pool by then.
                raw = cast(
                    sqlite3.Connection,
                    cast(object, session.connection().connection.dbapi_connection),
                )
                _ = raw.execute(f"PRAGMA busy_timeout={_RETRY_BUSY_TIMEOUT_MS}")
            apply()
            session.commit()
            return True
        except OperationalError as e:
            session.rollback()
            if not _is_locked_error(e) or attempt == attempts:
                logger.error(
                    "Failed to commit to database",
                    attempt=attempt,
                    error=str(e.orig),
                )
                return False
            logger.warning("Database is locked, retrying commit", attempt=attempt)
            await asyncio.sleep(delay * attempt)
        finally:
            if raw is not None:
                _ = raw.execute(f"PRAGMA busy_timeout={_DEFAULT_BUSY_TIMEOUT_MS}")
    return False
