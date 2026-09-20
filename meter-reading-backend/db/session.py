"""
db/session.py

Connects db/models.py's table definitions to an actual database
connection. models.py itself never imports this file -- the dependency
only goes one direction (this file imports Base from models.py), which
is why the sqlite smoke test used to verify models.py worked without
ever touching this file or a real Postgres connection.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

# Reads backend/.env (DATABASE_URL, etc.) and loads it into os.environ --
# without this, values written in .env are just text in a file that
# Python never actually sees. Must run BEFORE the os.environ.get(...)
# call just below, or DATABASE_URL would still fall back to the default.
load_dotenv()

# TODO: move to .env once that's wired up. Expected format for Supabase:
# postgresql://postgres:[password]@[project-ref].supabase.co:5432/postgres
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/meter_reading",
)

# The engine manages a small pool of actual network connections to
# Postgres and reuses them -- creating a fresh TCP connection for every
# single query would be slow. pool_pre_ping checks a connection is still
# alive before handing it out, since Supabase (and most hosted Postgres)
# can silently drop idle connections after a while.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# A factory for creating new sessions, all configured the same way.
# This does NOT create a session itself -- it creates something that
# creates sessions, on demand, one per call. See get_session() below for
# where that actually happens.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """
    Creates every table defined in db/models.py, if it doesn't already
    exist. Safe to call more than once -- existing tables are left alone.

    Fine for local dev. Once real data exists in these tables, further
    schema CHANGES should go through Alembic migrations instead of this --
    this function only ever adds missing tables, it never alters ones
    that already exist (so a renamed/added column here wouldn't actually
    update a table that was already created).
    """
    Base.metadata.create_all(engine)


@contextmanager
def get_session():
    """
    Hands out exactly ONE session, and guarantees it gets closed
    afterward -- even if the code using it raises an exception.

    - On success: commits automatically before closing.
    - On any exception: rolls back (undoes any half-finished changes)
      before closing, and re-raises the same exception so the caller
      still sees what went wrong.

    This is a context manager specifically so both FastAPI request
    handlers AND Celery task functions can use the exact same pattern:

        with get_session() as session:
            job = session.get(Job, job_id)
            job.status = "done"
            # no explicit commit needed -- happens automatically on
            # a clean exit from this `with` block

    A fresh session per request/task is what keeps concurrent requests
    from interfering with each other -- see this file's module docstring
    reasoning, or just: never share one session across two requests/tasks.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()