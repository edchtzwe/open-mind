"""Test fixtures: one explicit store, destroyed and reseeded per test.

`db.connect()` with no argument falls back to `[store] path`, then to
`./openmind.db`. A test that forgot its argument would therefore write to the
live store. The `store` fixture never inherits that default: it resolves one of
two places and refuses anything else, loudly.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from openmind import db
from openmind.config import get
from seed import seed

TEST_DB_ENV = "OPENMIND_TEST_DB"
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DEFAULT_TEST_DB = TESTS_DIR / "openmind-test.db"


@pytest.fixture(autouse=True)
def clean_queue():
    """The Huey queue file is repo-global: any test that writes a document
    enqueues a real task. Empty before and after, so no run leaks pending
    tasks into the next one — same contract as the destroyed test store."""
    from openmind import worker

    worker.huey.flush()
    yield
    worker.huey.flush()


def _refuse(message: str) -> None:
    """Abort the run. Destroying the store is unconditional, so a doubtful
    path is a stop, never a warning-and-continue."""
    pytest.exit(
        "\n\n  TEST STORE REFUSED\n"
        f"  {message}\n\n"
        "  Nothing was deleted.\n",
        returncode=2,
    )


def _live_store_paths() -> set[Path]:
    """The paths a test must never touch: the store the server runs on."""
    live = {(REPO_ROOT / db.DEFAULT_PATH).resolve()}
    configured = get("store", "path")
    if configured:
        live.add((REPO_ROOT / Path(str(configured))).resolve())
    return live


def resolve_test_db() -> Path:
    """`OPENMIND_TEST_DB` if set, else `tests/openmind-test.db`."""
    raw = os.environ.get(TEST_DB_ENV, "").strip()
    path = Path(raw).expanduser().resolve() if raw else DEFAULT_TEST_DB.resolve()

    # Unconditional deletion makes naming a safety control, not a style choice.
    if "test" not in path.name.lower():
        _refuse(
            f"{TEST_DB_ENV} points at {path}, whose name says nothing about being\n"
            "  disposable. The file and its journals are deleted before every test.\n"
            f"  Fix: rename it so the filename contains 'test' (e.g. {path.parent}/openmind-test.db)\n"
            f"  or unset {TEST_DB_ENV}."
        )

    if path in _live_store_paths():
        _refuse(
            f"{path} is the live store (db.DEFAULT_PATH or [store] path from config.toml).\n"
            f"  Fix: point {TEST_DB_ENV} somewhere else."
        )

    if path.is_dir():
        _refuse(f"{path} is a directory, not a database file.")

    return path


def _destroy(path: Path) -> None:
    """Delete the database and both journals.

    Order matters: `-wal` holds committed pages that are not yet in the main
    file, so deleting the database first lets a stale WAL resurrect rows on the
    next open. Journals go first, and the caller has already closed any handle.
    """
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    path.unlink(missing_ok=True)


def pytest_report_header(config) -> str:
    return f"test store: {resolve_test_db()}"


@pytest.fixture()
def store() -> sqlite3.Connection:
    """A bootstrapped, seeded store that no other test can see.

    Destroyed at the start of each test rather than the end: a run that crashed
    half way through cannot leave state behind, and the last store stays on disk
    for inspection.
    """
    path = resolve_test_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    _destroy(path)

    conn = db.connect(path)
    try:
        assert db.bootstrap(conn) is True
        seed(conn)
        yield conn
    finally:
        conn.close()
