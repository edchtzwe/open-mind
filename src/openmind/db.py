"""SQLite bootstrap and connection handling for open-mind.

One place opens SQLite (see RULES.md, "one obvious way"): tools never build
their own connection. The schema ships inside the package, so an installed
wheel bootstraps without the repo checkout.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterator

from openmind.config import get

SCHEMA_VERSION = 1
DEFAULT_PATH = "openmind.db"


def read_schema() -> str:
    """The packaged schema, as text."""
    return (
        resources.files("openmind")
        .joinpath("schema.sql")
        .read_text(encoding="utf-8")
    )


def utcnow() -> str:
    """Now, ISO-8601 UTC to the second with a `Z` suffix.

    One clock for every write. Because the format sorts lexically, `since`
    filters and `ORDER BY created_at` work as plain string comparison — no
    parsing, no timezone drift between rows.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the store with the pragmas every caller needs.

    `path` falls back to `[store] path` in config.toml, then to a file beside
    the process. Foreign keys are off in SQLite by default; the schema's
    integrity depends on them, so they are switched on at every connection.
    """
    resolved = Path(path or get("store", "path") or DEFAULT_PATH)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def bootstrap(conn: sqlite3.Connection) -> bool:
    """Apply the schema when the database is new. Returns True if it ran.

    Version-gated instead of `IF NOT EXISTS`: one transaction, one stamp, so a
    half-applied schema cannot exist. Re-running against a current database is
    a no-op, which makes it safe to call on every startup.
    """
    applied = conn.execute("PRAGMA user_version").fetchone()[0]
    if applied >= SCHEMA_VERSION:
        return False

    # executescript() commits any open transaction and runs the script bare, so
    # the BEGIN/COMMIT here is what makes the whole schema atomic.
    conn.executescript(f"BEGIN;\n{read_schema()}\nCOMMIT;")
    return True


@contextmanager
def connection(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open, bootstrap, and close in one step — the read-side entry point.

    Writers own their transaction lifetime, so they call `connect()` directly;
    readers do not, and closing in a `finally` is their whole contract. Both
    still go through `connect()`, so there is one place that opens SQLite.
    """
    conn = connect(path)
    try:
        bootstrap(conn)
        yield conn
    finally:
        conn.close()
