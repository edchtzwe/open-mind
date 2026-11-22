"""The write path: one transaction per attempt, retried only when transient.

`write_document` is not five writes that happen to run together. It is one
request validated at the boundary, then committed as a single transaction:
the document, its sections, its progress record, the root log entry and the
cross-reference edges all land or none of them do. A rolled-back write leaves
no trace in the store by design, which is why failures go to telemetry
(stderr/JSON) instead of the log document.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import unicodedata
import uuid
from typing import Any

from openmind import db, telemetry
from openmind.config import get

# SQLite errors that a second attempt can survive. Anything else is
# deterministic and fails on the first attempt: retrying cannot help and
# costs the caller the backoff for nothing.
_TRANSIENT_MESSAGES = (
    "database is locked",
    "database table is locked",
    "disk i/o error",
    "disk full",
    "unable to open database file",
)

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

_CREATE_DOCUMENT = """
    INSERT INTO document (wiki_id, type_id, title, created_at, updated_at)
    VALUES (
        :wiki_id,
        (SELECT id FROM document_type WHERE name = 'document'),
        :title, :now, :now
    )
"""

_INSERT_PROGRESS = """
    INSERT INTO document (wiki_id, type_id, title, created_at, updated_at)
    VALUES (
        :wiki_id,
        (SELECT id FROM document_type WHERE name = 'progress'),
        :title, :now, :now
    )
"""

_INSERT_SECTION = """
    INSERT INTO document_content_section
        (document_id, position, heading, body, created_at, updated_at)
    VALUES (:document_id, :position, :heading, :body, :now, :now)
"""

_INSERT_EDGE = """
    INSERT OR IGNORE INTO document_cross_references
        (source_document_id, target_document_id, created_at)
    VALUES (:source, :target, :now)
"""


class WriteRejected(Exception):
    """Deterministic failure — the same request cannot succeed on a retry."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


class WriteFailed(Exception):
    """Retries exhausted. Carries every attempt's error so the model can act."""

    def __init__(self, attempts: int, errors: list[str]) -> None:
        self.attempts = attempts
        self.errors = errors
        super().__init__(f"write failed after {attempts} attempt(s): " + "; ".join(errors))


# --- wiki creation --------------------------------------------------------


def validate_wiki_name(value: Any, field: str = "name") -> str:
    """Normalise the name; casing is preserved exactly as given.

    NFKC + trim fix the invisible variants (combining marks, stray spaces);
    control characters are rejected outright. Matching elsewhere is
    `COLLATE NOCASE`, so `Open-Mind` and `open-mind` cannot coexist.
    """
    name = unicodedata.normalize("NFKC", _text(value, field))
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name):
        raise WriteRejected(field, "must not contain control characters")
    return name


def _duplicate_wiki(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM wiki WHERE name = :name COLLATE NOCASE LIMIT 1",
            {"name": name},
        ).fetchone()
        is not None
    )


def _commit_wiki(name: str) -> dict[str, Any]:
    """One attempt: one wiki row plus one root log entry. Caller owns retries."""
    conn = db.connect()
    conn.isolation_level = None  # explicit BEGIN IMMEDIATE, explicit COMMIT
    try:
        db.bootstrap(conn)
        conn.execute("BEGIN IMMEDIATE")
        now = db.utcnow()
        if _duplicate_wiki(conn, name):
            raise WriteRejected("name", f"a live wiki named {name!r} already exists")
        wiki_id = _insert_wiki_row(conn, name, now)
        log_id = _root_log(conn, now)
        log_position = _append_log(conn, log_id, f"Created wiki {name!r}.", now)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return {
        "wiki": name,
        "wiki_id": wiki_id,
        "log_document_id": log_id,
        "log_position": log_position,
    }


async def create_wiki(request: Any, *, write_id: str | None = None) -> dict[str, Any]:
    """Validate once, then commit one wiki plus its root log entry, retried
    only when transient. Raises on failure."""
    name = validate_wiki_name(request)
    write_id = write_id or uuid.uuid4().hex[:12]
    attempts = _attempts()
    errors: list[str] = []

    for attempt in range(1, attempts + 1):
        try:
            result = _commit_wiki(name)
            result["write_id"] = write_id
            result["attempts"] = attempt
            telemetry.log_event(
                "create_wiki.committed",
                write_id=write_id,
                tool="create_wiki",
                wiki=name,
                attempt=attempt,
            )
            return result
        except WriteRejected:
            raise
        except Exception as exc:  # noqa: BLE001 — classified below
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            transient = _is_transient(exc)
            telemetry.log_event(
                "create_wiki.retry" if transient and attempt < attempts else "create_wiki.failed",
                level=logging.WARNING,
                write_id=write_id,
                tool="create_wiki",
                wiki=name,
                attempt=attempt,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            if not transient or attempt == attempts:
                raise WriteFailed(attempt, errors) from exc
            await asyncio.sleep(_backoff())
    raise WriteFailed(attempts, errors)  # unreachable; keeps the type honest


# --- boundary validation ---------------------------------------------------


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise WriteRejected(field, "must be a string")
    text = value.strip()
    if not text and not allow_empty:
        raise WriteRejected(field, "must not be empty")
    return text


def _sections(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise WriteRejected(field, "must be a non-empty array")
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(value):
        where = f"{field}[{index}]"
        if not isinstance(raw, dict):
            raise WriteRejected(where, "must be an object")
        position = raw.get("position")
        if not isinstance(position, int) or isinstance(position, bool) or position < 1:
            raise WriteRejected(f"{where}.position", "must be an integer >= 1")
        if position in seen:
            raise WriteRejected(f"{where}.position", f"duplicate position {position}")
        seen.add(position)
        out.append(
            {
                "position": position,
                "heading": _text(raw.get("heading"), f"{where}.heading", allow_empty=True),
                "body": _text(raw.get("body"), f"{where}.body"),
            }
        )
    return sorted(out, key=lambda section: section["position"])


def _references(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise WriteRejected("references", "must be an array of {wiki, title} objects")
    out: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        where = f"references[{index}]"
        if not isinstance(raw, dict):
            raise WriteRejected(where, "must be an object")
        out.append(
            {
                "wiki": _text(raw.get("wiki"), f"{where}.wiki"),
                "title": _text(raw.get("title"), f"{where}.title"),
            }
        )
    return out


def _wiki_spec(value: Any) -> dict[str, Any]:
    """A wiki is a name string, or {"id", "title"}.

    A string resolves by name (case-insensitive, must exist). In the object
    form, `id: null` plus a title means "create this wiki inside the same
    transaction"; a supplied id must resolve to a live wiki or the write dies.
    """
    if isinstance(value, str):
        return {"id": None, "title": validate_wiki_name(value, "wiki"), "create": False}
    if not isinstance(value, dict):
        raise WriteRejected("wiki", "must be a name string or an {id, title} object")
    wiki_id = value.get("id")
    if wiki_id is not None and (
        not isinstance(wiki_id, int) or isinstance(wiki_id, bool) or wiki_id < 1
    ):
        raise WriteRejected("wiki.id", "must be null or a positive integer")
    title = value.get("title")
    if title is not None:
        title = validate_wiki_name(title, "wiki.title")
    if wiki_id is None and title is None:
        raise WriteRejected("wiki", "give wiki.id for an existing wiki, or wiki.title to create one")
    if wiki_id is not None and title is None:
        raise WriteRejected("wiki.title", "a wiki id must come with the title it was created with")
    return {"id": wiki_id, "title": title, "create": wiki_id is None}


def validate(request: Any) -> dict[str, Any]:
    """Normalise the request. Every rejection names the field to fix."""
    if not isinstance(request, dict):
        raise WriteRejected("request", "must be an object")

    wiki = _wiki_spec(request.get("wiki"))
    document_id = request.get("document_id")
    title = request.get("title")
    if document_id is None and title is None:
        raise WriteRejected("title", "give a title for a new document or a document_id for an edit")
    if document_id is not None:
        if not isinstance(document_id, int) or isinstance(document_id, bool) or document_id < 1:
            raise WriteRejected("document_id", "must be a positive integer")
    if title is not None:
        title = _text(title, "title")

    mode = request.get("mode", "replace_all")
    if mode not in ("replace_all", "replace_positions"):
        raise WriteRejected("mode", "must be 'replace_all' or 'replace_positions'")

    progress = request.get("progress")
    if not isinstance(progress, dict):
        raise WriteRejected("progress", "must be an object with slug and sections")
    slug = _text(progress.get("slug"), "progress.slug")
    if not _SLUG.match(slug):
        raise WriteRejected("progress.slug", "use lowercase letters, digits, _ or -")

    return {
        "wiki": wiki,
        "document_id": document_id,
        "title": title,
        "mode": mode,
        "sections": _sections(request.get("sections"), "sections"),
        "log_entry": _text(request.get("log_entry"), "log_entry"),
        "progress_slug": slug,
        "progress_sections": _sections(progress.get("sections"), "progress.sections"),
        "references": _references(request.get("references")),
    }


# --- resolution inside the transaction -------------------------------------


def _row(conn: sqlite3.Connection, sql: str, params: dict[str, Any], field: str, reason: str):
    row = conn.execute(sql, params).fetchone()
    if row is None:
        raise WriteRejected(field, reason)
    return row


def _resolve_wiki(conn: sqlite3.Connection, name: str) -> int:
    row = _row(
        conn,
        "SELECT id FROM wiki WHERE name = :name COLLATE NOCASE AND deleted_at IS NULL",
        {"name": name},
        "wiki",
        f"no live wiki named {name!r}",
    )
    return row["id"]


def _insert_wiki_row(conn: sqlite3.Connection, name: str, now: str) -> int:
    cursor = conn.execute(
        "INSERT INTO wiki (name, created_at, updated_at) VALUES (:name, :now, :now)",
        {"name": name, "now": now},
    )
    return cursor.lastrowid


def _resolve_wiki_spec(conn: sqlite3.Connection, spec: dict[str, Any], now: str) -> tuple[int, bool]:
    """Return (wiki_id, created). String form resolves by name; an id must
    resolve to a live wiki whose name matches; `create: True` inserts the
    wiki inside this transaction."""
    if spec["create"]:
        if _duplicate_wiki(conn, spec["title"]):
            raise WriteRejected(
                "wiki.id",
                f"a live wiki named {spec['title']!r} already exists; supply its wiki.id",
            )
        return _insert_wiki_row(conn, spec["title"], now), True
    if spec["id"] is not None:
        row = _row(
            conn,
            "SELECT id, name FROM wiki WHERE id = :id AND deleted_at IS NULL",
            {"id": spec["id"]},
            "wiki.id",
            f"no live wiki with id {spec['id']}",
        )
        if row["name"].casefold() != spec["title"].casefold():
            raise WriteRejected(
                "wiki.title",
                f"wiki id {spec['id']} is named {row['name']!r}, not {spec['title']!r}",
            )
        return row["id"], False
    return _resolve_wiki(conn, spec["title"]), False


def _resolve_document(conn, wiki_id: int, request: dict) -> tuple[int, bool]:
    """Return (document_id, created). Edits may not target progress or log."""
    if request["document_id"] is None:
        row = conn.execute(
            "SELECT id FROM document WHERE wiki_id = :wiki AND title = :title AND deleted_at IS NULL",
            {"wiki": wiki_id, "title": request["title"]},
        ).fetchone()
        if row is None:
            cursor = conn.execute(
                _CREATE_DOCUMENT,
                {"wiki_id": wiki_id, "title": request["title"], "now": db.utcnow()},
            )
            return cursor.lastrowid, True
        request["document_id"] = row["id"]

    row = _row(
        conn,
        """
        SELECT d.id, dt.name AS type_name, d.wiki_id
        FROM document d JOIN document_type dt ON dt.id = d.type_id
        WHERE d.id = :id AND d.deleted_at IS NULL
        """,
        {"id": request["document_id"]},
        "document_id",
        f"no live document with id {request['document_id']}",
    )
    if row["wiki_id"] != wiki_id:
        raise WriteRejected("document_id", "that document belongs to a different wiki")
    if row["type_name"] != "document":
        raise WriteRejected(
            "document_id",
            f"target is a {row['type_name']} document; progress is insert-only and log is system-only",
        )
    return row["id"], False


def _write_sections(conn, document_id: int, sections: list[dict], mode: str, now: str) -> None:
    """Position-addressed replacement. Mode decides the blast radius."""
    if mode == "replace_all":
        conn.execute(
            "DELETE FROM document_content_section WHERE document_id = :id",
            {"id": document_id},
        )
    else:
        for section in sections:
            conn.execute(
                "DELETE FROM document_content_section WHERE document_id = :id AND position = :position",
                {"id": document_id, "position": section["position"]},
            )
    for section in sections:
        conn.execute(_INSERT_SECTION, {"document_id": document_id, **section, "now": now})


def _insert_progress(conn, wiki_id: int, request: dict, now: str) -> tuple[int, str]:
    count = conn.execute(
        """
        SELECT COUNT(*) AS n FROM document
        WHERE wiki_id = :wiki
          AND type_id = (SELECT id FROM document_type WHERE name = 'progress')
        """,
        {"wiki": wiki_id},
    ).fetchone()["n"]
    title = f"{count + 1:02d}_{request['progress_slug']}"
    cursor = conn.execute(_INSERT_PROGRESS, {"wiki_id": wiki_id, "title": title, "now": now})
    progress_id = cursor.lastrowid
    for section in request["progress_sections"]:
        conn.execute(_INSERT_SECTION, {"document_id": progress_id, **section, "now": now})
    return progress_id, title


def _root_log(conn, now: str) -> int:
    """The single root log document. System-created; no tool can reach it."""
    row = conn.execute(
        """
        SELECT d.id FROM document d JOIN document_type dt ON dt.id = d.type_id
        WHERE d.wiki_id IS NULL AND dt.name = 'log' AND d.deleted_at IS NULL
        LIMIT 1
        """
    ).fetchone()
    if row is not None:
        return row["id"]
    cursor = conn.execute(
        """
        INSERT INTO document (wiki_id, type_id, title, created_at, updated_at)
        VALUES (NULL, (SELECT id FROM document_type WHERE name = 'log'), 'log', :now, :now)
        """,
        {"now": now},
    )
    return cursor.lastrowid


def _append_log(conn, log_id: int, entry: str, now: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(position), 0) AS top FROM document_content_section WHERE document_id = :id",
        {"id": log_id},
    ).fetchone()
    position = row["top"] + 1
    conn.execute(
        _INSERT_SECTION,
        {"document_id": log_id, "position": position, "heading": "", "body": entry, "now": now},
    )
    return position


def _link(conn, source: int, target: int, now: str) -> None:
    if source == target:
        return
    conn.execute(_INSERT_EDGE, {"source": source, "target": target, "now": now})


def _link_references(conn, document_id: int, wiki_id: int, references: list[dict], now: str) -> int:
    for ref in references:
        target_wiki = _resolve_wiki(conn, ref["wiki"])
        if target_wiki != wiki_id:
            raise WriteRejected(
                f"references[{ref['title']}]",
                f"{ref['wiki']!r} is a different wiki; a reference stays inside its own wiki",
            )
        row = _row(
            conn,
            "SELECT id FROM document WHERE wiki_id = :wiki AND title = :title AND deleted_at IS NULL",
            {"wiki": wiki_id, "title": ref["title"]},
            f"references[{ref['title']}]",
            "no live document with that title in this wiki",
        )
        if row["id"] == document_id:
            raise WriteRejected(f"references[{ref['title']}]", "a document cannot reference itself")
        _link(conn, document_id, row["id"], now)
    return len(references)


# --- the transaction -------------------------------------------------------


def commit(request: dict[str, Any]) -> dict[str, Any]:
    """One attempt. Caller owns retries."""
    conn = db.connect()
    conn.isolation_level = None  # explicit BEGIN IMMEDIATE, explicit COMMIT
    try:
        db.bootstrap(conn)
        conn.execute("BEGIN IMMEDIATE")
        now = db.utcnow()
        wiki_id, wiki_created = _resolve_wiki_spec(conn, request["wiki"], now)
        document_id, created = _resolve_document(conn, wiki_id, request)
        _write_sections(conn, document_id, request["sections"], request["mode"], now)
        progress_id, progress_title = _insert_progress(conn, wiki_id, request, now)
        _link(conn, progress_id, document_id, now)
        _link_references(conn, document_id, wiki_id, request["references"], now)
        log_id = _root_log(conn, now)
        log_position = _append_log(conn, log_id, request["log_entry"], now)
        _link(conn, log_id, progress_id, now)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return {
        "wiki": request["wiki"]["title"],
        "wiki_id": wiki_id,
        "wiki_created": wiki_created,
        "document_id": document_id,
        "created": created,
        "progress_document_id": progress_id,
        "progress_title": progress_title,
        "log_document_id": log_id,
        "log_position": log_position,
    }


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        return any(fragment in message for fragment in _TRANSIENT_MESSAGES)
    return isinstance(exc, (TimeoutError, ConnectionError))


def _attempts() -> int:
    return max(1, int(get("write", "max_document_write_attempts") or 3))


def _backoff() -> float:
    return max(0.0, float(get("write", "retry_backoff_seconds") or 1))


async def write_document(request: Any, *, write_id: str | None = None) -> dict[str, Any]:
    """Validate once, then commit with bounded retries. Raises on failure."""
    request = validate(request)
    write_id = write_id or uuid.uuid4().hex[:12]
    attempts = _attempts()
    errors: list[str] = []

    for attempt in range(1, attempts + 1):
        try:
            result = commit(request)
            result["write_id"] = write_id
            result["attempts"] = attempt
            telemetry.log_event(
                "write_document.committed",
                write_id=write_id,
                tool="write_document",
                wiki=request["wiki"]["title"],
                document_id=result["document_id"],
                attempt=attempt,
            )
            return result
        except WriteRejected:
            raise
        except Exception as exc:  # noqa: BLE001 — classified below
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            transient = _is_transient(exc)
            telemetry.log_event(
                "write_document.retry" if transient and attempt < attempts else "write_document.failed",
                level=logging.WARNING,
                write_id=write_id,
                tool="write_document",
                wiki=request["wiki"]["title"],
                attempt=attempt,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            if not transient or attempt == attempts:
                raise WriteFailed(attempt, errors) from exc
            await asyncio.sleep(_backoff())
    raise WriteFailed(attempts, errors)  # unreachable; keeps the type honest
