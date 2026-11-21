"""Persistence layer: bootstrap idempotence, seed, FTS sync, integrity, clock."""

from __future__ import annotations

import re
import sqlite3

import pytest

from openmind import db


def _wiki(conn: sqlite3.Connection) -> int:
    now = db.utcnow()
    cur = conn.execute(
        "INSERT INTO wiki (name, created_at, updated_at) VALUES (?, ?, ?)",
        ("notes", now, now),
    )
    return cur.lastrowid


def _document(conn: sqlite3.Connection) -> int:
    now = db.utcnow()
    wiki_id = _wiki(conn)
    type_id = conn.execute(
        "SELECT id FROM document_type WHERE name = 'document'"
    ).fetchone()[0]
    cur = conn.execute(
        "INSERT INTO document (wiki_id, type_id, title, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (wiki_id, type_id, "Doc", now, now),
    )
    return cur.lastrowid


def _section(conn: sqlite3.Connection, body: str, position: int = 1) -> int:
    now = db.utcnow()
    document_id = _document(conn)
    cur = conn.execute(
        "INSERT INTO document_content_section"
        " (document_id, position, heading, body, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (document_id, position, "Heading", body, now, now),
    )
    return cur.lastrowid


def _fts_hits(conn: sqlite3.Connection, term: str) -> list[int]:
    rows = conn.execute(
        "SELECT rowid FROM section_fts WHERE section_fts MATCH ? ORDER BY rowid",
        (term,),
    ).fetchall()
    return [row[0] for row in rows]


@pytest.fixture()
def conn(tmp_path):
    connection = db.connect(tmp_path / "store.db")
    assert db.bootstrap(connection) is True
    yield connection
    connection.close()


def test_bootstrap_is_idempotent(tmp_path):
    path = tmp_path / "fresh.db"
    connection = db.connect(path)

    assert db.bootstrap(connection) is True
    assert connection.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION

    assert db.bootstrap(connection) is False
    assert connection.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    connection.close()


def test_document_types_are_seeded_once(conn):
    names = [row[0] for row in conn.execute("SELECT name FROM document_type ORDER BY name")]
    assert names == ["document", "log", "progress"]

    conn.execute("INSERT OR IGNORE INTO document_type (name) VALUES ('document')")
    assert conn.execute("SELECT count(*) FROM document_type").fetchone()[0] == 3


def test_fts_follows_insert_update_and_delete(conn):
    section_id = _section(conn, "alpha beta")

    assert _fts_hits(conn, "alpha") == [section_id]

    conn.execute(
        "UPDATE document_content_section SET body = 'gamma delta' WHERE id = ?",
        (section_id,),
    )
    assert _fts_hits(conn, "alpha") == []
    assert _fts_hits(conn, "gamma") == [section_id]

    conn.execute("DELETE FROM document_content_section WHERE id = ?", (section_id,))
    assert _fts_hits(conn, "gamma") == []
    assert conn.execute("SELECT count(*) FROM section_fts").fetchone()[0] == 0


def test_soft_delete_keeps_fts_row(conn):
    section_id = _section(conn, "epsilon")

    conn.execute(
        "UPDATE document_content_section SET deleted_at = ? WHERE id = ?",
        (db.utcnow(), section_id),
    )

    # Search joins the live rows itself; the index is left intact so a restore
    # needs no reindex.
    assert _fts_hits(conn, "epsilon") == [section_id]
    live = conn.execute(
        "SELECT count(*) FROM document_content_section WHERE deleted_at IS NULL"
    ).fetchone()[0]
    assert live == 0


def test_foreign_keys_reject_orphans(conn):
    now = db.utcnow()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO document_content_section"
            " (document_id, position, heading, body, created_at, updated_at)"
            " VALUES (?, 1, 'h', 'b', ?, ?)",
            (999_999, now, now),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO document_cross_references"
            " (source_document_id, target_document_id, created_at) VALUES (1, 1, ?)",
            (now,),
        )


def test_embedding_columns_are_all_or_nothing(conn):
    now = db.utcnow()
    document_id = _document(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO document_content_section"
            " (document_id, position, heading, body, embedding, created_at, updated_at)"
            " VALUES (?, 1, 'h', 'b', X'00', ?, ?)",
            (document_id, now, now),
        )


def test_connection_settings(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_utcnow_format_is_sortable_and_exact():
    stamps = [db.utcnow() for _ in range(3)]
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s) for s in stamps)
    assert stamps == sorted(stamps)
