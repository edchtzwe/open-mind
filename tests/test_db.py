"""Persistence layer: bootstrap, seed, FTS sync, integrity, clock."""

from __future__ import annotations

import re
import sqlite3

import pytest

from openmind import db


def _document_id(conn: sqlite3.Connection, title: str) -> int:
    return conn.execute("SELECT id FROM document WHERE title = ?", (title,)).fetchone()[0]


def _fts_hits(conn: sqlite3.Connection, term: str) -> list[int]:
    rows = conn.execute(
        "SELECT rowid FROM section_fts WHERE section_fts MATCH ? ORDER BY rowid",
        (term,),
    ).fetchall()
    return [row[0] for row in rows]


def _insert_section(conn: sqlite3.Connection, body: str, position: int = 3) -> int:
    now = db.utcnow()
    return conn.execute(
        "INSERT INTO document_content_section"
        " (document_id, position, heading, body, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (_document_id(conn, "Alpha Notes"), position, "Inserted", body, now, now),
    ).lastrowid


def test_bootstrap_is_idempotent(tmp_path):
    path = tmp_path / "fresh.db"
    connection = db.connect(path)

    assert db.bootstrap(connection) is True
    assert connection.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION

    assert db.bootstrap(connection) is False
    assert connection.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    connection.close()


def test_seed_shape(store):
    assert store.execute("SELECT count(*) FROM wiki").fetchone()[0] == 1
    assert store.execute("SELECT count(*) FROM document").fetchone()[0] == 3
    assert store.execute("SELECT count(*) FROM document_content_section").fetchone()[0] == 3
    assert store.execute("SELECT count(*) FROM document_cross_references").fetchone()[0] == 1

    types = {
        row[0]
        for row in store.execute(
            "SELECT t.name FROM document d JOIN document_type t ON t.id = d.type_id"
        )
    }
    assert types == {"document", "log", "progress"}


def test_document_types_are_seeded_once(store):
    names = [row[0] for row in store.execute("SELECT name FROM document_type ORDER BY name")]
    assert names == ["document", "log", "progress"]

    store.execute("INSERT OR IGNORE INTO document_type (name) VALUES ('document')")
    assert store.execute("SELECT count(*) FROM document_type").fetchone()[0] == 3


def test_seeded_sections_are_searchable(store):
    notes_id = _document_id(store, "Alpha Notes")
    assert _fts_hits(store, "kestrel") != []
    assert _fts_hits(store, "basalt") != []
    assert len(_fts_hits(store, "kestrel")) == 1
    assert _fts_hits(store, "kestrel") != _fts_hits(store, "basalt")
    assert notes_id in _fts_hits(store, "kestrel")


def test_fts_follows_insert_update_and_delete(store):
    section_id = _insert_section(store, "alpha beta")

    assert _fts_hits(store, "alpha") == [section_id]

    store.execute(
        "UPDATE document_content_section SET body = 'gamma delta' WHERE id = ?",
        (section_id,),
    )
    assert _fts_hits(store, "alpha") == []
    assert _fts_hits(store, "gamma") == [section_id]

    store.execute("DELETE FROM document_content_section WHERE id = ?", (section_id,))
    assert _fts_hits(store, "gamma") == []
    assert store.execute("SELECT count(*) FROM section_fts").fetchone()[0] == 3


def test_soft_delete_keeps_fts_row(store):
    section_id = _insert_section(store, "epsilon")

    store.execute(
        "UPDATE document_content_section SET deleted_at = ? WHERE id = ?",
        (db.utcnow(), section_id),
    )

    # Search joins the live rows itself; the index stays intact so a restore
    # needs no reindex.
    assert _fts_hits(store, "epsilon") == [section_id]
    live = store.execute(
        "SELECT count(*) FROM document_content_section WHERE deleted_at IS NULL"
    ).fetchone()[0]
    assert live == 3


def test_foreign_keys_reject_orphans(store):
    now = db.utcnow()
    with pytest.raises(sqlite3.IntegrityError):
        store.execute(
            "INSERT INTO document_content_section"
            " (document_id, position, heading, body, created_at, updated_at)"
            " VALUES (?, 1, 'h', 'b', ?, ?)",
            (999_999, now, now),
        )

    document_id = _document_id(store, "Alpha Notes")
    with pytest.raises(sqlite3.IntegrityError):
        store.execute(
            "INSERT INTO document_cross_references"
            " (source_document_id, target_document_id, created_at) VALUES (?, ?, ?)",
            (document_id, document_id, now),
        )


def test_embedding_columns_are_all_or_nothing(store):
    now = db.utcnow()
    with pytest.raises(sqlite3.IntegrityError):
        store.execute(
            "INSERT INTO document_content_section"
            " (document_id, position, heading, body, embedding, created_at, updated_at)"
            " VALUES (?, 1, 'h', 'b', X'00', ?, ?)",
            (_document_id(store, "Alpha Notes"), now, now),
        )


def test_connection_settings(store):
    assert store.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_utcnow_format_is_sortable_and_exact():
    stamps = [db.utcnow() for _ in range(3)]
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s) for s in stamps)
    assert stamps == sorted(stamps)
