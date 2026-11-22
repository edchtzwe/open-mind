"""Wiki creation: one transaction, one wiki row, one root log entry.

Every test drives `write.create_wiki` against the disposable store — never
the live one — by rebinding `write.db.connect` to the fixture's resolved
path before any call. Duplicate and invalid names are rejections; a failed
commit must leave no wiki row behind.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from openmind import db, write

NEW_WIKIS = {
    "fresh": "Alpha Project",
    "control": "bad\nname",
    "empty": "   ",
    "dupe": "test-wiki",  # seeded name
    "dupe_case": "Test-Wiki",  # same wiki, different casing
}


@pytest.fixture()
def path(store, monkeypatch):
    """Rebind the write path's connection to this test's store."""
    from conftest import resolve_test_db

    resolved = resolve_test_db()
    original = db.connect
    monkeypatch.setattr(write.db, "connect", lambda path=None: original(resolved))
    return resolved


def _count(store, sql, params=()):
    return store.execute(sql, params).fetchone()[0]


def test_create_wiki_lands_wiki_and_root_log(store, path):
    result = asyncio.run(write.create_wiki(NEW_WIKIS["fresh"], write_id="test-wiki-create"))
    assert result["wiki"] == NEW_WIKIS["fresh"]
    assert result["wiki_id"] >= 1
    assert result["write_id"] == "test-wiki-create"
    assert result["attempts"] == 1

    row = store.execute(
        "SELECT id FROM wiki WHERE name = ?", (NEW_WIKIS["fresh"],)
    ).fetchone()
    assert row is not None
    assert row["id"] == result["wiki_id"]

    root_log = store.execute(
        "SELECT d.id FROM document d JOIN document_type t ON t.id = d.type_id"
        " WHERE t.name = 'log' AND d.wiki_id IS NULL"
    ).fetchone()
    assert root_log is not None
    assert result["log_document_id"] == root_log["id"]

    entry = store.execute(
        "SELECT body FROM document_content_section WHERE document_id = ?"
        " AND position = ?",
        (root_log["id"], result["log_position"]),
    ).fetchone()
    assert entry is not None
    assert "Alpha Project" in entry["body"]


def test_casing_is_preserved_exactly(store, path):
    result = asyncio.run(write.create_wiki("Open-Mind Notes"))
    assert result["wiki"] == "Open-Mind Notes"
    stored = store.execute(
        "SELECT name FROM wiki WHERE id = ?", (result["wiki_id"],)
    ).fetchone()
    assert stored["name"] == "Open-Mind Notes"


def test_duplicate_name_rejected_case_insensitively(store, path):
    with pytest.raises(write.WriteRejected) as caught:
        asyncio.run(write.create_wiki(NEW_WIKIS["dupe_case"]))
    assert caught.value.field == "name"
    assert "already exists" in caught.value.reason
    assert _count(store, "SELECT COUNT(*) FROM wiki") == 1  # only the seed


def test_duplicate_rejected_before_any_write(store, path):
    log_before = _count(
        store,
        "SELECT COUNT(*) FROM document_content_section WHERE document_id ="
        " (SELECT id FROM document WHERE wiki_id IS NULL AND title = 'log')",
    )
    with pytest.raises(write.WriteRejected):
        asyncio.run(write.create_wiki(NEW_WIKIS["dupe"]))
    log_after = _count(
        store,
        "SELECT COUNT(*) FROM document_content_section WHERE document_id ="
        " (SELECT id FROM document WHERE wiki_id IS NULL AND title = 'log')",
    )
    assert log_after == log_before
    assert _count(store, "SELECT COUNT(*) FROM wiki") == 1


def test_control_characters_rejected(store, path):
    with pytest.raises(write.WriteRejected) as caught:
        asyncio.run(write.create_wiki(NEW_WIKIS["control"]))
    assert caught.value.field == "name"


def test_empty_name_rejected(store, path):
    with pytest.raises(write.WriteRejected) as caught:
        asyncio.run(write.create_wiki(NEW_WIKIS["empty"]))
    assert caught.value.field == "name"


def test_failed_commit_leaves_no_wiki_row(store, path, monkeypatch):
    def boom(conn, log_id, entry, now):
        raise sqlite3.OperationalError("disk i/o error")

    monkeypatch.setattr(write, "_append_log", boom)
    with pytest.raises(write.WriteFailed):
        asyncio.run(write.create_wiki("Alpha Project"))
    assert _count(store, "SELECT COUNT(*) FROM wiki WHERE name = 'Alpha Project'") == 0
    assert _count(store, "SELECT COUNT(*) FROM wiki") == 1  # seed intact
