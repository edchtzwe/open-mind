"""Read tools: one store, no bodies where nobody asked for bodies.

Every test drives `openmind.tools.read` against the disposable store — the same
rebinding trick the write tests use — so no assertion can touch the live store.

The fixture world adds what `seed.py` deliberately omits: a second wiki, a
soft-deleted document, a document inside a soft-deleted wiki, and a root log
with two entries at different timestamps.
"""

from __future__ import annotations

import sqlite3

import pytest

from openmind import db
from openmind.tools import read as read_tools

NOW_OLD = "2026-11-20T09:00:00Z"
NOW_NEW = "2026-11-21T09:00:00Z"


# --- store plumbing --------------------------------------------------------


@pytest.fixture()
def path(store, monkeypatch):
    """Rebind the read tools' connection to this test's store."""
    from conftest import resolve_test_db

    resolved = resolve_test_db()
    original = db.connect
    monkeypatch.setattr(read_tools.db, "connect", lambda path=None: original(resolved))
    return resolved


def _type_id(store: sqlite3.Connection, name: str) -> int:
    return store.execute("SELECT id FROM document_type WHERE name = ?", (name,)).fetchone()[0]


def _wiki(store: sqlite3.Connection, name: str) -> int:
    now = db.utcnow()
    return store.execute(
        "INSERT INTO wiki (name, created_at, updated_at) VALUES (?, ?, ?)",
        (name, now, now),
    ).lastrowid


def _document(
    store: sqlite3.Connection,
    wiki_id: int | None,
    type_name: str,
    title: str,
    *,
    deleted: bool = False,
) -> int:
    now = db.utcnow()
    return store.execute(
        "INSERT INTO document (wiki_id, type_id, title, created_at, updated_at, deleted_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (wiki_id, _type_id(store, type_name), title, now, now, now if deleted else None),
    ).lastrowid


def _section(
    store: sqlite3.Connection,
    document_id: int,
    position: int,
    body: str,
    *,
    heading: str = "",
    created_at: str | None = None,
) -> None:
    ts = created_at or db.utcnow()
    store.execute(
        "INSERT INTO document_content_section"
        " (document_id, position, heading, body, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (document_id, position, heading, body, ts, ts),
    )


def _edge(store: sqlite3.Connection, source: int, target: int, created_at: str) -> None:
    store.execute(
        "INSERT INTO document_cross_references"
        " (source_document_id, target_document_id, created_at) VALUES (?, ?, ?)",
        (source, target, created_at),
    )


def _doc_id(store: sqlite3.Connection, title: str) -> int:
    return store.execute("SELECT id FROM document WHERE title = ?", (title,)).fetchone()[0]


@pytest.fixture()
def world(store):
    """Second wiki, hidden rows, and a root log whose entries carry edges."""
    second = _wiki(store, "Second Wiki")
    beta = _document(store, second, "document", "Beta Notes")
    _section(store, beta, 1, "granite ledger")

    hidden = _document(store, second, "document", "Beta Hidden", deleted=True)
    _section(store, hidden, 1, "not to be seen")

    ghost = _wiki(store, "Ghost Wiki")
    ghost_doc = _document(store, ghost, "document", "Ghost Notes")
    _section(store, ghost_doc, 1, "should never surface")
    store.execute("UPDATE wiki SET deleted_at = ? WHERE id = ?", (db.utcnow(), ghost))

    alpha = _doc_id(store, "Alpha Notes")
    progress = _doc_id(store, "Alpha Progress")
    _edge(store, alpha, progress, db.utcnow())
    _edge(store, alpha, hidden, db.utcnow())  # must not surface

    log = _document(store, None, "log", "log")
    _section(store, log, 1, "Created wiki 'test-wiki'.", created_at=NOW_OLD)
    _section(store, log, 2, "Wrote 'Alpha Notes'.", created_at=NOW_NEW)
    _edge(store, log, progress, NOW_NEW)  # same transaction, same timestamp

    store.commit()
    return {
        "wiki": store.execute("SELECT id FROM wiki WHERE name = 'test-wiki'").fetchone()[0],
        "alpha": alpha,
        "progress": progress,
        "log": log,
        "hidden": hidden,
        "beta": beta,
    }


# --- list_wikis ------------------------------------------------------------


def test_list_wikis_counts_live_documents(world, path):
    wikis = {row["name"]: row for row in read_tools.list_wikis()}

    assert set(wikis) == {"test-wiki", "Second Wiki"}
    assert wikis["test-wiki"]["document_count"] == 3  # seeded notes, log, progress
    assert wikis["Second Wiki"]["document_count"] == 1  # Beta Hidden is deleted
    for row in wikis.values():
        assert {"id", "name", "created_at", "updated_at", "document_count"} <= set(row)

    assert [row["name"] for row in read_tools.list_wikis()] == ["Second Wiki", "test-wiki"]


# --- list_documents --------------------------------------------------------


def test_list_documents_hides_deleted_rows_and_scopes_by_wiki(world, path):
    everything = read_tools.list_documents()
    titles = [row["title"] for row in everything]

    assert titles == sorted(titles, key=str.lower)
    assert {"Alpha Log", "Alpha Notes", "Alpha Progress", "Beta Notes", "log"} <= set(titles)
    assert "Beta Hidden" not in titles  # soft-deleted document
    assert "Ghost Notes" not in titles  # document inside a soft-deleted wiki

    scoped = read_tools.list_documents("SECOND WIKI")  # case-insensitive name match
    assert [row["title"] for row in scoped] == ["Beta Notes"]
    assert scoped[0]["wiki"] == "Second Wiki"
    assert scoped[0]["type"] == "document"
    assert "sections" not in scoped[0]

    with pytest.raises(ValueError, match="no live wiki"):
        read_tools.list_documents("Ghost Wiki")


# --- read_document ---------------------------------------------------------


def test_read_document_returns_sections_and_bare_references(world, path):
    doc = read_tools.read_document(world["alpha"])

    assert doc["document_id"] == world["alpha"]
    assert doc["title"] == "Alpha Notes"
    assert doc["type"] == "document"
    assert doc["wiki"] == {"id": world["wiki"], "name": "test-wiki"}
    assert [section["position"] for section in doc["sections"]] == [1, 2]
    assert doc["sections"][0]["body"] == "kestrel roosts on the cliff"

    assert doc["references"] == [
        {"document_id": world["progress"], "title": "Alpha Progress", "sections": None}
    ]


def test_read_document_flag_inlines_reference_sections_one_level(world, path):
    doc = read_tools.read_document(world["alpha"], include_cross_reference_content=True)

    reference = doc["references"][0]
    assert reference["sections"] == [
        {
            "position": 1,
            "heading": "Done",
            "body": "scaffold and persistence layer landed",
        }
    ]
    assert "references" not in reference  # one level only

    root_log = read_tools.read_document(world["log"], include_cross_reference_content=True)
    assert root_log["type"] == "log"
    assert root_log["wiki"] is None
    assert [section["body"] for section in root_log["sections"]] == [
        "Created wiki 'test-wiki'.",
        "Wrote 'Alpha Notes'.",
    ]
    assert root_log["references"][0]["sections"][0]["body"] == (
        "scaffold and persistence layer landed"
    )


def test_read_document_refuses_deleted_ids(world, path):
    with pytest.raises(ValueError, match="no live document"):
        read_tools.read_document(world["hidden"])

    with pytest.raises(ValueError, match="must be an integer"):
        read_tools.read_document("7")  # type: ignore[arg-type]


# --- log_recall ------------------------------------------------------------


def test_log_recall_orders_entries_and_resolves_progress(world, path):
    entries = read_tools.log_recall()["entries"]

    assert [entry["entry"] for entry in entries] == [
        "Created wiki 'test-wiki'.",
        "Wrote 'Alpha Notes'.",
    ]
    assert entries[0]["progress"] is None  # wiki creation writes no progress record
    assert entries[1]["progress"] == {
        "document_id": world["progress"],
        "title": "Alpha Progress",
        "wiki": "test-wiki",
    }
    assert entries[1]["created_at"] == NOW_NEW


def test_log_recall_filters_by_since_and_limit(world, path):
    assert [entry["entry"] for entry in read_tools.log_recall(since="2026-11-21")["entries"]] == [
        "Wrote 'Alpha Notes'."
    ]
    assert [entry["entry"] for entry in read_tools.log_recall(limit=1)["entries"]] == [
        "Wrote 'Alpha Notes'."
    ]

    with pytest.raises(ValueError, match="ISO-8601"):
        read_tools.log_recall(since="yesterday")
    with pytest.raises(ValueError, match=">= 1"):
        read_tools.log_recall(limit=0)


def test_log_recall_on_a_store_without_a_root_log(store, path):
    assert read_tools.log_recall() == {"entries": []}
