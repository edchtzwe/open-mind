"""The write path: rejection, rollback, numbering and the log append.

Every test drives `write.commit` against the disposable store — never the live
one — by rebinding `write.db.connect` to the fixture's resolved path before any
call. The retry wrapper is exercised once; the rest test the transaction itself.
"""

from __future__ import annotations

import asyncio

import pytest

from openmind import db, write
from seed import WIKI

BUILD = {
    "wiki": WIKI,
    "title": "Beta Notes",
    "mode": "replace_all",
    "sections": [
        {"position": 1, "heading": "Opening", "body": "granite and kelp"},
        {"position": 2, "heading": "", "body": "a lead section with no heading"},
    ],
    "log_entry": "Wrote Beta Notes.",
    "progress": {
        "slug": "beta-notes",
        "sections": [{"position": 1, "heading": "What changed", "body": "New page."}],
    },
    "references": [{"wiki": WIKI, "title": "Alpha Notes"}],
}


def _request(**overrides):
    request = {**BUILD, **overrides}
    request["progress"] = {**BUILD["progress"], **overrides.get("progress", {})}
    return request


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


def test_missing_wiki_is_rejected_by_field():
    with pytest.raises(write.WriteRejected) as caught:
        write.validate({k: v for k, v in BUILD.items() if k != "wiki"})
    assert caught.value.field == "wiki"


def test_duplicate_positions_rejected():
    sections = [
        {"position": 1, "heading": "a", "body": "one"},
        {"position": 1, "heading": "b", "body": "two"},
    ]
    with pytest.raises(write.WriteRejected) as caught:
        write.validate(_request(sections=sections))
    assert "duplicate position" in caught.value.reason


def test_bad_progress_slug_rejected():
    with pytest.raises(write.WriteRejected) as caught:
        write.validate(_request(progress={"slug": "Beta Notes", "sections": BUILD["progress"]["sections"]}))
    assert caught.value.field == "progress.slug"


def test_unknown_wiki_rejected_before_any_write(store, path):
    before = _count(store, "SELECT COUNT(*) FROM document")
    with pytest.raises(write.WriteRejected) as caught:
        write.commit(write.validate(_request(wiki="no-such-wiki")))
    assert caught.value.field == "wiki"
    assert _count(store, "SELECT COUNT(*) FROM document") == before


def test_progress_document_is_not_an_edit_target(store, path):
    progress_id = store.execute(
        "SELECT id FROM document WHERE title = 'Alpha Progress'"
    ).fetchone()[0]
    with pytest.raises(write.WriteRejected) as caught:
        write.commit(write.validate(_request(document_id=progress_id, title=None)))
    assert caught.value.field == "document_id"


def test_write_lands_document_progress_log_and_edges(store, path):
    result = write.commit(write.validate(_request()))
    assert result["created"] is True
    assert result["progress_title"] == "02_beta-notes"  # seed already holds 01

    root_log = store.execute(
        "SELECT d.id, d.wiki_id FROM document d JOIN document_type t ON t.id = d.type_id"
        " WHERE t.name = 'log' AND d.wiki_id IS NULL"
    ).fetchone()
    assert root_log is not None
    assert result["log_document_id"] == root_log[0]
    assert result["log_position"] == 1

    edges = _count(
        store,
        "SELECT COUNT(*) FROM document_cross_references WHERE source_document_id IN (?, ?)",
        (result["document_id"], result["progress_document_id"]),
    )
    assert edges == 2  # document -> Alpha Notes, progress -> document


def test_log_appends_one_section_per_write(store, path):
    first = write.commit(write.validate(_request()))
    second = write.commit(write.validate(_request(title="Gamma Notes", log_entry="Wrote Gamma Notes.")))
    assert (first["log_position"], second["log_position"]) == (1, 2)
    assert _count(
        store,
        "SELECT COUNT(*) FROM document_content_section WHERE document_id = ?",
        (first["log_document_id"],),
    ) == 2


def test_bad_reference_rolls_the_whole_write_back(store, path):
    before = _count(store, "SELECT COUNT(*) FROM document")
    with pytest.raises(write.WriteRejected):
        write.commit(
            write.validate(_request(references=[{"wiki": WIKI, "title": "Ghost Page"}]))
        )
    assert _count(store, "SELECT COUNT(*) FROM document") == before
    assert _count(store, "SELECT COUNT(*) FROM document WHERE title = 'Beta Notes'") == 0


def test_replace_positions_keeps_the_other_sections(store, path):
    write.commit(write.validate(_request()))
    write.commit(
        write.validate(
            _request(
                title=None,
                document_id=store.execute(
                    "SELECT id FROM document WHERE title = 'Beta Notes'"
                ).fetchone()[0],
                mode="replace_positions",
                sections=[{"position": 2, "heading": "Fixed", "body": "steel and fog"}],
            )
        )
    )
    rows = store.execute(
        "SELECT position, heading FROM document_content_section"
        " WHERE document_id = (SELECT id FROM document WHERE title = 'Beta Notes')"
        " ORDER BY position"
    ).fetchall()
    assert [tuple(row) for row in rows] == [(1, "Opening"), (2, "Fixed")]


def test_async_wrapper_returns_attempt_count(store, path):
    result = asyncio.run(write.write_document(_request(), write_id="test-write"))
    assert result["write_id"] == "test-write"
    assert result["attempts"] == 1
