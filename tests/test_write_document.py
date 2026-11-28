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


def test_write_defers_embedding_and_enqueues_after_commit(store, path):
    """Sections land with NULL vectors inside the transaction; the enqueue is
    post-commit — one task carrying every section this write inserted."""
    from openmind import worker

    before = set(worker.pending_section_ids(store))  # seed rows, never embedded
    result = write.commit(write.validate(_request()))
    # All three payloads: document sections, progress sections, log entry.
    assert worker.huey.pending_count() == 1

    embedded = _count(
        store,
        "SELECT COUNT(*) FROM document_content_section WHERE embedding IS NOT NULL",
    )
    assert embedded == 0  # nothing talked to a model during the write

    new_sections = _count(
        store,
        "SELECT COUNT(*) FROM document_content_section WHERE id > 3",  # beyond the seed
    )
    assert new_sections == 4  # 2 document + 1 progress + 1 log
    pending = set(worker.pending_section_ids(store))
    assert len(pending - before) == 4  # the write's four sections joined the queue


def test_enqueue_failure_never_fails_the_write(store, path, monkeypatch):
    from openmind import worker

    monkeypatch.setattr(worker, "enqueue", lambda ids: False)
    result = write.commit(write.validate(_request()))
    assert result["created"] is True  # the document is durable regardless


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


def test_one_shot_creates_wiki_document_progress_and_log(store, path):
    result = write.commit(
        write.validate(_request(wiki={"id": None, "title": "New Wiki"}, references=[]))
    )
    assert result["wiki_created"] is True
    assert result["created"] is True

    wiki = store.execute(
        "SELECT id FROM wiki WHERE name = 'New Wiki'"
    ).fetchone()
    assert wiki["id"] == result["wiki_id"]

    progress = store.execute(
        "SELECT COUNT(*) FROM document WHERE wiki_id = ? AND type_id ="
        " (SELECT id FROM document_type WHERE name = 'progress')",
        (result["wiki_id"],),
    ).fetchone()[0]
    assert progress == 1  # server numbers it from COUNT(*)

    root_log = store.execute(
        "SELECT d.id FROM document d JOIN document_type t ON t.id = d.type_id"
        " WHERE t.name = 'log' AND d.wiki_id IS NULL"
    ).fetchone()
    assert result["log_document_id"] == root_log["id"]

    edge = store.execute(
        "SELECT COUNT(*) FROM document_cross_references"
        " WHERE source_document_id = ? AND target_document_id = ?",
        (result["progress_document_id"], result["document_id"]),
    ).fetchone()[0]
    assert edge == 1  # progress -> document, one write, one wiki


def test_one_shot_with_foreign_reference_is_rejected(store, path):
    before = _count(store, "SELECT COUNT(*) FROM wiki")
    with pytest.raises(write.WriteRejected):
        write.commit(
            write.validate(
                _request(wiki={"id": None, "title": "New Wiki"}, references=BUILD["references"])
            )
        )
    assert _count(store, "SELECT COUNT(*) FROM wiki") == before
    assert _count(store, "SELECT COUNT(*) FROM document WHERE title = 'Beta Notes'") == 0


def test_unknown_wiki_id_dies(store, path):
    before = _count(store, "SELECT COUNT(*) FROM document")
    with pytest.raises(write.WriteRejected) as caught:
        write.commit(write.validate(_request(wiki={"id": 999, "title": "test-wiki"})))
    assert caught.value.field == "wiki.id"
    assert _count(store, "SELECT COUNT(*) FROM document") == before


def test_null_id_with_existing_name_rejects_as_duplicate(store, path):
    before = _count(store, "SELECT COUNT(*) FROM wiki")
    with pytest.raises(write.WriteRejected) as caught:
        write.commit(write.validate(_request(wiki={"id": None, "title": "Test-Wiki"})))
    assert caught.value.field == "wiki.id"
    assert "already exists" in caught.value.reason
    assert _count(store, "SELECT COUNT(*) FROM wiki") == before
