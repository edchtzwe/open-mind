"""doctor: counts, the suggestion, and the refusals — never auto-reconcile.

The CLI's exit code is the contract a wake script branches on: 0 healthy,
1 anything to act on. `--reconcile` enqueues; it never consumes.
"""

from __future__ import annotations

import pytest

from openmind import db, doctor, embedding, worker


@pytest.fixture()
def path(store, monkeypatch):
    resolved = __import__("conftest").resolve_test_db()
    original = db.connect
    monkeypatch.setattr(db, "connect", lambda path=None: original(resolved))
    monkeypatch.setattr(worker.db, "connect", lambda path=None: original(resolved))
    return resolved


def _set_stale(store, section_id):
    store.execute(
        "UPDATE document_content_section SET embedding = ?, embedding_model = 'old-model',"
        " embedding_text = 'text' WHERE id = ?",
        (embedding.vector_to_blob([1.0]), section_id),
    )
    store.commit()


def test_healthy_exit_is_zero_when_everything_is_embedded(store, path, capsys):
    # Embed every live section with the configured model.
    for row in store.execute(
        "SELECT id FROM document_content_section WHERE embedding IS NULL"
    ).fetchall():
        store.execute(
            "UPDATE document_content_section SET embedding = ?, embedding_model = ?,"
            " embedding_text = 'text' WHERE id = ?",
            (embedding.vector_to_blob([0.1]), embedding.configured_model_name(), row["id"]),
        )
    store.commit()

    assert doctor.main([]) == 0
    out = capsys.readouterr().out
    assert "healthy" in out
    assert worker.huey.pending_count() == 0  # diagnosis only


def test_pending_suggests_reconcile_and_exits_one(store, path, capsys):
    assert doctor.main([]) == 1
    out = capsys.readouterr().out
    assert "pending" in out
    assert "--reconcile" in out
    assert worker.huey.pending_count() == 0  # suggestion, not action


def test_reconcile_enqueues_pending_then_recounts(store, path, capsys):
    before = len(worker.pending_section_ids(store))

    assert doctor.main(["--reconcile"]) == 1  # pending goes to 0, but was unhealthy
    out = capsys.readouterr().out
    assert f"queued {before} section(s)" in out
    assert worker.huey.pending_count() == 1  # one batch task for all ids


def test_stale_is_loud_and_names_the_choices(store, path, capsys):
    _set_stale(store, worker.pending_section_ids(store)[0])
    # Embed the rest so 'stale' is the only complaint.
    for row in store.execute(
        "SELECT id FROM document_content_section WHERE embedding IS NULL"
    ).fetchall():
        store.execute(
            "UPDATE document_content_section SET embedding = ?, embedding_model = ?,"
            " embedding_text = 'text' WHERE id = ?",
            (embedding.vector_to_blob([0.1]), embedding.configured_model_name(), row["id"]),
        )
    store.commit()

    assert doctor.main([]) == 1
    out = capsys.readouterr().out
    assert "reembed" in out and "openrouter_model" in out
    assert worker.huey.pending_count() == 0  # reconcile never fixes drift


def test_broken_records_are_named_as_an_architecture_fault(store, path, capsys, monkeypatch):
    """The CHECK makes a broken row impossible to write through the API, so
    the counter itself is asserted clean above; the report path is driven
    directly to prove the loud refusal fires."""
    monkeypatch.setattr(
        doctor, "report", lambda conn: {"pending": 0, "stale": 0, "broken": 2, "model": "m"}
    )
    assert doctor.main([]) == 1
    out = capsys.readouterr().out
    assert "corrupted embedding record" in out
    assert "semantic search is unsafe" in out
    assert worker.huey.pending_count() == 0  # broken is not reconcilable
