"""The embed worker: backfill, skip rules and the reconciler's query.

The Huey queue is real but never consumed — `enqueue` is asserted at the
storage level (one pending task, correct payload), and `embed_sections` is
run through `call_local`, which executes the task body without the consumer.
The embedder is stubbed at `configured_embedder`: no network, no model.
"""

from __future__ import annotations

import pytest

from openmind import db, embedding, worker


@pytest.fixture()
def path(store, monkeypatch):
    """Rebind the worker's connection (and the write path's) to this store."""
    from conftest import resolve_test_db

    resolved = resolve_test_db()
    original = db.connect
    monkeypatch.setattr(worker.db, "connect", lambda path=None: original(resolved))
    return resolved


class StubEmbedder:
    model_name = "stub-model"

    def embed(self, texts):
        return [[0.5] * 3 for _ in texts]


@pytest.fixture()
def stub_embedder(monkeypatch):
    monkeypatch.setattr(worker, "configured_embedder", lambda: StubEmbedder())
    return StubEmbedder


def _pending(store):
    return worker.pending_section_ids(store)


def test_section_text_joins_heading_and_body_only_when_heading_exists():
    assert worker.section_text("Heading", "body") == "Heading\nbody"
    assert worker.section_text("", "body") == "body"


def test_backfill_stamps_vector_model_text_and_worker(store, path, stub_embedder):
    ids = _pending(store)
    assert ids  # seed sections have no vectors

    worker.embed_sections.call_local(ids)

    row = store.execute(
        "SELECT embedding, embedding_model, embedding_text, updated_at, updated_by"
        " FROM document_content_section WHERE id = ?",
        (ids[0],),
    ).fetchone()
    assert row["embedding"] == embedding.vector_to_blob([0.5] * 3)
    assert row["embedding_model"] == "stub-model"
    assert row["embedding_text"] == worker.section_text("Overview", "kestrel roosts on the cliff")
    assert row["updated_by"] == db.WORKER  # audit trail: comb logs by time, get rows back
    store.commit()
    assert _pending(store) == []


def test_backfill_skips_already_embedded_and_deleted(store, path, stub_embedder):
    ids = _pending(store)
    worker.embed_sections.call_local(ids)  # embed everything once
    # Reset all three columns together: the CHECK forbids a lone NULL.
    store.execute(
        "UPDATE document_content_section SET embedding = NULL, embedding_model = NULL,"
        " embedding_text = NULL WHERE id IN (?, ?)",
        (ids[0], ids[1]),
    )
    store.execute(
        "UPDATE document_content_section SET deleted_at = '2026-11-20T00:00:00Z',"
        " deleted_by = 't' WHERE id = ?",
        (ids[1],),
    )
    store.commit()

    worker.embed_sections.call_local(ids)

    embedded = store.execute(
        "SELECT COUNT(*) FROM document_content_section WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    assert embedded == len(ids) - 1  # live one re-embedded, deleted one never
    still = _pending(store)
    assert ids[0] not in still
    assert ids[1] not in still  # deleted rows are not pending (filter is live-only)


def test_backfill_never_touches_a_stale_vector(store, path, stub_embedder):
    """Model drift is the operator's problem; the worker must not loop on it."""
    stale_id = _pending(store)[0]
    store.execute(
        "UPDATE document_content_section SET embedding = ?, embedding_model = 'old-model',"
        " embedding_text = 'text' WHERE id = ?",
        (embedding.vector_to_blob([9.0]), stale_id),
    )
    store.commit()

    worker.embed_sections.call_local([stale_id])

    row = store.execute(
        "SELECT embedding_model FROM document_content_section WHERE id = ?", (stale_id,)
    ).fetchone()
    assert row["embedding_model"] == "old-model"  # untouched
    health = embedding.embedding_health(store)
    assert health["stale"] == 1

def test_enqueue_returns_true_and_leaves_one_pending_task(store, path):
    ids = _pending(store)
    assert worker.enqueue(ids) is True
    assert worker.huey.pending_count() == 1
    assert worker.enqueue([]) is False
    assert worker.huey.pending_count() == 1


def test_enqueue_failure_is_logged_and_downgraded(store, path, monkeypatch):
    def boom(section_ids):
        raise OSError("queue file is gone")

    monkeypatch.setattr(worker, "embed_sections", boom)
    assert worker.enqueue([1, 2]) is False  # never raises into the write path


def test_reconciler_query_returns_pending_live_ids_oldest_first(store, path):
    ids = worker.pending_section_ids(store)
    assert ids == sorted(ids)
    assert set(ids) <= {
        row["id"]
        for row in store.execute(
            "SELECT id FROM document_content_section WHERE embedding IS NULL"
            " AND deleted_at IS NULL"
        )
    }

    store.execute(
        "DELETE FROM document_content_section WHERE id = ?", (ids[0],)
    )
    store.commit()
    assert ids[0] not in worker.pending_section_ids(store)
    assert worker.pending_section_ids(store, limit=1) == worker.pending_section_ids(store)[:1]
