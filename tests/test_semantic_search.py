"""semantic_search: the health gate first, then cosine over live vectors.

Embeddings are written straight into the disposable store (no worker, no
network); the query embed is stubbed at `read.configured_embedder`, which is
the seam `semantic_search` calls — the real constructor never runs.
"""

from __future__ import annotations

import pytest

from openmind import db, embedding
from openmind.tools import read as read_tools


@pytest.fixture()
def path(store, monkeypatch):
    """Rebind the read tools' connection to this test's store."""
    from conftest import resolve_test_db

    resolved = resolve_test_db()
    original = db.connect
    monkeypatch.setattr(read_tools.db, "connect", lambda path=None: original(resolved))
    return resolved


class StubEmbedder:
    """Returns the same vector for every query: ranking is driven by the
    stored vectors, so a test can place each row's distance by construction."""

    model_name = "stub-model"

    def __init__(self, vector):
        self.vector = vector

    def embed(self, texts):
        return [list(self.vector) for _ in texts]


def _store_vector(store, section_id, vector, model=None):
    store.execute(
        "UPDATE document_content_section SET embedding = ?, embedding_model = ?,"
        " embedding_text = 'text' WHERE id = ?",
        (
            embedding.vector_to_blob(vector),
            model or embedding.configured_model_name(),
            section_id,
        ),
    )
    store.commit()


def _section_id(store, title, position):
    return store.execute(
        "SELECT s.id FROM document_content_section s JOIN document d ON d.id = s.document_id"
        " WHERE d.title = ? AND s.position = ?",
        (title, position),
    ).fetchone()[0]


@pytest.fixture()
def vectors(store, path, monkeypatch):
    """Alpha Notes §1 and §2 embedded; the seed's third section left pending."""
    monkeypatch.setattr(read_tools, "configured_embedder", lambda: StubEmbedder([1.0, 0.0]))
    _store_vector(store, _section_id(store, "Alpha Notes", 1), [1.0, 0.0])
    _store_vector(store, _section_id(store, "Alpha Notes", 2), [0.0, 1.0])
    return store


def test_returns_every_embedded_section_best_first(vectors, path):
    hits = read_tools.semantic_search("kestrel")

    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[0]["section_id"] == _section_id(vectors, "Alpha Notes", 1)
    for hit in hits:
        assert {"section_id", "score", "document_id", "document_title", "wiki",
                "position", "heading", "body"} <= set(hit)
        assert hit["document_title"] == "Alpha Notes"


def test_pending_sections_are_absent_not_fatal(vectors, path):
    """Mid-embed is normal: the un-embedded seed section is simply missing."""
    hits = read_tools.semantic_search("anything")
    assert _section_id(vectors, "Alpha Progress", 1) not in {h["section_id"] for h in hits}


def test_wiki_scope_filters(vectors, path, store):
    store.execute("INSERT INTO wiki (name, created_at, created_by, updated_at, updated_by)"
                  " VALUES ('elsewhere', '2026-11-20T00:00:00Z', 't', '2026-11-20T00:00:00Z', 't')")
    with pytest.raises(ValueError, match="no live wiki"):
        read_tools.semantic_search("q", wiki="nowhere")


def test_model_drift_refuses_loud(vectors, path, monkeypatch):
    values = {("embedding", "openrouter_model"): "openai/text-embedding-3-small"}
    monkeypatch.setattr(embedding, "get", lambda section, key: values.get((section, key)))

    with pytest.raises(embedding.EmbeddingUnhealthy, match="drift"):
        read_tools.semantic_search("kestrel")


def test_dimension_mismatch_is_drift_not_zero(vectors, path, monkeypatch):
    monkeypatch.setattr(
        read_tools, "configured_embedder", lambda: StubEmbedder([1.0, 0.0, 0.0])
    )
    with pytest.raises(embedding.EmbeddingUnhealthy, match="dimension"):
        read_tools.semantic_search("kestrel")


def test_empty_query_rejected_at_the_boundary(vectors, path):
    with pytest.raises(ValueError, match="non-empty"):
        read_tools.semantic_search("   ")


def test_store_without_any_embedding_returns_empty(store, path, monkeypatch):
    monkeypatch.setattr(read_tools, "configured_embedder", lambda: StubEmbedder([1.0]))
    assert read_tools.semantic_search("anything") == []


def test_health_counts_drive_the_gate(vectors, path):
    health = embedding.embedding_health(vectors)
    assert health["pending"] == 1  # Alpha Progress never embedded
    assert health["stale"] == 0
    assert health["broken"] == 0
    assert health["model"] == embedding.configured_model_name()
