"""Embedding: the wiring is ours, the model is a stub.

No test loads a model, downloads a wheel cache or touches the network. The
stub replaces `fastembed.TextEmbedding` itself, so the code under test is the
real constructor call, cache-directory creation and text pass-through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openmind import embedding


class StubTextEmbedding:
    """Stands in for fastembed's class; records how it was constructed."""

    instances: list["StubTextEmbedding"] = []

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedded: list[str] = []
        StubTextEmbedding.instances.append(self)

    def embed(self, texts):
        self.embedded = list(texts)
        return [tuple([0.1, 0.2, 0.3]) for _ in texts]  # tuples, to prove conversion


@pytest.fixture()
def stub_fastembed(monkeypatch):
    import fastembed

    StubTextEmbedding.instances = []
    monkeypatch.setattr(fastembed, "TextEmbedding", StubTextEmbedding)
    return StubTextEmbedding


def test_model_load_is_lazy_and_configured(stub_fastembed, tmp_path):
    embedder = embedding.FastEmbedEmbedder("BAAI/bge-base-en-v1.5", tmp_path / "cache")

    assert stub_fastembed.instances == []  # constructed, not loaded
    assert embedder.model_name == "BAAI/bge-base-en-v1.5"

    embedder.embed(["kestrel roosts on the cliff"])

    assert len(stub_fastembed.instances) == 1
    assert stub_fastembed.instances[0].model_name == "BAAI/bge-base-en-v1.5"
    assert stub_fastembed.instances[0].embedded == ["kestrel roosts on the cliff"]

    embedder.embed(["basalt columns"])
    assert len(stub_fastembed.instances) == 1  # loaded once, reused


def test_embed_returns_plain_lists_per_text(stub_fastembed, tmp_path):
    embedder = embedding.FastEmbedEmbedder("BAAI/bge-base-en-v1.5", tmp_path / "cache")

    vectors = embedder.embed(["one", "two"])

    assert len(vectors) == 2
    assert all(isinstance(vector, list) for vector in vectors)
    assert vectors[0] == [0.1, 0.2, 0.3]


def test_cache_directory_is_created_where_config_says(stub_fastembed, tmp_path):
    cache = tmp_path / "nested" / "fastembed_cache"
    assert not cache.exists()

    embedding.FastEmbedEmbedder("BAAI/bge-base-en-v1.5", cache).embed(["text"])

    assert cache.is_dir()
    assert stub_fastembed.instances[0].cache_dir == str(cache)


def test_configured_embedder_reads_config(monkeypatch, tmp_path):
    values = {
        ("embedding", "model"): "BAAI/bge-large-en-v1.5",
        ("embedding", "cache_path"): str(tmp_path / "cfg_cache"),
    }
    monkeypatch.setattr(embedding, "get", lambda section, key: values.get((section, key)))

    embedder = embedding.configured_embedder()

    assert isinstance(embedder, embedding.FastEmbedEmbedder)
    assert embedder.model_name == "BAAI/bge-large-en-v1.5"
    assert embedder.cache_dir == Path(tmp_path / "cfg_cache")


def test_configured_embedder_falls_back_to_the_documented_defaults(monkeypatch):
    monkeypatch.setattr(embedding, "get", lambda section, key: None)

    embedder = embedding.configured_embedder()

    assert embedder.model_name == embedding.DEFAULT_MODEL == "BAAI/bge-base-en-v1.5"
    assert embedder.cache_dir == Path(embedding.DEFAULT_CACHE_PATH)


def test_fastembed_satisfies_the_protocol():
    assert isinstance(embedding.FastEmbedEmbedder("m", ".c"), embedding.Embedder)
