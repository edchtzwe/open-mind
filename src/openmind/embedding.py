"""Embedding behind a one-method interface.

Vectors leave this module as plain lists; nothing above it knows whether they
came from fastembed, a stub, or a future provider. That is the seam tests use,
and the reason no test downloads a model.

The model name travels with every vector into `document_content_section
.embedding_model`: vectors from two models are not comparable, so the column is
the only thing that can tell a live row from a stale one. Choosing a model is
the operator's job — `config.toml` names it, this module does not second-guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from openmind.config import get

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CACHE_PATH = ".fastembed_cache"


@runtime_checkable
class Embedder(Protocol):
    """What the write path and search need, and nothing more."""

    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order."""
        ...


class FastEmbedEmbedder:
    """fastembed's TextEmbedding, configured by `[embedding]` in config.toml.

    Two reasons this is not a bare call to fastembed: the model load is lazy
    (importing onnxruntime costs seconds and a caller that never embeds should
    not pay it), and `cache_path` is explicit. fastembed's own default cache is
    `$TMPDIR/fastembed_cache`, which a reboot erases — a restart would then
    silently re-download the model.
    """

    def __init__(self, model: str, cache_path: str | Path) -> None:
        self.model_name = model
        self.cache_dir = Path(cache_path)
        self._encoder: object | None = None

    def _load(self) -> object:
        if self._encoder is None:
            from fastembed import TextEmbedding

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._encoder = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
            )
        return self._encoder

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().embed(texts)
        return [list(vector) for vector in vectors]


def configured_embedder() -> Embedder:
    """The embedder `config.toml` describes.

    Read at call time, so an edited `[embedding]` section applies on the next
    tool call without a restart — the same contract as every other setting.
    """
    model = get("embedding", "model") or DEFAULT_MODEL
    cache_path = get("embedding", "cache_path") or DEFAULT_CACHE_PATH
    return FastEmbedEmbedder(str(model), str(cache_path))
