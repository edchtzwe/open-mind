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

from array import array
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openmind import db
from openmind.config import ENV_FILE, GLOBAL_ENV_FILE, env, get

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CACHE_PATH = ".fastembed_cache"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = "OPENROUTER_API_KEY"


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


class OpenRouterEmbedder:
    """Embeddings through OpenRouter's OpenAI-compatible endpoint.

    The openai SDK with `base_url` pointed at OpenRouter: one dependency,
    no client of our own. The model string comes from
    `[embedding] openrouter_model` and is sent as-is — if the operator
    names a model OpenRouter rejects, that error surfaces, it is not ours
    to paper over.

    The key comes from the `.env` chain (colocated, then global), never
    from config.toml. Construction is lazy in the same way fastembed's is:
    the HTTP client is built on first use, so a caller that never embeds
    never opens a socket.
    """

    def __init__(self, model: str, api_key: str) -> None:
        self.model_name = model
        self.api_key = api_key
        self._client: object | None = None

    def _load(self) -> object:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=self.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._load().embeddings.create(model=self.model_name, input=list(texts))  # type: ignore[attr-defined]
        # OpenRouter returns data ordered by index; sort defensively so a
        # reordered response cannot silently pair texts with the wrong vector.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in ordered]


def openrouter_key() -> str | None:
    """The OpenRouter token from the `.env` chain, or None if nowhere has one."""
    return env(OPENROUTER_API_KEY)


def configured_embedder() -> Embedder:
    """The embedder `config.toml` describes.

    Read at call time, so an edited `[embedding]` section applies on the next
    tool call without a restart — the same contract as every other setting.

    A non-empty `openrouter_model` means the operator chose OpenRouter: the
    model string is theirs, and a missing token is a loud failure, never a
    silent fall back to fastembed (which would embed with a different model
    and poison the vectors).
    """
    openrouter_model = (get("embedding", "openrouter_model") or "").strip()
    if openrouter_model:
        key = openrouter_key()
        if not key:
            raise RuntimeError(
                "embedding.openrouter_model is set but no OPENROUTER_API_KEY was found; "
                f"add it to {ENV_FILE} or {GLOBAL_ENV_FILE}, or clear openrouter_model"
            )
        return OpenRouterEmbedder(openrouter_model, key)

    model = get("embedding", "model") or DEFAULT_MODEL
    cache_path = get("embedding", "cache_path") or DEFAULT_CACHE_PATH
    return FastEmbedEmbedder(str(model), str(cache_path))


def configured_model_name() -> str:
    """The model name that any *new* vector will carry — read from config
    without constructing an embedder, so health checks never touch the network
    or the token chain."""
    openrouter_model = (get("embedding", "openrouter_model") or "").strip()
    if openrouter_model:
        return openrouter_model
    return str(get("embedding", "model") or DEFAULT_MODEL)


# --- vector storage --------------------------------------------------------

# float32 little-endian, the array module's native layout on every platform
# this runs on. Dimension is derived from the byte length at read time — the
# blob carries no header, so the model column stays the only metadata.
def vector_to_blob(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def blob_to_vector(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


# --- health -----------------------------------------------------------------


def embedding_health(conn=None) -> dict[str, int]:
    """Counts that decide whether semantic search may run: pending, stale,
    broken. Shared verbatim by `semantic_search` (the gate) and `doctor`
    (the wake report) — one query shape, two consumers, no chance of the
    two disagreeing about what "healthy" means.

    - pending: live sections with no vector yet (normal mid-embed state;
      the reconciler's job, not a fault)
    - stale: live vectors carrying a model other than the configured one
      (drift after a model change — the operator's to fix)
    - broken: live rows violating the all-or-nothing embedding CHECK
      (should be impossible; if it fires, the architecture is sick)

    All three scan live rows only: a deleted section can neither match nor
    poison a search, so it is not counted.
    """
    own = conn is None
    if own:
        conn = db.connect()
    try:
        current = configured_model_name()
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) AS pending,
                SUM(CASE
                    WHEN embedding IS NOT NULL
                     AND (embedding_model IS NULL
                          OR embedding_model <> :model
                          OR embedding_text IS NULL)
                    THEN 1 ELSE 0 END) AS stale,
                SUM(CASE
                    WHEN (embedding IS NULL) <> (embedding_model IS NULL)
                      OR (embedding IS NULL) <> (embedding_text IS NULL)
                    THEN 1 ELSE 0 END) AS broken
            FROM document_content_section
            WHERE deleted_at IS NULL
            """,
            {"model": current},
        ).fetchone()
    finally:
        if own:
            conn.close()
    return {
        "pending": int(row["pending"] or 0),
        "stale": int(row["stale"] or 0),
        "broken": int(row["broken"] or 0),
        "model": current,
    }


class EmbeddingUnhealthy(RuntimeError):
    """The search gate: drift or corruption refuses loudly with the choices."""
