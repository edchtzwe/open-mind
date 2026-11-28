"""Read and search tools: the surface an agent reads the store through.

Reads open one connection per call and never write. Every listing filters
soft-deleted rows, because a hidden document must not surface through a
reference, a count or a log entry — deletion that still leaks is not deletion.

Token discipline is part of the contract here, not an optimisation. A
reference carries `{document_id, title}` and `sections: null` unless the caller
asks for content; with the flag, bodies are inlined one level deep only, never
recursively. Following a reference is its own call, made when the body is
actually needed.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict, deque
from typing import Any

from openmind import db
from openmind.embedding import (
    EmbeddingUnhealthy,
    blob_to_vector,
    configured_embedder,
    embedding_health,
)
from openmind.tools import mcp

_SINCE = re.compile(r"^\d{4}-\d{2}-\d{2}")


# --- plumbing --------------------------------------------------------------


def _sections(conn: sqlite3.Connection, document_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT position, heading, body
        FROM document_content_section
        WHERE document_id = :id AND deleted_at IS NULL
        ORDER BY position
        """,
        {"id": document_id},
    ).fetchall()
    return [
        {"position": row["position"], "heading": row["heading"], "body": row["body"]}
        for row in rows
    ]


def _live_document(conn: sqlite3.Connection, document_id: Any) -> sqlite3.Row:
    if not isinstance(document_id, int) or isinstance(document_id, bool):
        raise ValueError("document_id: must be an integer")
    # LEFT JOIN, not JOIN: a root document has no wiki row to join. The root
    # document is the log — the single legitimate case of wiki_id IS NULL.
    row = conn.execute(
        """
        SELECT d.id, d.title, dt.name AS type_name, d.created_at, d.updated_at,
               d.wiki_id, w.name AS wiki_name
        FROM document d
        JOIN document_type dt ON dt.id = d.type_id
        LEFT JOIN wiki w ON w.id = d.wiki_id
        WHERE d.id = :id AND d.deleted_at IS NULL
        """,
        {"id": document_id},
    ).fetchone()
    if row is None:
        raise ValueError(f"document_id: no live document with id {document_id}")
    return row


def _wiki_id(conn: sqlite3.Connection, name: Any) -> int:
    """Resolve a wiki name case-insensitively, as every lookup in the store does."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("wiki: must be a non-empty string")
    row = conn.execute(
        """
        SELECT id FROM wiki
        WHERE name = :name COLLATE NOCASE AND deleted_at IS NULL
        LIMIT 1
        """,
        {"name": name.strip()},
    ).fetchone()
    if row is None:
        raise ValueError(f"wiki: no live wiki named {name!r}; list_wikis() has the names")
    return row["id"]


# --- tools -----------------------------------------------------------------


@mcp.tool()
def list_wikis() -> list[dict[str, Any]]:
    """List every live wiki with its document count.

    Call this before writing into a wiki that may already exist: the write
    tools take a wiki name, and a name is only useful if the caller knows it.
    `document_count` excludes deleted documents; root-level documents belong to
    no wiki and are not counted.
    """
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT w.id, w.name, w.created_at, w.updated_at,
                   (SELECT COUNT(*) FROM document d
                    WHERE d.wiki_id = w.id AND d.deleted_at IS NULL) AS document_count
            FROM wiki w
            WHERE w.deleted_at IS NULL
            ORDER BY w.name COLLATE NOCASE
            """
        ).fetchall()
    return [dict(row) for row in rows]


@mcp.tool()
def list_documents(wiki: str | None = None) -> list[dict[str, Any]]:
    """List live documents so the caller can pick an edit target.

    With `wiki`, only that wiki's documents are returned and the name match
    ignores case; without it, every live wiki plus root-level documents. Each
    row carries `id`, `title`, `type` and `updated_at` — no bodies. Read one
    with `read_document` once you know which you need.
    """
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.title, dt.name AS type_name, d.updated_at,
                   w.name AS wiki_name
            FROM document d
            JOIN document_type dt ON dt.id = d.type_id
            LEFT JOIN wiki w ON w.id = d.wiki_id
            WHERE d.deleted_at IS NULL
              AND (d.wiki_id IS NULL OR w.deleted_at IS NULL)
              AND (:wiki_id IS NULL OR d.wiki_id = :wiki_id)
            ORDER BY d.title COLLATE NOCASE
            """,
            {"wiki_id": _wiki_id(conn, wiki) if wiki is not None else None},
        ).fetchall()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "type": row["type_name"],
            "wiki": row["wiki_name"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


@mcp.tool()
def read_document(document_id: int, include_cross_reference_content: bool = False) -> dict[str, Any]:
    """Return one document in full: metadata, every section, its references.

    Sections come back whole and in `position` order — the section is the
    return unit and nothing is truncated. A reference is always
    `{document_id, title, sections: null}` and never disappears, so the caller
    can decide what it needs.

    `include_cross_reference_content=True` adds each referenced document's
    sections, one level deep and no further. That can multiply the payload by
    the number of references, so pass it only when every referenced body is
    needed now; otherwise call `read_document` on the id you actually want.
    """
    with db.connection() as conn:
        row = _live_document(conn, document_id)
        refs = conn.execute(
            """
            SELECT d.id AS document_id, d.title
            FROM document_cross_references r
            JOIN document d ON d.id = r.target_document_id
            LEFT JOIN wiki w ON w.id = d.wiki_id
            WHERE r.source_document_id = :id
              AND d.deleted_at IS NULL
              AND (d.wiki_id IS NULL OR w.deleted_at IS NULL)
            ORDER BY d.title COLLATE NOCASE
            """,
            {"id": row["id"]},
        ).fetchall()
        references = [
            {
                "document_id": ref["document_id"],
                "title": ref["title"],
                "sections": (
                    _sections(conn, ref["document_id"])
                    if include_cross_reference_content
                    else None
                ),
            }
            for ref in refs
        ]
        payload = {
            "document_id": row["id"],
            "title": row["title"],
            "type": row["type_name"],
            "wiki": (
                {"id": row["wiki_id"], "name": row["wiki_name"]}
                if row["wiki_id"] is not None
                else None
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "sections": _sections(conn, row["id"]),
            "references": references,
        }
    return payload


@mcp.tool()
def log_recall(since: str | None = None, limit: int = 50) -> dict[str, Any]:
    """Read the store's single log, oldest first: what every write did.

    `since` is an ISO-8601 date or timestamp (`2026-11-22` works); `limit` caps
    the returned entries and keeps the newest ones. Each entry carries its
    progress document as `{document_id, title, wiki}` — read that document for
    the detail. Entries without a progress record (wiki creation) carry `null`.
    This is the session's opening read.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit: must be an integer >= 1")
    if since is not None:
        if not isinstance(since, str) or not _SINCE.match(since.strip()):
            raise ValueError("since: must be an ISO-8601 date or timestamp, e.g. 2026-11-22")

    with db.connection() as conn:
        log = conn.execute(
            """
            SELECT d.id FROM document d
            JOIN document_type dt ON dt.id = d.type_id
            WHERE d.wiki_id IS NULL AND dt.name = 'log' AND d.deleted_at IS NULL
            ORDER BY d.id
            LIMIT 1
            """
        ).fetchone()
        if log is None:
            return {"entries": []}

        rows = conn.execute(
            """
            SELECT position, body, created_at
            FROM document_content_section
            WHERE document_id = :id
              AND deleted_at IS NULL
              AND (:since IS NULL OR created_at >= :since)
            ORDER BY created_at DESC, position DESC
            LIMIT :limit
            """,
            {
                "id": log["id"],
                "since": since.strip() if since else None,
                "limit": limit,
            },
        ).fetchall()

        edges = conn.execute(
            """
            SELECT r.target_document_id, r.created_at, d.title, w.name AS wiki_name
            FROM document_cross_references r
            JOIN document d ON d.id = r.target_document_id
            LEFT JOIN wiki w ON w.id = d.wiki_id
            WHERE r.source_document_id = :id
              AND d.deleted_at IS NULL
              AND (d.wiki_id IS NULL OR w.deleted_at IS NULL)
            ORDER BY r.id
            """,
            {"id": log["id"]},
        ).fetchall()

    # A write appends its entry and its edge in one transaction, so they share
    # `created_at`. Pairing per timestamp keeps an entry with no edge (wiki
    # creation) empty, and stays correct when several writes land in one second.
    by_time: dict[str, deque[sqlite3.Row]] = defaultdict(deque)
    for edge in edges:
        by_time[edge["created_at"]].append(edge)

    entries: list[dict[str, Any]] = []
    for row in reversed(rows):
        queue = by_time.get(row["created_at"])
        progress = None
        if queue:
            edge = queue.popleft()
            progress = {
                "document_id": edge["target_document_id"],
                "title": edge["title"],
                "wiki": edge["wiki_name"],
            }
        entries.append(
            {
                "created_at": row["created_at"],
                "entry": row["body"],
                "progress": progress,
            }
        )
    return {"entries": entries}


# --- search: not implemented yet -------------------------------------------
# These three stay advertised so the surface does not silently shrink; each
# raises rather than returning an empty result that would read as "no matches".


@mcp.tool()
def semantic_search(query: str, wiki: str | None = None) -> list[dict[str, Any]]:
    """Vector match over section embeddings — the default search.

    The health gate runs first: vectors from a different model than the one
    configured are not comparable, so model drift refuses loudly with the
    three ways out (reembed, clear `openrouter_model`, don't use semantic
    search) instead of quietly returning matches against the wrong space.
    Sections mid-embed are simply absent — the vectors that exist are valid,
    and the worker finishes the rest.

    Returns whole sections, best match first, each carrying its document and
    wiki so the caller can follow up with `read_document`. Cosine similarity
    is computed over every live vector in Python: at document-edit volume
    brute force beats an index, and it keeps the store free of extensions.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query: must be a non-empty string")

    with db.connection() as conn:
        health = embedding_health(conn)
        if health["broken"]:
            raise EmbeddingUnhealthy(
                f"{health['broken']} section(s) have a corrupted embedding record "
                "(vector, model and text must be all set or all empty); semantic "
                "search is unsafe — reconcile with your user (doctor)"
            )
        if health["stale"]:
            raise EmbeddingUnhealthy(
                f"{health['stale']} section(s) are embedded with a model other than "
                f"{health['model']!r} (model drift); vectors from two models are not "
                "comparable. Choose: reembed the store, clear [embedding] "
                "openrouter_model, or don't use semantic search"
            )

        wiki_id = _wiki_id(conn, wiki) if wiki is not None else None
        rows = conn.execute(
            """
            SELECT s.id, s.position, s.heading, s.body, s.embedding, s.document_id,
                   d.title AS document_title, w.name AS wiki_name
            FROM document_content_section s
            JOIN document d ON d.id = s.document_id
            LEFT JOIN wiki w ON w.id = d.wiki_id
            WHERE s.deleted_at IS NULL
              AND s.embedding IS NOT NULL
              AND d.deleted_at IS NULL
              AND (d.wiki_id IS NULL OR w.deleted_at IS NULL)
              AND (:wiki_id IS NULL OR d.wiki_id = :wiki_id)
            """,
            {"wiki_id": wiki_id},
        ).fetchall()

    if not rows:
        return []

    query_vector = configured_embedder().embed([query.strip()])[0]
    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        vector = blob_to_vector(row["embedding"])
        if len(vector) != len(query_vector):
            # A dimension mismatch means the stored vector came from a model
            # with a different width than the configured one under the same
            # name — treat it as drift, not as a zero score.
            raise EmbeddingUnhealthy(
                f"section {row['id']} carries a {len(vector)}-dimension vector; "
                f"{health['model']!r} produces {len(query_vector)} — model drift, "
                "reembed or clear openrouter_model"
            )
        scored.append((_cosine(query_vector, vector), row))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "section_id": row["id"],
            "score": round(score, 6),
            "document_id": row["document_id"],
            "document_title": row["document_title"],
            "wiki": row["wiki_name"],
            "position": row["position"],
            "heading": row["heading"],
            "body": row["body"],
        }
        for score, row in scored
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Zero-norm vectors (a degenerate embed) score 0 —
    matching nothing beats dividing by zero."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


@mcp.tool()
def fts_search(query: str, wiki: str | None = None) -> list[dict]:
    """Exact-token match over section heading+body via FTS5."""
    raise NotImplementedError("stub — pending the read/search group")


@mcp.tool()
def table_search(wiki_id: int, contains: str | None = None) -> list[dict]:
    """Crawl wiki → documents → sections as plain rows; optional substring filter."""
    raise NotImplementedError("stub — pending the read/search group")
