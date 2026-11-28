"""Deferred embedding: the write path never talks to a model.

Sections land in SQLite with NULL vectors inside the write transaction; the
enqueue happens *after* COMMIT, on purpose — the document is durable before
any partner integration runs, so a slow or dead OpenRouter can never hold up
or roll back a write. The crash in the gap between COMMIT and enqueue is the
one hole, and it is covered: those sections still have NULL vectors, which is
exactly what `embedding_health()` counts as pending and what `doctor
--reconcile` re-enqueues.

The Huey instance is module-level state, like `config.py`'s mtime memo and
`tools/__init__`'s MCPServer: the task registry must exist before anything
can enqueue, and a consumer process needs one stable task object to run.
Tests swap `embed_sections`, not the queue.

Consumer: `python -m openmind.worker` (one blocking HTTP call per batch).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from huey import SqliteHuey

from openmind import db, telemetry
from openmind.config import get
from openmind.embedding import configured_embedder, vector_to_blob

QUEUE_NAME = "openmind-embed"
DEFAULT_QUEUE_PATH = "huey.sqlite3"
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY_SECONDS = 15
# One task per write; inside it, embed in batches so a document of ten
# sections costs one network round trip, not ten.
BATCH = 64


def section_text(heading: str, body: str) -> str:
    """The exact text a vector is computed over — written back into
    `embedding_text` so an audit can re-run the same string and prove what
    the vector was made from."""
    return f"{heading}\n{body}" if heading else body


huey = SqliteHuey(
    QUEUE_NAME,
    filename=str(get("worker", "queue_path") or DEFAULT_QUEUE_PATH),
    results=False,  # no task ever returns a value anyone reads
)


def _retries() -> int:
    return max(0, int(get("worker", "max_retries") or DEFAULT_MAX_RETRIES))


def _retry_delay() -> int:
    return max(1, int(get("worker", "retry_delay_seconds") or DEFAULT_RETRY_DELAY_SECONDS))


@huey.task(retries=_retries(), retry_delay=_retry_delay())
def embed_sections(section_ids: list[int]) -> None:
    """Backfill vectors for the given sections. Returns None: Huey stores no
    result row, so the queue table holds pending tasks only.

    Each row is re-read at run time — a section deleted or already embedded
    since enqueue is skipped, never re-written. The `embedding IS NULL`
    predicate is the contract: a stale vector (model drift) is deliberately
    left alone; drift is the operator's to fix by reembedding, not something
    this worker loops over.
    """
    if not section_ids:
        return
    embedder = configured_embedder()
    # One local snapshot BEFORE any network call: each row is written back
    # only if it is still pending and its content is unchanged (the
    # `body IS :body` guard), so an edit racing between embed and write can
    # never have fresh content stamped with a vector of the stale text —
    # the fresh row just stays pending and gets enqueued again.
    with db.connection() as conn:
        # json_each, not a string-built placeholder list: the id set is
        # parameterised like every other query (RULES.md A03), however
        # untrusted the count ever is.
        rows = conn.execute(
            """
            SELECT id, document_id, position, heading, body
            FROM document_content_section
            WHERE id IN (SELECT value FROM json_each(:ids))
              AND embedding IS NULL AND deleted_at IS NULL
            """,
            {"ids": json.dumps(section_ids)},
        ).fetchall()
    if not rows:
        return
    snapshot = {row["id"]: (row["heading"], row["body"]) for row in rows}

    # The embed call happens OUTSIDE every transaction and outside the
    # connection above: a slow or dead provider must never hold SQLite's
    # writer lock, and a failure here leaves nothing half-written — the task
    # retries (or the reconciler re-enqueues) with the rows still pending.
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        texts = [section_text(row["heading"], row["body"]) for row in batch]
        vectors = embedder.embed(texts)
        if len(vectors) != len(batch):
            # Fewer vectors than texts would silently pair the wrong vector
            # with a row via zip. Refuse the batch; rows stay pending.
            raise RuntimeError(
                f"embedder returned {len(vectors)} vector(s) for "
                f"{len(batch)} text(s) — refusing the batch, rows stay pending"
            )
        with db.connection() as conn:
            _store(conn, embedder.model_name, batch, texts, vectors, snapshot)


def _store(
    conn: Any,
    model_name: str,
    rows: list[Any],
    texts: list[str],
    vectors: list[list[float]],
    snapshot: dict[int, tuple[str, str]],
) -> None:
    """Write the vectors. One transaction for the batch; a row is won only
    if it is still pending, still live, and byte-identical to the snapshot
    the vector was computed from — otherwise it stays NULL for a fresh job.

    `updated_at`/`updated_by` move with the vector on purpose — the doc-edit
    timestamp doubles as the embed audit trail: a fault window in the logs
    maps straight onto rows whose updated_at falls in it, tagged with the
    worker as author, so you comb logs by time and get the touched rows back.
    """
    now = db.utcnow()
    conn.execute("BEGIN IMMEDIATE")
    try:
        for row, text, vector in zip(rows, texts, vectors):
            snapshot_heading, snapshot_body = snapshot[row["id"]]
            conn.execute(
                """
                UPDATE document_content_section
                SET embedding = :embedding, embedding_model = :model,
                    embedding_text = :text, updated_at = :now, updated_by = :by
                WHERE id = :id AND embedding IS NULL AND deleted_at IS NULL
                  AND heading IS :snap_heading AND body IS :snap_body
                """,
                {
                    "embedding": vector_to_blob(vector),
                    "model": model_name,
                    "text": text,
                    "now": now,
                    "by": db.WORKER,
                    "id": row["id"],
                    "snap_heading": snapshot_heading,
                    "snap_body": snapshot_body,
                },
            )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    telemetry.log_event(
        "embed.backfill",
        source="worker",
        model=model_name,
        sections=len(rows),
        document_ids=sorted({row["document_id"] for row in rows}),
        section_ids=[row["id"] for row in rows],
    )


def enqueue(section_ids: list[int]) -> bool:
    """Queue one backfill task. Called after COMMIT, never inside the write
    transaction: the document is already durable, so an enqueue failure must
    not undo it. Returns False when the queue refuses — the sections keep
    their NULL vectors and `doctor --reconcile` picks them up.

    Enqueueing ids that were never written (or already embedded) is harmless:
    the task's WHERE filters them out.
    """
    if not section_ids:
        return False
    try:
        embed_sections(section_ids)
    except Exception as exc:  # noqa: BLE001 — logged, then downgraded
        telemetry.log_event(
            "embed.enqueue_failed",
            level=logging.WARNING,
            error_type=type(exc).__name__,
            error_message=str(exc),
            section_ids=section_ids,
        )
        return False
    return True


def pending_section_ids(conn: Any, limit: int = 1000) -> list[int]:
    """The reconciler's query: live sections with no vector, oldest first.

    Backed by the partial index on (document_id) WHERE embedding IS NULL —
    the same count `embedding_health()` reports, so what doctor counts is
    exactly what reconcile enqueues.
    """
    rows = conn.execute(
        """
        SELECT id FROM document_content_section
        WHERE embedding IS NULL AND deleted_at IS NULL
        ORDER BY id
        LIMIT :limit
        """,
        {"limit": limit},
    ).fetchall()
    return [row["id"] for row in rows]


def main() -> None:
    """Run the consumer. Blocking forever; one section-batch per dequeue."""
    telemetry.configure()
    telemetry.log_event(
        "worker.starting",
        queue=huey.name,
        queue_path=str(get("worker", "queue_path") or DEFAULT_QUEUE_PATH),
    )
    huey.create_consumer().run()


if __name__ == "__main__":
    main()
