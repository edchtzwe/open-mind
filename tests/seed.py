"""Deterministic fixture rows for the test store.

Separate from conftest so a test can seed a second store, and so assertions
can name these values instead of re-deriving them. Bodies carry unusual tokens
(`kestrel`, `basalt`) so full-text assertions cannot match by accident.
"""

from __future__ import annotations

import sqlite3

from openmind import db

WIKI = "test-wiki"

# title -> document_type.name
DOCUMENTS = {
    "Alpha Notes": "document",
    "Alpha Log": "log",
    "Alpha Progress": "progress",
}

# (document title, position, heading, body)
SECTIONS = (
    ("Alpha Notes", 1, "Overview", "kestrel roosts on the cliff"),
    ("Alpha Notes", 2, "Detail", "basalt columns and sea spray"),
    ("Alpha Progress", 1, "Done", "scaffold and persistence layer landed"),
)

# (source title, target title)
CROSS_REFERENCE = ("Alpha Log", "Alpha Progress")


def seed(conn: sqlite3.Connection) -> None:
    """Insert the fixture rows. Committed, so a second connection sees them."""
    now = db.utcnow()
    wiki_id = conn.execute(
        "INSERT INTO wiki (name, created_at, created_by, updated_at, updated_by)"
        " VALUES (?, ?, ?, ?, ?)",
        (WIKI, now, db.AUTHORED_BY, now, db.AUTHORED_BY),
    ).lastrowid

    document_ids: dict[str, int] = {}
    for title, type_name in DOCUMENTS.items():
        type_id = conn.execute(
            "SELECT id FROM document_type WHERE name = ?", (type_name,)
        ).fetchone()[0]
        document_ids[title] = conn.execute(
            "INSERT INTO document (wiki_id, type_id, title, created_at, created_by, updated_at, updated_by)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (wiki_id, type_id, title, now, db.AUTHORED_BY, now, db.AUTHORED_BY),
        ).lastrowid

    for title, position, heading, body in SECTIONS:
        conn.execute(
            "INSERT INTO document_content_section"
            " (document_id, position, heading, body, created_at, created_by, updated_at, updated_by)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document_ids[title],
                position,
                heading,
                body,
                now,
                db.AUTHORED_BY,
                now,
                db.AUTHORED_BY,
            ),
        )

    source, target = CROSS_REFERENCE
    conn.execute(
        "INSERT INTO document_cross_references"
        " (source_document_id, target_document_id, created_at, created_by) VALUES (?, ?, ?, ?)",
        (document_ids[source], document_ids[target], now, db.AUTHORED_BY),
    )

    conn.commit()
