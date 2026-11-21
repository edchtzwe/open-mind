-- open-mind schema — five tables, approved 2026-11-21.
-- Copied verbatim from the Open Mind Schema design page.

CREATE TABLE wiki (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,   -- NFKC-normalised, lowercase
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    deleted_at  TEXT
);

CREATE TABLE document_type (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE   -- seeded: 'document', 'log', 'progress'
);

CREATE TABLE document (
    id          INTEGER PRIMARY KEY,
    wiki_id     INTEGER REFERENCES wiki(id),   -- NULL: root/cross-cutting document
    type_id     INTEGER NOT NULL REFERENCES document_type(id),
    title       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    deleted_at  TEXT,
    UNIQUE (wiki_id, title)
);

CREATE TABLE document_content_section (
    id              INTEGER PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES document(id),
    position        INTEGER NOT NULL,      -- 1-based, dense within the document
    heading         TEXT    NOT NULL,      -- '' legal for a lead section
    body            TEXT    NOT NULL,      -- Markdown
    embedding       BLOB,
    embedding_model TEXT,                  -- model that produced the vector
    embedding_text  TEXT,                  -- exact text embedded, for debugging
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    deleted_at      TEXT,
    UNIQUE (document_id, position),
    CHECK (
        (embedding IS NULL AND embedding_model IS NULL AND embedding_text IS NULL)
        OR (embedding IS NOT NULL AND embedding_model IS NOT NULL AND embedding_text IS NOT NULL)
    )
);

CREATE TABLE document_cross_references (
    id                 INTEGER PRIMARY KEY,
    source_document_id INTEGER NOT NULL REFERENCES document(id),
    target_document_id INTEGER NOT NULL REFERENCES document(id),
    created_at         TEXT    NOT NULL,
    UNIQUE (source_document_id, target_document_id),
    CHECK (source_document_id <> target_document_id)
);
