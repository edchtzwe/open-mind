-- open-mind schema — five tables plus the FTS index over section text.
-- Approved 2026-11-21. Ships inside the package: db.bootstrap() applies this.
--
-- Version: 2  (PRAGMA user_version is stamped at the end; bump both together)
-- v2: every *_at column gained its *_by counterpart — a timestamp with no
--     author cannot answer "who wrote this". deleted_* stay paired by CHECK.

CREATE TABLE wiki (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,   -- NFKC-normalised, casing preserved; lookups COLLATE NOCASE
    created_at  TEXT    NOT NULL,
    created_by  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    updated_by  TEXT    NOT NULL,
    deleted_at  TEXT,
    deleted_by  TEXT,
    CHECK ((deleted_at IS NULL AND deleted_by IS NULL)
        OR (deleted_at IS NOT NULL AND deleted_by IS NOT NULL))
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
    created_by  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    updated_by  TEXT    NOT NULL,
    deleted_at  TEXT,
    deleted_by  TEXT,
    UNIQUE (wiki_id, title),
    CHECK ((deleted_at IS NULL AND deleted_by IS NULL)
        OR (deleted_at IS NOT NULL AND deleted_by IS NOT NULL))
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
    created_by      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    updated_by      TEXT    NOT NULL,
    deleted_at      TEXT,
    deleted_by      TEXT,
    UNIQUE (document_id, position),
    CHECK (
        (embedding IS NULL AND embedding_model IS NULL AND embedding_text IS NULL)
        OR (embedding IS NOT NULL AND embedding_model IS NOT NULL AND embedding_text IS NOT NULL)
    ),
    CHECK ((deleted_at IS NULL AND deleted_by IS NULL)
        OR (deleted_at IS NOT NULL AND deleted_by IS NOT NULL))
);

CREATE TABLE document_cross_references (
    id                 INTEGER PRIMARY KEY,
    source_document_id INTEGER NOT NULL REFERENCES document(id),
    target_document_id INTEGER NOT NULL REFERENCES document(id),
    created_at         TEXT    NOT NULL,
    created_by         TEXT    NOT NULL,
    UNIQUE (source_document_id, target_document_id),
    CHECK (source_document_id <> target_document_id)
);

-- Type lookup is by name in every tool signature; keep the three stable ids.
INSERT OR IGNORE INTO document_type (id, name) VALUES (1, 'document'), (2, 'log'), (3, 'progress');

-- Search: external-content FTS5, so the index holds words only and the text
-- lives once, in document_content_section. Triggers keep it in step inside the
-- same write transaction — no reindex step, no path where they diverge.
CREATE VIRTUAL TABLE section_fts USING fts5(
    heading,
    body,
    content='document_content_section',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER section_fts_after_insert AFTER INSERT ON document_content_section BEGIN
    INSERT INTO section_fts (rowid, heading, body)
    VALUES (new.id, new.heading, new.body);
END;

-- External content requires the old row's values on delete, hence the 'delete'
-- command with explicit values rather than a plain DELETE.
CREATE TRIGGER section_fts_after_delete AFTER DELETE ON document_content_section BEGIN
    INSERT INTO section_fts (section_fts, rowid, heading, body)
    VALUES ('delete', old.id, old.heading, old.body);
END;

CREATE TRIGGER section_fts_after_update AFTER UPDATE OF heading, body ON document_content_section BEGIN
    INSERT INTO section_fts (section_fts, rowid, heading, body)
    VALUES ('delete', old.id, old.heading, old.body);
    INSERT INTO section_fts (rowid, heading, body)
    VALUES (new.id, new.heading, new.body);
END;

-- list_documents and table_search scan by wiki and type; log_recall scans by time.
CREATE INDEX document_wiki_type_idx ON document (wiki_id, type_id);
CREATE INDEX document_created_at_idx ON document (created_at);
CREATE INDEX document_cross_references_target_idx
    ON document_cross_references (target_document_id);

-- embedding_health()'s pending count: the reconciler and semantic_search both
-- ask "how many live sections have no vector yet" on every call.
CREATE INDEX section_pending_embedding_idx
    ON document_content_section (document_id)
    WHERE embedding IS NULL AND deleted_at IS NULL;

PRAGMA user_version = 2;
