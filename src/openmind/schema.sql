-- open-mind schema — five tables plus the FTS index over section text.
-- Approved 2026-11-21. Ships inside the package: db.bootstrap() applies this.
--
-- Version: 1  (PRAGMA user_version is stamped at the end; bump both together)

CREATE TABLE wiki (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,   -- NFKC-normalised, casing preserved; lookups COLLATE NOCASE
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

-- Type lookup is by name in every tool signature; keep the three stable ids.
INSERT OR IGNORE INTO document_type (name) VALUES ('document'), ('log'), ('progress');

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

PRAGMA user_version = 1;
