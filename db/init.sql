CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document TEXT NOT NULL,
    page INTEGER NOT NULL,
    chunk INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),

    CONSTRAINT unique_document_chunk
        UNIQUE (document, page, chunk)
);