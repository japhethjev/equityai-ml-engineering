-- ============================================================
-- EquityAI RAG Document Registry
-- Migration 001
--
-- Adds first-class document/equity metadata without removing
-- or changing existing document_chunks data.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,

    -- Stable application identifier
    document_id UUID NOT NULL DEFAULT gen_random_uuid(),

    -- File identity / duplicate detection
    document_name TEXT NOT NULL,
    document_hash TEXT,

    -- Equity identity
    company_name TEXT,
    ticker TEXT,
    market TEXT,
    country TEXT,

    -- Filing/report metadata
    document_type TEXT,
    reporting_period TEXT,
    publication_date DATE,

    -- Ingestion lifecycle
    ingestion_status TEXT NOT NULL DEFAULT 'pending',
    total_pages INTEGER,
    processed_pages INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER,
    processed_chunks INTEGER NOT NULL DEFAULT 0,

    -- Failure/resume information
    last_processed_page INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT documents_document_id_unique
        UNIQUE (document_id),

    CONSTRAINT documents_status_check
        CHECK (
            ingestion_status IN (
                'pending',
                'processing',
                'completed',
                'failed'
            )
        ),

    CONSTRAINT documents_pages_nonnegative
        CHECK (
            processed_pages >= 0
            AND last_processed_page >= 0
        ),

    CONSTRAINT documents_chunks_nonnegative
        CHECK (processed_chunks >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_documents_hash
ON documents(document_hash)
WHERE document_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS
    idx_documents_ticker
ON documents(ticker);

CREATE INDEX IF NOT EXISTS
    idx_documents_market_ticker
ON documents(market, ticker);

CREATE INDEX IF NOT EXISTS
    idx_documents_company
ON documents(company_name);

CREATE INDEX IF NOT EXISTS
    idx_documents_type
ON documents(document_type);

CREATE INDEX IF NOT EXISTS
    idx_documents_reporting_period
ON documents(reporting_period);

CREATE INDEX IF NOT EXISTS
    idx_documents_status
ON documents(ingestion_status);


-- ------------------------------------------------------------
-- Link existing vector chunks to the document registry.
--
-- Nullable initially so all existing production chunks remain
-- valid while old documents are backfilled.
-- ------------------------------------------------------------

ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS document_id UUID;

CREATE INDEX IF NOT EXISTS
    idx_document_chunks_document_id
ON document_chunks(document_id);

CREATE INDEX IF NOT EXISTS
    idx_document_chunks_document_page
ON document_chunks(document_id, page);


-- ------------------------------------------------------------
-- Keep updated_at current automatically.
-- ------------------------------------------------------------

CREATE OR REPLACE FUNCTION
update_documents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS
    trg_documents_updated_at
ON documents;

CREATE TRIGGER trg_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW
EXECUTE FUNCTION update_documents_updated_at();
