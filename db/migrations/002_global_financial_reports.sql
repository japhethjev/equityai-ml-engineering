-- ============================================================
-- EquityAI Global Financial Report Metadata
-- Migration 002
-- ============================================================

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS exchange TEXT,
    ADD COLUMN IF NOT EXISTS currency TEXT,
    ADD COLUMN IF NOT EXISTS report_type TEXT,
    ADD COLUMN IF NOT EXISTS fiscal_year INTEGER,
    ADD COLUMN IF NOT EXISTS fiscal_quarter INTEGER,
    ADD COLUMN IF NOT EXISTS fiscal_half INTEGER,
    ADD COLUMN IF NOT EXISTS period_start DATE,
    ADD COLUMN IF NOT EXISTS period_end DATE,
    ADD COLUMN IF NOT EXISTS filing_version INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS supersedes_document_id UUID;

-- Quarterly reports may be Q1-Q4.
-- Annual, half-year and other filings leave this NULL.
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_fiscal_quarter_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_fiscal_quarter_check
    CHECK (
        fiscal_quarter IS NULL
        OR fiscal_quarter BETWEEN 1 AND 4
    );

-- Half-year reports may be H1/H2.
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_fiscal_half_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_fiscal_half_check
    CHECK (
        fiscal_half IS NULL
        OR fiscal_half BETWEEN 1 AND 2
    );

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_filing_version_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_filing_version_check
    CHECK (filing_version >= 1);

-- Allow future markets/reporting conventions without forcing
-- report_type into a rigid database enum.
CREATE INDEX IF NOT EXISTS
    idx_documents_security
ON documents(exchange, ticker);

CREATE INDEX IF NOT EXISTS
    idx_documents_fiscal_year
ON documents(fiscal_year);

CREATE INDEX IF NOT EXISTS
    idx_documents_reporting_identity
ON documents(
    exchange,
    ticker,
    report_type,
    fiscal_year,
    fiscal_quarter,
    fiscal_half,
    period_end
);

CREATE INDEX IF NOT EXISTS
    idx_documents_current_reports
ON documents(exchange, ticker, is_current);

CREATE INDEX IF NOT EXISTS
    idx_documents_period_end
ON documents(period_end);

CREATE INDEX IF NOT EXISTS
    idx_documents_supersedes
ON documents(supersedes_document_id);

-- A revised/restated filing can point to the document it supersedes.
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_supersedes_fk;

ALTER TABLE documents
    ADD CONSTRAINT documents_supersedes_fk
    FOREIGN KEY (supersedes_document_id)
    REFERENCES documents(document_id)
    ON DELETE SET NULL;
