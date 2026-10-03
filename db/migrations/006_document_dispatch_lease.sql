-- ============================================================
-- EquityAI Document Dispatch Lease
-- Migration 006
--
-- Records the period during which an API request owns dispatch
-- of a queued document to durable S3 + SQS infrastructure.
-- ============================================================

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS
        dispatch_lease_expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS
    idx_documents_dispatch_lease
ON documents(dispatch_lease_expires_at)
WHERE ingestion_status = 'queued';
