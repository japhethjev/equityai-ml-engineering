-- ============================================================
-- EquityAI Document Processing Lease
-- Migration 005
--
-- Allows an SQS worker to recover a document left in
-- 'processing' after an ECS task terminates unexpectedly.
-- ============================================================

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS
        processing_lease_expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS
    idx_documents_processing_lease
ON documents(processing_lease_expires_at)
WHERE ingestion_status = 'processing';
