-- ============================================================
-- EquityAI Asynchronous Document Ingestion
-- Migration 004
--
-- Adds the queued lifecycle state required by the durable
-- S3 + SQS ingestion architecture.
-- ============================================================

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_status_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_status_check
    CHECK (
        ingestion_status IN (
            'pending',
            'queued',
            'processing',
            'completed',
            'failed'
        )
    );
