-- ============================================================
-- EquityAI RAG Document Chunk Uniqueness
-- Migration 003
--
-- Prevent duplicate chunks within the same registered document.
-- Legacy chunks with document_id IS NULL remain unaffected.
-- ============================================================

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_document_chunks_unique_document_page_chunk
ON document_chunks (
    document_id,
    page,
    chunk
)
WHERE document_id IS NOT NULL;
