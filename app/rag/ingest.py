import sys
import hashlib
from pathlib import Path

from app.rag.document_loader import iter_pdf_pages
from app.rag.text_chunker import chunk_pages
from app.rag.embedding_service import embed_text
from app.rag.vector_store import (
    chunk_exists,
    store_chunks,
    register_document,
    update_document_progress,
    complete_document,
    fail_document,
)


DEFAULT_BATCH_SIZE = 20


def calculate_file_hash(
    file_path: str,
    block_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 incrementally.

    The file is never loaded entirely into memory, regardless
    of annual-report size.
    """

    digest = hashlib.sha256()

    with open(file_path, "rb") as file_handle:
        while True:
            block = file_handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def ingest_document(
    file_path: str,
    document_name: str | None = None,
    ticker: str | None = None,
    company_name: str | None = None,
    market: str | None = None,
    exchange: str | None = None,
    country: str | None = None,
    currency: str | None = None,
    document_type: str | None = None,
    report_type: str | None = None,
    reporting_period: str | None = None,
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
    fiscal_half: int | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
    publication_date: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    registry: dict | None = None,
) -> dict:
    """
    Incrementally ingest a financial-report PDF into EquityAI.

    No application-level document-size or page-count limit is
    imposed. Pages are extracted incrementally and vector chunks
    are written in bounded batches.

    Every newly ingested report is registered in the documents
    table and all new vector chunks are linked to its document_id.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if fiscal_quarter is not None and fiscal_quarter not in (1, 2, 3, 4):
        raise ValueError("fiscal_quarter must be between 1 and 4")

    if fiscal_half is not None and fiscal_half not in (1, 2):
        raise ValueError("fiscal_half must be 1 or 2")

    safe_document_name = (
        Path(document_name).name
        if document_name
        else Path(file_path).name
    )

    safe_ticker = ticker.strip().upper() if ticker else None
    safe_company_name = company_name.strip() if company_name else None
    safe_exchange = exchange.strip().upper() if exchange else None
    safe_currency = currency.strip().upper() if currency else None

    print(f"Streaming document: {safe_document_name}")

    if safe_company_name:
        print(f"Company: {safe_company_name}")
    if safe_ticker:
        print(f"Ticker: {safe_ticker}")
    if safe_exchange:
        print(f"Exchange: {safe_exchange}")

    # Hash and register documents for normal callers.
    #
    # Production workers may receive a registry record that was
    # created by the upload API before the job was queued. In that
    # case, do not register the same document a second time.
    if registry is None:
        document_hash = calculate_file_hash(file_path)

        registry = register_document(
            document_name=safe_document_name,
            document_hash=document_hash,
            company_name=safe_company_name,
            ticker=safe_ticker,
            market=market,
            exchange=safe_exchange,
            country=country,
            currency=safe_currency,
            document_type=document_type,
            report_type=report_type,
            reporting_period=reporting_period,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            fiscal_half=fiscal_half,
            period_start=period_start,
            period_end=period_end,
            publication_date=publication_date,
        )
    else:
        document_hash = registry.get("document_hash")

    document_id = registry["document_id"]
    registry_status = registry["status"]

    print(f"Document ID: {document_id}")
    print(f"Registry status: {registry_status}")

    # An exact SHA-256 duplicate that has already completed
    # ingestion must not generate embeddings again.
    if not registry["is_new"] and registry_status == "completed":
        print("Document already ingested. No work required.")

        return {
            "document_id": document_id,
            "document": safe_document_name,
            "document_hash": document_hash,
            "ticker": safe_ticker,
            "company_name": safe_company_name,
            "exchange": safe_exchange,
            "report_type": report_type,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "fiscal_half": fiscal_half,
            "period_start": period_start,
            "period_end": period_end,
            "pages": registry["total_pages"] or 0,
            "total_chunks": registry["total_chunks"] or 0,
            "new_chunks": 0,
            "existing_chunks": registry["total_chunks"] or 0,
            "inserted": 0,
            "skipped": registry["total_chunks"] or 0,
            "status": "already_ingested",
        }

    # Do not allow two workers to ingest the same exact file
    # concurrently. A newly-created registry row also has status
    # processing, so is_new must be checked here.
    if not registry["is_new"] and registry_status == "processing":
        raise RuntimeError(
            "This exact document is already being processed."
        )

    if not registry["is_new"] and registry_status == "failed":
        print(
            "Resuming failed ingestion. "
            f"Previously processed pages: "
            f"{registry['processed_pages']}"
        )

    total_pages = 0
    total_chunks = 0
    new_chunks_count = 0
    existing_count = 0
    inserted_count = 0
    insert_skipped_count = 0
    last_processed_page = 0

    pending_chunks = []
    pending_embeddings = []

    def flush_batch() -> None:
        nonlocal inserted_count
        nonlocal insert_skipped_count

        if not pending_chunks:
            return

        print(
            f"Writing batch of {len(pending_chunks)} chunks..."
        )

        result = store_chunks(
            pending_chunks,
            pending_embeddings,
            document_id=document_id,
        )

        inserted_count += result["inserted"]
        insert_skipped_count += result["skipped"]

        pending_chunks.clear()
        pending_embeddings.clear()

        update_document_progress(
            document_id=document_id,
            processed_pages=total_pages,
            processed_chunks=inserted_count,
            last_processed_page=last_processed_page,
        )

    try:
        for page in iter_pdf_pages(file_path):
            total_pages += 1
            last_processed_page = page["page"]

            # Preserve the original uploaded filename rather than
            # a temporary server-side upload filename.
            page["document"] = safe_document_name

            page_chunks = chunk_pages([page])
            total_chunks += len(page_chunks)

            for chunk in page_chunks:
                exists = chunk_exists(
                    document_id=document_id,
                    page=chunk["page"],
                    chunk=chunk["chunk"],
                )

                if exists:
                    existing_count += 1
                    continue

                embedding = embed_text(chunk["text"])

                pending_chunks.append(chunk)
                pending_embeddings.append(embedding)
                new_chunks_count += 1

                if len(pending_chunks) >= batch_size:
                    flush_batch()

            if total_pages % 25 == 0:
                # Persist progress even when recent pages contained
                # no new chunks.
                update_document_progress(
                    document_id=document_id,
                    processed_pages=total_pages,
                    processed_chunks=inserted_count,
                    last_processed_page=last_processed_page,
                )

                print(
                    "Progress: "
                    f"{total_pages} pages, "
                    f"{total_chunks} chunks, "
                    f"{inserted_count} inserted"
                )

        # Commit final partial batch.
        flush_batch()

        complete_document(
            document_id=document_id,
            total_pages=total_pages,
            total_chunks=total_chunks,
        )

    except Exception as exc:
        try:
            fail_document(
                document_id=document_id,
                error_message=str(exc),
            )
        except Exception as registry_exc:
            print(
                "WARNING: could not record ingestion failure: "
                f"{registry_exc}"
            )

        raise

    print("Ingestion complete.")
    print(f"Pages processed: {total_pages}")
    print(f"Total chunks: {total_chunks}")
    print(f"Inserted: {inserted_count}")

    return {
        "document_id": document_id,
        "document": safe_document_name,
        "document_hash": document_hash,
        "ticker": safe_ticker,
        "company_name": safe_company_name,
        "exchange": safe_exchange,
        "report_type": report_type,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "fiscal_half": fiscal_half,
        "period_start": period_start,
        "period_end": period_end,
        "pages": total_pages,
        "total_chunks": total_chunks,
        "new_chunks": new_chunks_count,
        "existing_chunks": existing_count,
        "inserted": inserted_count,
        "skipped": existing_count + insert_skipped_count,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m app.rag.ingest "
            "<path-to-pdf>"
        )
        sys.exit(1)

    ingest_document(sys.argv[1])
