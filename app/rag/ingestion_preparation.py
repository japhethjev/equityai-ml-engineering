"""Prepare uploaded documents for asynchronous ingestion."""

from pathlib import Path

from app.infrastructure.document_storage import (
    upload_document_file,
)
from app.rag.ingest import calculate_file_hash
from app.rag.vector_store import (
    claim_document_for_dispatch,
    fail_document,
    register_document,
)


def prepare_document_ingestion(
    file_path: str,
    *,
    document_name: str,
    company_name: str | None = None,
    ticker: str | None = None,
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
) -> dict:
    """
    Register an uploaded document and persist it to durable storage.

    Expensive PDF parsing, chunking and embedding are deliberately
    excluded. Those operations belong to the ingestion worker.
    """

    source_path = Path(file_path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Document file does not exist: {file_path}"
        )

    safe_document_name = Path(document_name).name

    if not safe_document_name:
        raise ValueError("document_name is required")

    document_hash = calculate_file_hash(
        str(source_path)
    )

    registry = register_document(
        document_name=safe_document_name,
        document_hash=document_hash,
        company_name=company_name,
        ticker=ticker,
        market=market,
        exchange=exchange,
        country=country,
        currency=currency,
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

    # Completed exact duplicates need no storage upload and no
    # additional ingestion job.
    if (
        not registry["is_new"]
        and registry["status"] == "completed"
    ):
        return {
            "document_id": registry["document_id"],
            "document_hash": document_hash,
            "document": safe_document_name,
            "status": "already_ingested",
            "registry": registry,
            "storage": None,
        }

    # A queued duplicate may represent either a legitimate job
    # awaiting a worker or an orphaned dispatch whose API process
    # died before publishing to SQS. Only an expired dispatch lease
    # may be atomically reclaimed for redispatch.
    if (
        not registry["is_new"]
        and registry["status"] == "queued"
    ):
        dispatch_claim = claim_document_for_dispatch(
            registry["document_id"]
        )

        if not dispatch_claim["claimed"]:
            return {
                "document_id": registry["document_id"],
                "document_hash": document_hash,
                "document": safe_document_name,
                "status": "already_queued",
                "registry": registry,
                "storage": None,
            }

        registry = {
            **dispatch_claim,
            "document_hash": document_hash,
        }

    # A second request must not create another ingestion job while
    # the exact same document is already being processed.
    if (
        not registry["is_new"]
        and registry["status"] == "processing"
    ):
        raise RuntimeError(
            "This exact document is already being processed."
        )

    try:
        storage = upload_document_file(
            file_path=str(source_path),
            document_id=registry["document_id"],
            filename=safe_document_name,
        )

    except Exception as exc:
        fail_document(
            document_id=registry["document_id"],
            error_message=(
                "Failed to persist document to S3: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        raise

    return {
        "document_id": registry["document_id"],
        "document_hash": document_hash,
        "document": safe_document_name,
        "status": "prepared",
        "registry": {
            **registry,
            "document_hash": document_hash,
        },
        "storage": storage,
    }
