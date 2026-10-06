"""Coordinate durable preparation and queue dispatch."""

from app.infrastructure.ingestion_queue import (
    enqueue_ingestion_job,
)
from app.rag.ingestion_preparation import (
    prepare_document_ingestion,
)


def dispatch_document_ingestion(
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
    Persist an uploaded document and dispatch its ingestion job.

    If queue publication fails after registration/storage, leave the
    registry record queued so its dispatch lease can be reclaimed safely.
    """

    prepared = prepare_document_ingestion(
        file_path,
        document_name=document_name,
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

    if prepared["status"] in {
        "already_ingested",
        "already_queued",
    }:
        return prepared

    document_id = prepared["document_id"]
    storage = prepared["storage"]

    metadata = {
        "company_name": company_name,
        "ticker": ticker,
        "market": market,
        "exchange": exchange,
        "country": country,
        "currency": currency,
        "document_type": document_type,
        "report_type": report_type,
        "reporting_period": reporting_period,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "fiscal_half": fiscal_half,
        "period_start": period_start,
        "period_end": period_end,
        "publication_date": publication_date,
    }

    try:
        queue_result = enqueue_ingestion_job(
            document_id=document_id,
            document_hash=prepared["document_hash"],
            bucket=storage["bucket"],
            object_key=storage["object_key"],
            document_name=prepared["document"],
            metadata=metadata,
        )

    except Exception:
        # Keep the document queued. Its dispatch lease allows the
        # reconciler or a later redelivery to reclaim publication.
        raise

    return {
        **prepared,
        "status": "queued",
        "queue": {
            "message_id": queue_result["message_id"],
        },
    }
