from unittest.mock import patch

import pytest

from app.rag.ingestion_dispatch import (
    dispatch_document_ingestion,
)


@patch(
    "app.rag.ingestion_dispatch.fail_document"
)
@patch(
    "app.rag.ingestion_dispatch.enqueue_ingestion_job"
)
@patch(
    "app.rag.ingestion_dispatch.prepare_document_ingestion"
)
def test_dispatch_new_document(
    mock_prepare,
    mock_enqueue,
    mock_fail,
):
    mock_prepare.return_value = {
        "document_id": "doc-123",
        "document_hash": "hash-123",
        "document": "report.pdf",
        "status": "prepared",
        "registry": {
            "document_id": "doc-123",
            "status": "queued",
            "is_new": True,
        },
        "storage": {
            "bucket": "equityai-documents",
            "object_key": (
                "documents/doc-123/report.pdf"
            ),
        },
    }

    mock_enqueue.return_value = {
        "message_id": "message-123",
        "queue_url": "queue-url",
        "message": {},
    }

    result = dispatch_document_ingestion(
        "/tmp/report.pdf",
        document_name="report.pdf",
        company_name="Example Plc",
        ticker="EXM",
        exchange="NGX",
        country="Nigeria",
        currency="NGN",
        document_type="financial_report",
        report_type="annual",
        fiscal_year=2025,
        period_end="2025-12-31",
    )

    assert result["status"] == "queued"

    assert result["queue"] == {
        "message_id": "message-123",
    }

    mock_enqueue.assert_called_once()

    kwargs = mock_enqueue.call_args.kwargs

    assert kwargs["document_id"] == "doc-123"
    assert kwargs["document_hash"] == "hash-123"
    assert kwargs["bucket"] == "equityai-documents"
    assert (
        kwargs["object_key"]
        == "documents/doc-123/report.pdf"
    )

    assert kwargs["metadata"]["company_name"] == (
        "Example Plc"
    )
    assert kwargs["metadata"]["fiscal_year"] == 2025

    mock_fail.assert_not_called()


@patch(
    "app.rag.ingestion_dispatch.fail_document"
)
@patch(
    "app.rag.ingestion_dispatch.enqueue_ingestion_job"
)
@patch(
    "app.rag.ingestion_dispatch.prepare_document_ingestion"
)
def test_completed_duplicate_is_not_queued(
    mock_prepare,
    mock_enqueue,
    mock_fail,
):
    mock_prepare.return_value = {
        "document_id": "doc-existing",
        "document_hash": "hash-existing",
        "document": "report.pdf",
        "status": "already_ingested",
        "registry": {
            "document_id": "doc-existing",
            "status": "completed",
            "is_new": False,
        },
        "storage": None,
    }

    result = dispatch_document_ingestion(
        "/tmp/report.pdf",
        document_name="report.pdf",
    )

    assert result["status"] == "already_ingested"

    mock_enqueue.assert_not_called()
    mock_fail.assert_not_called()


@patch(
    "app.rag.ingestion_dispatch.fail_document"
)
@patch(
    "app.rag.ingestion_dispatch.enqueue_ingestion_job"
)
@patch(
    "app.rag.ingestion_dispatch.prepare_document_ingestion"
)
def test_queue_failure_marks_document_failed(
    mock_prepare,
    mock_enqueue,
    mock_fail,
):
    mock_prepare.return_value = {
        "document_id": "doc-123",
        "document_hash": "hash-123",
        "document": "report.pdf",
        "status": "prepared",
        "registry": {
            "document_id": "doc-123",
            "status": "queued",
            "is_new": True,
        },
        "storage": {
            "bucket": "equityai-documents",
            "object_key": (
                "documents/doc-123/report.pdf"
            ),
        },
    }

    mock_enqueue.side_effect = RuntimeError(
        "SQS unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="SQS unavailable",
    ):
        dispatch_document_ingestion(
            "/tmp/report.pdf",
            document_name="report.pdf",
        )

    mock_fail.assert_called_once()

    kwargs = mock_fail.call_args.kwargs

    assert kwargs["document_id"] == "doc-123"
    assert "Failed to dispatch ingestion job" in (
        kwargs["error_message"]
    )
    assert "SQS unavailable" in kwargs["error_message"]


@patch(
    "app.rag.ingestion_dispatch.fail_document"
)
@patch(
    "app.rag.ingestion_dispatch.enqueue_ingestion_job"
)
@patch(
    "app.rag.ingestion_dispatch.prepare_document_ingestion"
)
def test_queued_duplicate_is_not_queued_again(
    mock_prepare,
    mock_enqueue,
    mock_fail,
):
    mock_prepare.return_value = {
        "document_id": "doc-queued",
        "document_hash": "hash-queued",
        "document": "report.pdf",
        "status": "already_queued",
        "registry": {
            "document_id": "doc-queued",
            "status": "queued",
            "is_new": False,
        },
        "storage": None,
    }

    result = dispatch_document_ingestion(
        "/tmp/report.pdf",
        document_name="report.pdf",
    )

    assert result["document_id"] == "doc-queued"
    assert result["status"] == "already_queued"

    mock_enqueue.assert_not_called()
    mock_fail.assert_not_called()
