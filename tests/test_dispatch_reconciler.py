from unittest.mock import patch

import pytest

from app.workers.dispatch_reconciler import (
    reconcile_one_orphaned_dispatch,
)


def orphaned_document():
    return {
        "document_id": "doc-123",
        "document_hash": "hash-123",
        "document_name": "report.pdf",
        "metadata": {
            "company_name": "Example Plc",
            "ticker": "EXM",
            "market": "Nigeria",
            "exchange": "NGX",
            "country": "Nigeria",
            "currency": "NGN",
            "document_type": "financial_report",
            "report_type": "annual",
            "reporting_period": "FY2025",
            "fiscal_year": 2025,
            "fiscal_quarter": None,
            "fiscal_half": None,
            "period_start": "2025-01-01",
            "period_end": "2025-12-31",
            "publication_date": "2026-03-01",
        },
    }


@patch(
    "app.workers.dispatch_reconciler."
    "claim_next_orphaned_dispatch"
)
def test_reconciler_is_idle_when_no_orphan(mock_claim):
    mock_claim.return_value = None

    result = reconcile_one_orphaned_dispatch()

    assert result == {
        "status": "idle",
        "reconciled": False,
    }


@patch(
    "app.workers.dispatch_reconciler.fail_document"
)
@patch(
    "app.workers.dispatch_reconciler.enqueue_ingestion_job"
)
@patch(
    "app.workers.dispatch_reconciler.document_object_exists"
)
@patch(
    "app.workers.dispatch_reconciler.get_document_bucket"
)
@patch(
    "app.workers.dispatch_reconciler."
    "claim_next_orphaned_dispatch"
)
def test_reconciler_redispatches_existing_object(
    mock_claim,
    mock_bucket,
    mock_exists,
    mock_enqueue,
    mock_fail,
):
    mock_claim.return_value = orphaned_document()
    mock_bucket.return_value = "equityai-documents"
    mock_exists.return_value = True
    mock_enqueue.return_value = {
        "message_id": "message-123",
    }

    result = reconcile_one_orphaned_dispatch()

    assert result == {
        "document_id": "doc-123",
        "status": "redispatched",
        "reconciled": True,
        "queue": {
            "message_id": "message-123",
        },
    }

    mock_exists.assert_called_once_with(
        bucket="equityai-documents",
        object_key="documents/doc-123/report.pdf",
    )

    mock_enqueue.assert_called_once_with(
        document_id="doc-123",
        document_hash="hash-123",
        bucket="equityai-documents",
        object_key="documents/doc-123/report.pdf",
        document_name="report.pdf",
        metadata=orphaned_document()["metadata"],
    )

    mock_fail.assert_not_called()


@patch(
    "app.workers.dispatch_reconciler.fail_document"
)
@patch(
    "app.workers.dispatch_reconciler.enqueue_ingestion_job"
)
@patch(
    "app.workers.dispatch_reconciler.document_object_exists"
)
@patch(
    "app.workers.dispatch_reconciler.get_document_bucket"
)
@patch(
    "app.workers.dispatch_reconciler."
    "claim_next_orphaned_dispatch"
)
def test_missing_s3_object_marks_document_failed(
    mock_claim,
    mock_bucket,
    mock_exists,
    mock_enqueue,
    mock_fail,
):
    mock_claim.return_value = orphaned_document()
    mock_bucket.return_value = "equityai-documents"
    mock_exists.return_value = False

    result = reconcile_one_orphaned_dispatch()

    assert result["document_id"] == "doc-123"
    assert result["status"] == "failed"
    assert result["reconciled"] is True

    mock_fail.assert_called_once()

    kwargs = mock_fail.call_args.kwargs

    assert kwargs["document_id"] == "doc-123"
    assert "missing from S3" in kwargs["error_message"]

    mock_enqueue.assert_not_called()


@patch(
    "app.workers.dispatch_reconciler.fail_document"
)
@patch(
    "app.workers.dispatch_reconciler.enqueue_ingestion_job"
)
@patch(
    "app.workers.dispatch_reconciler.document_object_exists"
)
@patch(
    "app.workers.dispatch_reconciler.get_document_bucket"
)
@patch(
    "app.workers.dispatch_reconciler."
    "claim_next_orphaned_dispatch"
)
def test_s3_error_is_retried_after_lease_expiry(
    mock_claim,
    mock_bucket,
    mock_exists,
    mock_enqueue,
    mock_fail,
):
    mock_claim.return_value = orphaned_document()
    mock_bucket.return_value = "equityai-documents"
    mock_exists.side_effect = RuntimeError(
        "S3 unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="S3 unavailable",
    ):
        reconcile_one_orphaned_dispatch()

    mock_enqueue.assert_not_called()
    mock_fail.assert_not_called()


@patch(
    "app.workers.dispatch_reconciler.fail_document"
)
@patch(
    "app.workers.dispatch_reconciler.enqueue_ingestion_job"
)
@patch(
    "app.workers.dispatch_reconciler.document_object_exists"
)
@patch(
    "app.workers.dispatch_reconciler.get_document_bucket"
)
@patch(
    "app.workers.dispatch_reconciler."
    "claim_next_orphaned_dispatch"
)
def test_sqs_error_is_retried_after_lease_expiry(
    mock_claim,
    mock_bucket,
    mock_exists,
    mock_enqueue,
    mock_fail,
):
    mock_claim.return_value = orphaned_document()
    mock_bucket.return_value = "equityai-documents"
    mock_exists.return_value = True

    mock_enqueue.side_effect = RuntimeError(
        "SQS unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="SQS unavailable",
    ):
        reconcile_one_orphaned_dispatch()

    mock_fail.assert_not_called()


@patch(
    "app.workers.dispatch_reconciler."
    "reconcile_one_orphaned_dispatch"
)
def test_reconcile_batch_stops_when_idle(mock_reconcile):
    from app.workers.dispatch_reconciler import (
        reconcile_orphaned_dispatches,
    )

    mock_reconcile.side_effect = [
        {
            "document_id": "doc-1",
            "status": "redispatched",
            "reconciled": True,
        },
        {
            "document_id": "doc-2",
            "status": "redispatched",
            "reconciled": True,
        },
        {
            "status": "idle",
            "reconciled": False,
        },
    ]

    result = reconcile_orphaned_dispatches(
        max_documents=10
    )

    assert result == 2
    assert mock_reconcile.call_count == 3


@patch(
    "app.workers.dispatch_reconciler."
    "reconcile_one_orphaned_dispatch"
)
def test_reconcile_batch_respects_limit(mock_reconcile):
    from app.workers.dispatch_reconciler import (
        reconcile_orphaned_dispatches,
    )

    mock_reconcile.return_value = {
        "document_id": "doc-1",
        "status": "redispatched",
        "reconciled": True,
    }

    result = reconcile_orphaned_dispatches(
        max_documents=3
    )

    assert result == 3
    assert mock_reconcile.call_count == 3


def test_reconcile_batch_requires_positive_limit():
    from app.workers.dispatch_reconciler import (
        reconcile_orphaned_dispatches,
    )

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        reconcile_orphaned_dispatches(
            max_documents=0
        )
