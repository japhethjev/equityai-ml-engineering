import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.workers.ingestion_worker import (
    process_ingestion_message,
    validate_ingestion_message,
)


def make_message():
    return {
        "version": 1,
        "document_id": "doc-123",
        "document_hash": "hash-123",
        "bucket": "equityai-documents",
        "object_key": (
            "documents/doc-123/report.pdf"
        ),
        "document_name": "report.pdf",
        "metadata": {
            "company_name": "Example Plc",
            "ticker": "EXM",
            "exchange": "NGX",
            "country": "Nigeria",
            "currency": "NGN",
            "document_type": "financial_report",
            "report_type": "annual",
            "fiscal_year": 2025,
            "period_end": "2025-12-31",
        },
    }


def test_validate_ingestion_message():
    validate_ingestion_message(
        make_message()
    )


def test_validate_rejects_wrong_version():
    message = make_message()
    message["version"] = 2

    with pytest.raises(
        ValueError,
        match="version",
    ):
        validate_ingestion_message(message)


def test_validate_rejects_missing_document_id():
    message = make_message()
    message["document_id"] = ""

    with pytest.raises(
        ValueError,
        match="document_id",
    ):
        validate_ingestion_message(message)


def test_validate_rejects_invalid_metadata():
    message = make_message()
    message["metadata"] = "invalid"

    with pytest.raises(
        ValueError,
        match="metadata",
    ):
        validate_ingestion_message(message)


@patch(
    "app.workers.ingestion_worker.ingest_document"
)
@patch(
    "app.workers.ingestion_worker."
    "download_document_file"
)
@patch(
    "app.workers.ingestion_worker."
    "claim_document_for_ingestion"
)
def test_process_ingestion_message(
    mock_claim,
    mock_download,
    mock_ingest,
):
    message = make_message()

    mock_claim.return_value = {
        "document_id": "doc-123",
        "status": "processing",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": True,
        "claimed": True,
    }

    mock_ingest.return_value = {
        "document_id": "doc-123",
        "status": "completed",
        "inserted": 10,
    }

    result = process_ingestion_message(
        message
    )

    mock_download.assert_called_once()

    download_kwargs = (
        mock_download.call_args.kwargs
    )

    assert (
        download_kwargs["bucket"]
        == "equityai-documents"
    )

    assert (
        download_kwargs["object_key"]
        == "documents/doc-123/report.pdf"
    )

    assert (
        Path(
            download_kwargs[
                "destination_path"
            ]
        ).name
        == "report.pdf"
    )

    mock_ingest.assert_called_once()

    ingest_args = mock_ingest.call_args

    assert (
        Path(ingest_args.args[0]).name
        == "report.pdf"
    )

    kwargs = ingest_args.kwargs

    assert kwargs["document_name"] == (
        "report.pdf"
    )

    assert kwargs["company_name"] == (
        "Example Plc"
    )

    assert kwargs["ticker"] == "EXM"

    assert kwargs["registry"] == {
        "document_id": "doc-123",
        "document_hash": "hash-123",
        "status": "processing",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": True,
        "claimed": True,
    }

    assert result["status"] == "completed"


@patch(
    "app.workers.ingestion_worker.ingest_document"
)
@patch(
    "app.workers.ingestion_worker."
    "download_document_file"
)
@patch(
    "app.workers.ingestion_worker."
    "claim_document_for_ingestion"
)
def test_worker_propagates_ingestion_failure(
    mock_claim,
    mock_download,
    mock_ingest,
):
    message = make_message()

    mock_claim.return_value = {
        "document_id": "doc-123",
        "status": "processing",
        "processed_pages": 0,
        "processed_chunks": 0,
        "last_processed_page": 0,
        "total_pages": None,
        "total_chunks": None,
        "is_new": True,
        "claimed": True,
    }

    mock_ingest.side_effect = RuntimeError(
        "Embedding service unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding service unavailable",
    ):
        process_ingestion_message(
            message
        )

    mock_download.assert_called_once()
    mock_ingest.assert_called_once()


@patch(
    "app.workers.ingestion_worker.ingest_document"
)
@patch(
    "app.workers.ingestion_worker."
    "download_document_file"
)
@patch(
    "app.workers.ingestion_worker."
    "claim_document_for_ingestion"
)
def test_completed_redelivery_skips_processing(
    mock_claim,
    mock_download,
    mock_ingest,
):
    mock_claim.return_value = {
        "document_id": "doc-123",
        "status": "completed",
        "processed_pages": 100,
        "processed_chunks": 300,
        "last_processed_page": 100,
        "total_pages": 100,
        "total_chunks": 300,
        "is_new": False,
        "claimed": False,
    }

    result = process_ingestion_message(
        make_message()
    )

    assert result == {
        "document_id": "doc-123",
        "status": "already_completed",
        "processed": False,
    }

    mock_download.assert_not_called()
    mock_ingest.assert_not_called()


@patch(
    "app.workers.ingestion_worker.ingest_document"
)
@patch(
    "app.workers.ingestion_worker."
    "download_document_file"
)
@patch(
    "app.workers.ingestion_worker."
    "claim_document_for_ingestion"
)
def test_concurrent_redelivery_skips_processing(
    mock_claim,
    mock_download,
    mock_ingest,
):
    mock_claim.return_value = {
        "document_id": "doc-123",
        "status": "processing",
        "processed_pages": 10,
        "processed_chunks": 30,
        "last_processed_page": 10,
        "total_pages": None,
        "total_chunks": None,
        "is_new": False,
        "claimed": False,
    }

    result = process_ingestion_message(
        make_message()
    )

    assert result == {
        "document_id": "doc-123",
        "status": "already_processing",
        "processed": False,
    }

    mock_download.assert_not_called()
    mock_ingest.assert_not_called()


@patch(
    "app.workers.ingestion_worker."
    "process_ingestion_message"
)
def test_run_worker_once_deletes_successful_message(
    mock_process,
):
    from unittest.mock import MagicMock

    from app.workers.ingestion_worker import (
        run_worker_once,
    )

    sqs = MagicMock()

    sqs.receive_message.return_value = {
        "Messages": [
            {
                "Body": (
                    '{"version":1,'
                    '"document_id":"doc-123"}'
                ),
                "ReceiptHandle": "receipt-123",
            }
        ]
    }

    mock_process.return_value = {
        "document_id": "doc-123",
        "status": "completed",
    }

    handled = run_worker_once(
        sqs_client=sqs,
        queue_url="queue-url",
    )

    assert handled == 1

    mock_process.assert_called_once_with(
        {
            "version": 1,
            "document_id": "doc-123",
        }
    )

    sqs.delete_message.assert_called_once_with(
        QueueUrl="queue-url",
        ReceiptHandle="receipt-123",
    )


@patch(
    "app.workers.ingestion_worker."
    "process_ingestion_message"
)
def test_run_worker_once_does_not_delete_failed_message(
    mock_process,
):
    from unittest.mock import MagicMock

    from app.workers.ingestion_worker import (
        run_worker_once,
    )

    sqs = MagicMock()

    sqs.receive_message.return_value = {
        "Messages": [
            {
                "Body": (
                    '{"version":1,'
                    '"document_id":"doc-123"}'
                ),
                "ReceiptHandle": "receipt-123",
            }
        ]
    }

    mock_process.side_effect = RuntimeError(
        "OpenAI unavailable"
    )

    handled = run_worker_once(
        sqs_client=sqs,
        queue_url="queue-url",
    )

    assert handled == 0

    sqs.delete_message.assert_not_called()


def test_run_worker_once_handles_empty_queue():
    from unittest.mock import MagicMock

    from app.workers.ingestion_worker import (
        run_worker_once,
    )

    sqs = MagicMock()

    sqs.receive_message.return_value = {}

    handled = run_worker_once(
        sqs_client=sqs,
        queue_url="queue-url",
    )

    assert handled == 0
    sqs.delete_message.assert_not_called()


@patch(
    "app.workers.ingestion_worker."
    "process_ingestion_message"
)
def test_run_worker_once_rejects_invalid_json(
    mock_process,
):
    from unittest.mock import MagicMock

    from app.workers.ingestion_worker import (
        run_worker_once,
    )

    sqs = MagicMock()

    sqs.receive_message.return_value = {
        "Messages": [
            {
                "Body": "not-json",
                "ReceiptHandle": "receipt-123",
            }
        ]
    }

    handled = run_worker_once(
        sqs_client=sqs,
        queue_url="queue-url",
    )

    assert handled == 0
    mock_process.assert_not_called()
    sqs.delete_message.assert_not_called()


@patch(
    "app.workers.ingestion_worker."
    "process_ingestion_message"
)
def test_run_worker_once_extends_visibility_before_processing(
    mock_process,
):
    from unittest.mock import MagicMock

    from app.workers.ingestion_worker import (
        run_worker_once,
    )

    sqs = MagicMock()

    sqs.receive_message.return_value = {
        "Messages": [
            {
                "Body": (
                    '{"version":1,'
                    '"document_id":"doc-123"}'
                ),
                "ReceiptHandle": "receipt-123",
            }
        ]
    }

    mock_process.return_value = {
        "document_id": "doc-123",
        "status": "completed",
    }

    handled = run_worker_once(
        sqs_client=sqs,
        queue_url="queue-url",
    )

    assert handled == 1

    sqs.change_message_visibility.assert_called_once_with(
        QueueUrl="queue-url",
        ReceiptHandle="receipt-123",
        VisibilityTimeout=900,
    )

    visibility_call = (
        sqs.change_message_visibility.call_args
    )
    delete_call = sqs.delete_message.call_args

    assert visibility_call is not None
    assert delete_call is not None


@patch(
    "app.workers.ingestion_worker."
    "process_ingestion_message"
)
def test_run_worker_once_does_not_delete_already_processing(
    mock_process,
):
    from unittest.mock import MagicMock

    from app.workers.ingestion_worker import (
        run_worker_once,
    )

    sqs = MagicMock()

    sqs.receive_message.return_value = {
        "Messages": [
            {
                "Body": (
                    '{"version":1,'
                    '"document_id":"doc-123"}'
                ),
                "ReceiptHandle": "receipt-123",
            }
        ]
    }

    mock_process.return_value = {
        "document_id": "doc-123",
        "status": "already_processing",
        "processed": False,
    }

    handled = run_worker_once(
        sqs_client=sqs,
        queue_url="queue-url",
    )

    assert handled == 0

    mock_process.assert_called_once_with(
        {
            "version": 1,
            "document_id": "doc-123",
        }
    )

    sqs.change_message_visibility.assert_called_once_with(
        QueueUrl="queue-url",
        ReceiptHandle="receipt-123",
        VisibilityTimeout=900,
    )

    sqs.delete_message.assert_not_called()


@patch(
    "app.workers.dispatch_reconciler."
    "reconcile_orphaned_dispatches"
)
@patch(
    "app.workers.ingestion_worker.run_worker_once"
)
def test_run_worker_reconciles_on_start(
    mock_run_once,
    mock_reconcile,
):
    from app.workers.ingestion_worker import run_worker

    mock_run_once.side_effect = KeyboardInterrupt()

    run_worker(
        reconciliation_interval_seconds=300
    )

    mock_reconcile.assert_not_called()


@patch(
    "app.workers.dispatch_reconciler."
    "reconcile_orphaned_dispatches"
)
@patch(
    "app.workers.ingestion_worker.run_worker_once"
)
def test_run_worker_reconciles_after_first_poll(
    mock_run_once,
    mock_reconcile,
):
    from app.workers.ingestion_worker import run_worker

    mock_run_once.side_effect = [
        0,
        KeyboardInterrupt(),
    ]

    run_worker(
        reconciliation_interval_seconds=300
    )

    mock_reconcile.assert_called_once_with(
        max_documents=10
    )


@patch(
    "app.workers.dispatch_reconciler."
    "reconcile_orphaned_dispatches"
)
@patch(
    "app.workers.ingestion_worker.run_worker_once"
)
def test_reconciliation_failure_does_not_stop_worker(
    mock_run_once,
    mock_reconcile,
):
    from app.workers.ingestion_worker import run_worker

    mock_run_once.side_effect = [
        0,
        KeyboardInterrupt(),
    ]

    mock_reconcile.side_effect = RuntimeError(
        "S3 unavailable"
    )

    run_worker(
        reconciliation_interval_seconds=300
    )

    assert mock_run_once.call_count == 2
    mock_reconcile.assert_called_once_with(
        max_documents=10
    )


def test_run_worker_requires_positive_reconciliation_interval():
    from app.workers.ingestion_worker import run_worker

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        run_worker(
            reconciliation_interval_seconds=0
        )


def make_staged_upload_message():
    return {
        "version": 1,
        "message_type": "staged_upload",
        "upload_id": "upload-123",
        "bucket": "equityai-documents",
        "object_key": "staging/upload-123/report.pdf",
        "document_name": "report.pdf",
        "metadata": {
            "company_name": "Example Plc",
            "ticker": "EXM",
            "exchange": "LSE",
            "market": "UK",
            "country": "UK",
            "currency": "GBP",
            "document_type": "financial_report",
            "report_type": "annual",
            "fiscal_year": 2025,
            "period_end": "2025-12-31",
        },
    }


@patch(
    "app.workers.ingestion_worker.dispatch_document_ingestion"
)
@patch(
    "app.workers.ingestion_worker.download_document_file"
)
def test_process_staged_upload_message(
    mock_download,
    mock_dispatch,
):
    from app.workers.ingestion_worker import (
        process_staged_upload_message,
    )

    mock_dispatch.return_value = {
        "document_id": "doc-123",
        "document_hash": "hash-123",
        "status": "queued",
        "storage": {
            "bucket": "equityai-documents",
            "object_key": "documents/doc-123/report.pdf",
        },
    }

    result = process_staged_upload_message(
        make_staged_upload_message()
    )

    mock_download.assert_called_once()
    mock_dispatch.assert_called_once()

    prepare_kwargs = mock_dispatch.call_args.kwargs

    assert prepare_kwargs["document_name"] == "report.pdf"
    assert prepare_kwargs["company_name"] == "Example Plc"
    assert prepare_kwargs["exchange"] == "LSE"
    assert prepare_kwargs["market"] == "UK"
    assert prepare_kwargs["fiscal_year"] == 2025

    assert result["status"] == "queued"


@patch(
    "app.workers.ingestion_worker.dispatch_document_ingestion"
)
@patch(
    "app.workers.ingestion_worker.download_document_file"
)
def test_staged_upload_failure_preserves_staging_object(
    mock_download,
    mock_dispatch,
):
    from app.workers.ingestion_worker import (
        process_staged_upload_message,
    )

    mock_dispatch.side_effect = RuntimeError(
        "Document dispatch failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Document dispatch failed",
    ):
        process_staged_upload_message(
            make_staged_upload_message()
        )

    mock_download.assert_called_once()
    mock_dispatch.assert_called_once()


def test_staged_upload_rejects_wrong_object_key():
    from app.workers.ingestion_worker import (
        process_staged_upload_message,
    )

    message = make_staged_upload_message()
    message["object_key"] = (
        "staging/different-upload/report.pdf"
    )

    with pytest.raises(
        ValueError,
        match="does not belong to upload_id",
    ):
        process_staged_upload_message(message)


@patch(
    "app.workers.ingestion_worker.process_staged_upload_message"
)
@patch(
    "app.workers.ingestion_worker.process_ingestion_message"
)
def test_run_worker_once_routes_staged_upload(
    mock_process_ingestion,
    mock_process_staged,
):
    from unittest.mock import MagicMock

    from app.workers.ingestion_worker import (
        run_worker_once,
    )

    mock_sqs = MagicMock()

    mock_sqs.receive_message.return_value = {
        "Messages": [
            {
                "Body": json.dumps(
                    make_staged_upload_message()
                ),
                "ReceiptHandle": "receipt-staged-123",
            }
        ]
    }

    mock_process_staged.return_value = {
        "document_id": "doc-123",
        "status": "queued",
    }

    handled = run_worker_once(
        sqs_client=mock_sqs,
        queue_url="queue-url",
    )

    mock_process_staged.assert_called_once_with(
        make_staged_upload_message()
    )

    mock_process_ingestion.assert_not_called()

    mock_sqs.delete_message.assert_called_once_with(
        QueueUrl="queue-url",
        ReceiptHandle="receipt-staged-123",
    )

    assert handled == 1
