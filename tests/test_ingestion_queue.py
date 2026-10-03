import json
from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.ingestion_queue import (
    build_ingestion_message,
    enqueue_ingestion_job,
    get_ingestion_queue_url,
)


def test_get_ingestion_queue_url(monkeypatch):
    monkeypatch.setenv(
        "INGESTION_QUEUE_URL",
        "https://sqs.example/queue",
    )

    assert (
        get_ingestion_queue_url()
        == "https://sqs.example/queue"
    )


def test_get_ingestion_queue_url_requires_configuration(
    monkeypatch,
):
    monkeypatch.delenv(
        "INGESTION_QUEUE_URL",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="INGESTION_QUEUE_URL",
    ):
        get_ingestion_queue_url()


def test_build_ingestion_message():
    message = build_ingestion_message(
        document_id="doc-123",
        document_hash="hash-123",
        bucket="equityai-documents",
        object_key="documents/doc-123/report.pdf",
        document_name="report.pdf",
        metadata={
            "company_name": "Example Plc",
            "fiscal_year": 2025,
        },
    )

    assert message == {
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
            "fiscal_year": 2025,
        },
    }


def test_build_ingestion_message_requires_document_id():
    with pytest.raises(
        ValueError,
        match="document_id",
    ):
        build_ingestion_message(
            document_id="",
            document_hash="hash",
            bucket="bucket",
            object_key="key",
            document_name="report.pdf",
        )


@patch(
    "app.infrastructure.ingestion_queue.boto3.client"
)
def test_enqueue_ingestion_job(
    mock_boto_client,
    monkeypatch,
):
    monkeypatch.setenv(
        "INGESTION_QUEUE_URL",
        "https://sqs.example/queue",
    )

    mock_sqs = MagicMock()
    mock_sqs.send_message.return_value = {
        "MessageId": "message-123",
    }

    mock_boto_client.return_value = mock_sqs

    result = enqueue_ingestion_job(
        document_id="doc-123",
        document_hash="hash-123",
        bucket="equityai-documents",
        object_key="documents/doc-123/report.pdf",
        document_name="report.pdf",
        metadata={
            "company_name": "Example Plc",
        },
    )

    mock_boto_client.assert_called_once_with("sqs")

    mock_sqs.send_message.assert_called_once()

    kwargs = mock_sqs.send_message.call_args.kwargs

    assert (
        kwargs["QueueUrl"]
        == "https://sqs.example/queue"
    )

    body = json.loads(kwargs["MessageBody"])

    assert body["version"] == 1
    assert body["document_id"] == "doc-123"
    assert body["document_hash"] == "hash-123"
    assert body["bucket"] == "equityai-documents"
    assert body["document_name"] == "report.pdf"

    assert result["message_id"] == "message-123"
