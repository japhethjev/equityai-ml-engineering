"""SQS queue integration for asynchronous document ingestion."""

import json
import os

import boto3


def get_ingestion_queue_url() -> str:
    """Return the configured ingestion SQS queue URL."""

    queue_url = os.getenv(
        "INGESTION_QUEUE_URL",
        "",
    ).strip()

    if not queue_url:
        raise RuntimeError(
            "INGESTION_QUEUE_URL environment variable "
            "is required."
        )

    return queue_url


def build_ingestion_message(
    *,
    document_id: str,
    document_hash: str,
    bucket: str,
    object_key: str,
    document_name: str,
    metadata: dict | None = None,
) -> dict:
    """Build the durable message consumed by the ingestion worker."""

    if not document_id.strip():
        raise ValueError("document_id is required")

    if not document_hash.strip():
        raise ValueError("document_hash is required")

    if not bucket.strip():
        raise ValueError("bucket is required")

    if not object_key.strip():
        raise ValueError("object_key is required")

    if not document_name.strip():
        raise ValueError("document_name is required")

    return {
        "version": 1,
        "document_id": document_id.strip(),
        "document_hash": document_hash.strip(),
        "bucket": bucket.strip(),
        "object_key": object_key.strip(),
        "document_name": document_name.strip(),
        "metadata": metadata or {},
    }


def enqueue_ingestion_job(
    *,
    document_id: str,
    document_hash: str,
    bucket: str,
    object_key: str,
    document_name: str,
    metadata: dict | None = None,
) -> dict:
    """Publish a document-ingestion job to SQS."""

    queue_url = get_ingestion_queue_url()

    message = build_ingestion_message(
        document_id=document_id,
        document_hash=document_hash,
        bucket=bucket,
        object_key=object_key,
        document_name=document_name,
        metadata=metadata,
    )

    sqs = boto3.client("sqs")

    response = sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(
            message,
            separators=(",", ":"),
        ),
    )

    return {
        "message_id": response["MessageId"],
        "queue_url": queue_url,
        "message": message,
    }
