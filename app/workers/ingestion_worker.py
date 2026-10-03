"""Worker logic for asynchronous EquityAI document ingestion."""

import tempfile
from pathlib import Path

from app.infrastructure.document_storage import (
    download_document_file,
)
from app.rag.ingest import ingest_document
from app.rag.vector_store import (
    claim_document_for_ingestion,
)


REQUIRED_MESSAGE_FIELDS = (
    "document_id",
    "document_hash",
    "bucket",
    "object_key",
    "document_name",
)


def validate_ingestion_message(message: dict) -> None:
    """Validate the minimum contract required by the worker."""

    if not isinstance(message, dict):
        raise ValueError(
            "Ingestion message must be a dictionary."
        )

    if message.get("version") != 1:
        raise ValueError(
            "Unsupported ingestion message version."
        )

    for field in REQUIRED_MESSAGE_FIELDS:
        value = message.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field} is required"
            )

    metadata = message.get("metadata", {})

    if not isinstance(metadata, dict):
        raise ValueError(
            "metadata must be a dictionary"
        )


def process_ingestion_message(message: dict) -> dict:
    """
    Process one already-registered ingestion job.

    SQS acknowledgement is deliberately outside this function.
    The polling loop must delete a message only after this function
    returns successfully.
    """

    validate_ingestion_message(message)

    document_id = message["document_id"].strip()
    document_hash = message["document_hash"].strip()
    bucket = message["bucket"].strip()
    object_key = message["object_key"].strip()
    document_name = Path(
        message["document_name"]
    ).name

    metadata = message.get("metadata") or {}

    registry = claim_document_for_ingestion(
        document_id
    )

    # A redelivered message for a completed document is safe to
    # acknowledge without downloading or re-embedding the PDF.
    if registry["status"] == "completed":
        return {
            "document_id": document_id,
            "status": "already_completed",
            "processed": False,
        }

    # Another worker already owns this document. Do not perform
    # duplicate OpenAI embedding or pgvector writes.
    if not registry.get("claimed"):
        return {
            "document_id": document_id,
            "status": "already_processing",
            "processed": False,
        }

    registry = {
        **registry,
        "document_hash": document_hash,
    }

    with tempfile.TemporaryDirectory(
        prefix="equityai-ingestion-"
    ) as temp_directory:

        local_path = (
            Path(temp_directory)
            / document_name
        )

        download_document_file(
            bucket=bucket,
            object_key=object_key,
            destination_path=str(local_path),
        )

        return ingest_document(
            str(local_path),
            document_name=document_name,
            company_name=metadata.get(
                "company_name"
            ),
            ticker=metadata.get("ticker"),
            market=metadata.get("market"),
            exchange=metadata.get("exchange"),
            country=metadata.get("country"),
            currency=metadata.get("currency"),
            document_type=metadata.get(
                "document_type"
            ),
            report_type=metadata.get(
                "report_type"
            ),
            reporting_period=metadata.get(
                "reporting_period"
            ),
            fiscal_year=metadata.get(
                "fiscal_year"
            ),
            fiscal_quarter=metadata.get(
                "fiscal_quarter"
            ),
            fiscal_half=metadata.get(
                "fiscal_half"
            ),
            period_start=metadata.get(
                "period_start"
            ),
            period_end=metadata.get(
                "period_end"
            ),
            publication_date=metadata.get(
                "publication_date"
            ),
            registry=registry,
        )


def run_worker_once(
    *,
    sqs_client=None,
    queue_url: str | None = None,
) -> int:
    """
    Poll SQS once and process available ingestion messages.

    Successfully handled messages are deleted. Failed messages are
    deliberately left in SQS so visibility timeout / DLQ policy can
    control retries.

    Returns the number of messages successfully handled.
    """

    import json

    import boto3

    from app.infrastructure.ingestion_queue import (
        get_ingestion_queue_url,
    )

    if sqs_client is None:
        sqs_client = boto3.client("sqs")

    if queue_url is None:
        queue_url = get_ingestion_queue_url()

    response = sqs_client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=20,
        AttributeNames=[
            "ApproximateReceiveCount",
        ],
    )

    messages = response.get(
        "Messages",
        [],
    )

    handled = 0

    for sqs_message in messages:
        receipt_handle = sqs_message.get(
            "ReceiptHandle"
        )

        try:
            body = json.loads(
                sqs_message["Body"]
            )

            if receipt_handle:
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=900,
                )

            result = process_ingestion_message(
                body
            )

            # Another worker still owns an active processing
            # lease. Leave this SQS message unacknowledged so it
            # can be retried after the visibility timeout. If the
            # owning worker crashes, a later delivery can reclaim
            # the document after its database lease expires.
            if (
                result.get("status")
                == "already_processing"
            ):
                continue

        except Exception:
            # Do not acknowledge failed work. SQS will make the
            # message visible again after the visibility timeout.
            continue

        if not receipt_handle:
            # A malformed SQS envelope must never be treated as
            # acknowledged work.
            continue

        sqs_client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
        )

        handled += 1

    return handled


def run_worker(
    *,
    reconciliation_interval_seconds: float = 300.0,
) -> None:
    """
    Continuously poll SQS and periodically recover orphaned dispatches.
    """

    import logging
    import time

    from app.workers.dispatch_reconciler import (
        reconcile_orphaned_dispatches,
    )

    if reconciliation_interval_seconds <= 0:
        raise ValueError(
            "reconciliation_interval_seconds must be positive"
        )

    logger = logging.getLogger(
        "equityai-ingestion-worker"
    )

    logger.info(
        "EquityAI ingestion worker started"
    )

    next_reconciliation = time.monotonic()

    while True:
        try:
            run_worker_once()

            now = time.monotonic()

            if now >= next_reconciliation:
                try:
                    reconciled = (
                        reconcile_orphaned_dispatches(
                            max_documents=10
                        )
                    )

                    if reconciled:
                        logger.info(
                            "Recovered %s orphaned "
                            "document dispatch(es)",
                            reconciled,
                        )

                except Exception:
                    logger.exception(
                        "Document dispatch reconciliation failed"
                    )

                finally:
                    next_reconciliation = (
                        time.monotonic()
                        + reconciliation_interval_seconds
                    )

        except KeyboardInterrupt:
            logger.info(
                "EquityAI ingestion worker stopped"
            )
            return

        except Exception:
            logger.exception(
                "Unexpected ingestion worker error"
            )
            time.sleep(5)


if __name__ == "__main__":
    run_worker()
