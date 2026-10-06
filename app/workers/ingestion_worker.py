"""Worker logic for asynchronous EquityAI document ingestion."""

import logging
import tempfile
import threading
from pathlib import Path

from app.infrastructure.document_storage import (
    download_document_file,
)
from app.rag.ingest import ingest_document
from app.rag.ingestion_dispatch import (
    dispatch_document_ingestion,
)
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

SQS_VISIBILITY_TIMEOUT_SECONDS = 900
SQS_VISIBILITY_HEARTBEAT_SECONDS = 300

logger = logging.getLogger("equityai-ingestion-worker")


class SQSVisibilityHeartbeat:
    """Keep one in-flight SQS message invisible during long processing."""

    def __init__(
        self,
        *,
        sqs_client,
        queue_url: str,
        receipt_handle: str,
        visibility_timeout_seconds: int = (
            SQS_VISIBILITY_TIMEOUT_SECONDS
        ),
        heartbeat_seconds: float = (
            SQS_VISIBILITY_HEARTBEAT_SECONDS
        ),
    ) -> None:
        if visibility_timeout_seconds <= 0:
            raise ValueError(
                "visibility_timeout_seconds must be positive"
            )

        if heartbeat_seconds <= 0:
            raise ValueError(
                "heartbeat_seconds must be positive"
            )

        if heartbeat_seconds >= visibility_timeout_seconds:
            raise ValueError(
                "heartbeat_seconds must be less than "
                "visibility_timeout_seconds"
            )

        self.sqs_client = sqs_client
        self.queue_url = queue_url
        self.receipt_handle = receipt_handle
        self.visibility_timeout_seconds = (
            visibility_timeout_seconds
        )
        self.heartbeat_seconds = heartbeat_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sqs-visibility-heartbeat",
            daemon=True,
        )

    def _extend_visibility(self) -> None:
        self.sqs_client.change_message_visibility(
            QueueUrl=self.queue_url,
            ReceiptHandle=self.receipt_handle,
            VisibilityTimeout=self.visibility_timeout_seconds,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(
            self.heartbeat_seconds
        ):
            try:
                self._extend_visibility()
            except Exception:
                logger.exception(
                    "SQS visibility heartbeat renewal failed"
                )

    def start(self) -> None:
        # Protect the message immediately before processing starts.
        self._extend_visibility()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join()


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



def process_staged_upload_message(message: dict) -> dict:
    """
    Prepare one PDF uploaded directly to S3 by an authenticated admin.

    The worker downloads the staged object, calculates its trusted hash
    through the existing preparation pipeline, registers/deduplicates it,
    persists any new document to final S3 storage, and leaves the
    staging object available for safe SQS redelivery until lifecycle expiry.
    """
    if not isinstance(message, dict):
        raise ValueError(
            "Staged upload message must be a dictionary."
        )

    if message.get("version") != 1:
        raise ValueError(
            "Unsupported staged upload message version."
        )

    if message.get("message_type") != "staged_upload":
        raise ValueError(
            "Unsupported staged upload message type."
        )

    required_fields = (
        "upload_id",
        "bucket",
        "object_key",
        "document_name",
    )

    for field in required_fields:
        value = message.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")

    upload_id = message["upload_id"].strip()
    bucket = message["bucket"].strip()
    object_key = message["object_key"].strip()
    document_name = Path(
        message["document_name"]
    ).name

    if not object_key.startswith(
        f"staging/{upload_id}/"
    ):
        raise ValueError(
            "object_key does not belong to upload_id"
        )

    metadata = message.get("metadata") or {}

    if not isinstance(metadata, dict):
        raise ValueError(
            "metadata must be a dictionary"
        )

    with tempfile.TemporaryDirectory(
        prefix="equityai-staged-upload-"
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

        dispatched = dispatch_document_ingestion(
            str(local_path),
            document_name=document_name,
            company_name=metadata.get("company_name"),
            ticker=metadata.get("ticker"),
            market=metadata.get("market"),
            exchange=metadata.get("exchange"),
            country=metadata.get("country"),
            currency=metadata.get("currency"),
            document_type=metadata.get("document_type"),
            report_type=metadata.get("report_type"),
            reporting_period=metadata.get(
                "reporting_period"
            ),
            fiscal_year=metadata.get("fiscal_year"),
            fiscal_quarter=metadata.get(
                "fiscal_quarter"
            ),
            fiscal_half=metadata.get("fiscal_half"),
            period_start=metadata.get("period_start"),
            period_end=metadata.get("period_end"),
            publication_date=metadata.get(
                "publication_date"
            ),
        )

        return dispatched

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

            heartbeat = None

            if receipt_handle:
                heartbeat = SQSVisibilityHeartbeat(
                    sqs_client=sqs_client,
                    queue_url=queue_url,
                    receipt_handle=receipt_handle,
                )
                heartbeat.start()

            try:
                if body.get("message_type") == "staged_upload":
                    result = process_staged_upload_message(
                        body
                    )
                else:
                    result = process_ingestion_message(
                        body
                    )
            finally:
                if heartbeat is not None:
                    heartbeat.stop()

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
