"""Recovery logic for orphaned document dispatches."""

from app.infrastructure.document_storage import (
    build_document_key,
    document_object_exists,
    get_document_bucket,
)
from app.infrastructure.ingestion_queue import (
    enqueue_ingestion_job,
)
from app.rag.vector_store import (
    claim_next_orphaned_dispatch,
    fail_document,
)


def reconcile_one_orphaned_dispatch() -> dict:
    """
    Recover at most one expired queued document.

    A document is redispatched only when its durable S3 object still
    exists. Missing objects are marked failed rather than publishing
    an SQS job that cannot succeed.
    """

    document = claim_next_orphaned_dispatch()

    if document is None:
        return {
            "status": "idle",
            "reconciled": False,
        }

    document_id = document["document_id"]
    document_name = document["document_name"]
    document_hash = document["document_hash"]

    bucket = get_document_bucket()

    object_key = build_document_key(
        document_id=document_id,
        filename=document_name,
    )

    try:
        exists = document_object_exists(
            bucket=bucket,
            object_key=object_key,
        )

    except Exception:
        # Infrastructure failures are deliberately not converted into
        # document failures. The renewed dispatch lease will expire
        # and permit a later reconciliation attempt.
        raise

    if not exists:
        fail_document(
            document_id=document_id,
            error_message=(
                "Orphaned dispatch could not be recovered: "
                "document object is missing from S3."
            ),
        )

        return {
            "document_id": document_id,
            "status": "failed",
            "reconciled": True,
        }

    try:
        queue_result = enqueue_ingestion_job(
            document_id=document_id,
            document_hash=document_hash,
            bucket=bucket,
            object_key=object_key,
            document_name=document_name,
            metadata=document["metadata"],
        )

    except Exception:
        # Do not mark the document failed for a transient SQS outage.
        # The dispatch lease will expire and the reconciler can retry.
        raise

    return {
        "document_id": document_id,
        "status": "redispatched",
        "reconciled": True,
        "queue": {
            "message_id": queue_result["message_id"],
        },
    }


def reconcile_orphaned_dispatches(
    max_documents: int = 10,
) -> int:
    """
    Recover a bounded number of orphaned document dispatches.

    Stops early when no eligible queued documents remain.
    """

    if max_documents < 1:
        raise ValueError(
            "max_documents must be at least 1"
        )

    reconciled = 0

    for _ in range(max_documents):
        result = reconcile_one_orphaned_dispatch()

        if result["status"] == "idle":
            break

        if result.get("reconciled"):
            reconciled += 1

    return reconciled
