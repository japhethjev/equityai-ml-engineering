import logging
import time
import uuid
from contextlib import contextmanager


# --------------------------------------------------
# EquityAI observability logger
# --------------------------------------------------

logger = logging.getLogger("equityai")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False


# --------------------------------------------------
# Supported RAG outcomes
# --------------------------------------------------

RAG_OUTCOMES = {
    "ANSWERED",
    "ABSTAINED",
    "ERROR",
}


def create_request_id() -> str:
    """
    Create a short unique identifier that allows
    logs belonging to one request to be traced.
    """

    return uuid.uuid4().hex[:12]


@contextmanager
def measure_stage(
    stage: str,
    timings: dict,
):
    """
    Measure the execution time of one RAG stage.

    Timing is recorded even when the stage raises
    an exception.
    """

    start = time.perf_counter()

    try:
        yield

    finally:
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        timings[f"{stage}_ms"] = round(
            elapsed_ms,
            2,
        )


def log_rag_request(
    *,
    request_id: str,
    outcome: str,
    retrieved_chunks: int,
    used_chunks: int,
    timings: dict,
) -> None:
    """
    Write one structured summary log for a
    successfully completed RAG request.

    Supported successful outcomes:
    - ANSWERED
    - ABSTAINED
    """

    if outcome not in RAG_OUTCOMES:
        raise ValueError(
            f"Unsupported RAG outcome: {outcome}"
        )

    logger.info(
        "event=rag_request "
        "request_id=%s "
        "outcome=%s "
        "retrieved_chunks=%s "
        "used_chunks=%s "
        "embedding_ms=%s "
        "retrieval_ms=%s "
        "reranking_ms=%s "
        "generation_ms=%s "
        "total_ms=%s",
        request_id,
        outcome,
        retrieved_chunks,
        used_chunks,
        timings.get("embedding_ms"),
        timings.get("retrieval_ms"),
        timings.get("reranking_ms"),
        timings.get("generation_ms"),
        timings.get("total_ms"),
    )


def log_rag_error(
    *,
    request_id: str,
    failed_stage: str,
    error: Exception,
    retrieved_chunks: int,
    used_chunks: int,
    timings: dict,
) -> None:
    """
    Write structured telemetry for a failed
    RAG request.

    Only operational metadata is logged.

    The user's question, retrieved document
    content, API credentials, and other sensitive
    data are deliberately excluded.
    """

    error_type = type(error).__name__

    logger.error(
        "event=rag_request "
        "request_id=%s "
        "outcome=ERROR "
        "failed_stage=%s "
        "error_type=%s "
        "retrieved_chunks=%s "
        "used_chunks=%s "
        "embedding_ms=%s "
        "retrieval_ms=%s "
        "reranking_ms=%s "
        "generation_ms=%s "
        "total_ms=%s",
        request_id,
        failed_stage,
        error_type,
        retrieved_chunks,
        used_chunks,
        timings.get("embedding_ms"),
        timings.get("retrieval_ms"),
        timings.get("reranking_ms"),
        timings.get("generation_ms"),
        timings.get("total_ms"),
    )