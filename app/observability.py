import logging
import threading
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


# --------------------------------------------------
# Aggregate RAG production metrics
# --------------------------------------------------

_metrics_lock = threading.Lock()

_rag_metrics = {
    "requests": 0,
    "outcomes": {
        "ANSWERED": 0,
        "ABSTAINED": 0,
        "ERROR": 0,
    },
    "latencies": {
        "embedding_ms": [],
        "retrieval_ms": [],
        "reranking_ms": [],
        "generation_ms": [],
        "total_ms": [],
    },
}


# --------------------------------------------------
# Request ID
# --------------------------------------------------


def create_request_id() -> str:
    """
    Create a short unique identifier that allows
    logs belonging to one request to be traced.
    """

    return uuid.uuid4().hex[:12]


# --------------------------------------------------
# Stage timing
# --------------------------------------------------


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


# --------------------------------------------------
# Percentile calculation
# --------------------------------------------------


def _percentile(
    values: list[float],
    percentile: float,
) -> float | None:
    """
    Calculate a percentile using linear
    interpolation.

    Returns None when no observations exist.
    """

    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return round(
            ordered[0],
            2,
        )

    position = (
        (len(ordered) - 1)
        * percentile
    )

    lower = int(position)

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = position - lower

    result = (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * weight
    )

    return round(
        result,
        2,
    )


# --------------------------------------------------
# Aggregate metrics recording
# --------------------------------------------------


def record_rag_metrics(
    *,
    outcome: str,
    timings: dict,
) -> None:
    """
    Record one completed RAG request in the
    aggregate production metrics.

    Only operational measurements are stored.

    User questions, retrieved document content,
    API credentials, and other sensitive data
    are deliberately excluded.
    """

    if outcome not in RAG_OUTCOMES:
        raise ValueError(
            f"Unsupported RAG outcome: {outcome}"
        )

    with _metrics_lock:

        _rag_metrics["requests"] += 1

        _rag_metrics[
            "outcomes"
        ][outcome] += 1

        for metric in _rag_metrics[
            "latencies"
        ]:

            value = timings.get(metric)

            if value is None:
                continue

            try:
                numeric_value = float(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

            _rag_metrics[
                "latencies"
            ][metric].append(
                numeric_value
            )


# --------------------------------------------------
# Aggregate metrics snapshot
# --------------------------------------------------


def get_rag_metrics() -> dict:
    """
    Return a thread-safe snapshot of aggregate
    RAG production metrics.

    The snapshot includes:

    - total request count;
    - outcome counts;
    - outcome rates;
    - P50 latency;
    - P95 latency;
    - P99 latency;
    - observation count for each latency metric.
    """

    with _metrics_lock:

        requests = _rag_metrics[
            "requests"
        ]

        outcomes = dict(
            _rag_metrics[
                "outcomes"
            ]
        )

        latencies = {
            name: list(values)
            for name, values
            in _rag_metrics[
                "latencies"
            ].items()
        }

    def rate(
        count: int,
    ) -> float:
        """
        Convert an outcome count into a
        percentage of total requests.
        """

        if requests == 0:
            return 0.0

        return round(
            (
                count
                / requests
            )
            * 100,
            2,
        )

    latency_summary = {}

    for (
        name,
        values,
    ) in latencies.items():

        latency_summary[name] = {
            "count": len(values),
            "p50": _percentile(
                values,
                0.50,
            ),
            "p95": _percentile(
                values,
                0.95,
            ),
            "p99": _percentile(
                values,
                0.99,
            ),
        }

    return {
        "requests_total": requests,
        "outcomes": {
            "answered": outcomes[
                "ANSWERED"
            ],
            "abstained": outcomes[
                "ABSTAINED"
            ],
            "errors": outcomes[
                "ERROR"
            ],
        },
        "rates": {
            "answered_pct": rate(
                outcomes[
                    "ANSWERED"
                ]
            ),
            "abstained_pct": rate(
                outcomes[
                    "ABSTAINED"
                ]
            ),
            "error_pct": rate(
                outcomes[
                    "ERROR"
                ]
            ),
        },
        "latency_ms": (
            latency_summary
        ),
    }


# --------------------------------------------------
# Metrics reset
# --------------------------------------------------


def reset_rag_metrics() -> None:
    """
    Reset all aggregate RAG metrics.

    This function is intended primarily for
    automated tests.

    Production code should normally never reset
    operational metrics manually.
    """

    with _metrics_lock:

        _rag_metrics[
            "requests"
        ] = 0

        for outcome in RAG_OUTCOMES:

            _rag_metrics[
                "outcomes"
            ][outcome] = 0

        for metric in _rag_metrics[
            "latencies"
        ]:

            _rag_metrics[
                "latencies"
            ][metric].clear()


# --------------------------------------------------
# Successful RAG request logging
# --------------------------------------------------


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

    Aggregate metrics are deliberately not
    recorded here yet.

    The RAG orchestration layer will explicitly
    record one metric observation per completed
    request. This prevents accidental
    double-counting when logging behaviour
    changes.
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
        timings.get(
            "embedding_ms"
        ),
        timings.get(
            "retrieval_ms"
        ),
        timings.get(
            "reranking_ms"
        ),
        timings.get(
            "generation_ms"
        ),
        timings.get(
            "total_ms"
        ),
    )


# --------------------------------------------------
# Failed RAG request logging
# --------------------------------------------------


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

    Aggregate metrics are deliberately not
    recorded here yet.

    The RAG orchestration layer will explicitly
    record the ERROR observation so each request
    is counted exactly once.
    """

    error_type = type(
        error
    ).__name__

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
        timings.get(
            "embedding_ms"
        ),
        timings.get(
            "retrieval_ms"
        ),
        timings.get(
            "reranking_ms"
        ),
        timings.get(
            "generation_ms"
        ),
        timings.get(
            "total_ms"
        ),
    )