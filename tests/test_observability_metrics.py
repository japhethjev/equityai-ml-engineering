import pytest

from app.observability import (
    get_rag_metrics,
    record_rag_metrics,
    reset_rag_metrics,
)


@pytest.fixture(autouse=True)
def clean_metrics():
    """
    Ensure every test starts and finishes with
    an empty aggregate metrics store.
    """

    reset_rag_metrics()

    yield

    reset_rag_metrics()


def test_empty_metrics():
    """
    A new metrics store should contain zero
    requests and no latency observations.
    """

    metrics = get_rag_metrics()

    assert metrics["requests_total"] == 0

    assert metrics["outcomes"] == {
        "answered": 0,
        "abstained": 0,
        "errors": 0,
    }

    assert metrics["rates"] == {
        "answered_pct": 0.0,
        "abstained_pct": 0.0,
        "error_pct": 0.0,
    }

    assert (
        metrics["latency_ms"]["total_ms"]["count"]
        == 0
    )

    assert (
        metrics["latency_ms"]["total_ms"]["p50"]
        is None
    )


def test_records_rag_outcomes():
    """
    ANSWERED, ABSTAINED and ERROR requests
    should be counted correctly.
    """

    record_rag_metrics(
        outcome="ANSWERED",
        timings={"total_ms": 100.0},
    )

    record_rag_metrics(
        outcome="ANSWERED",
        timings={"total_ms": 200.0},
    )

    record_rag_metrics(
        outcome="ABSTAINED",
        timings={"total_ms": 300.0},
    )

    record_rag_metrics(
        outcome="ERROR",
        timings={"total_ms": 400.0},
    )

    metrics = get_rag_metrics()

    assert metrics["requests_total"] == 4

    assert metrics["outcomes"] == {
        "answered": 2,
        "abstained": 1,
        "errors": 1,
    }

    assert metrics["rates"] == {
        "answered_pct": 50.0,
        "abstained_pct": 25.0,
        "error_pct": 25.0,
    }


def test_latency_percentiles():
    """
    Aggregate metrics should calculate
    latency percentiles correctly.
    """

    values = [
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
    ]

    for value in values:
        record_rag_metrics(
            outcome="ANSWERED",
            timings={
                "embedding_ms": value,
                "retrieval_ms": value,
                "reranking_ms": value,
                "generation_ms": value,
                "total_ms": value,
            },
        )

    metrics = get_rag_metrics()

    total = metrics["latency_ms"]["total_ms"]

    assert total["count"] == 5
    assert total["p50"] == 300.0
    assert total["p95"] == 480.0
    assert total["p99"] == 496.0

    embedding = (
        metrics["latency_ms"]["embedding_ms"]
    )

    assert embedding["count"] == 5
    assert embedding["p50"] == 300.0


def test_missing_stage_timings_are_safe():
    """
    Failed requests may not reach every RAG
    stage. Missing timing values should not
    create invalid observations.
    """

    record_rag_metrics(
        outcome="ERROR",
        timings={
            "embedding_ms": 120.0,
            "total_ms": 125.0,
        },
    )

    metrics = get_rag_metrics()

    assert metrics["requests_total"] == 1

    assert (
        metrics["latency_ms"]["embedding_ms"]["count"]
        == 1
    )

    assert (
        metrics["latency_ms"]["retrieval_ms"]["count"]
        == 0
    )

    assert (
        metrics["latency_ms"]["generation_ms"]["count"]
        == 0
    )

    assert (
        metrics["latency_ms"]["total_ms"]["count"]
        == 1
    )


def test_reset_rag_metrics():
    """
    Resetting metrics should remove all previous
    observations.
    """

    record_rag_metrics(
        outcome="ANSWERED",
        timings={
            "embedding_ms": 100.0,
            "total_ms": 500.0,
        },
    )

    reset_rag_metrics()

    metrics = get_rag_metrics()

    assert metrics["requests_total"] == 0

    assert metrics["outcomes"] == {
        "answered": 0,
        "abstained": 0,
        "errors": 0,
    }

    assert (
        metrics["latency_ms"]["total_ms"]["count"]
        == 0
    )


def test_invalid_outcome_rejected():
    """
    Unknown outcomes must not enter the
    production metrics store.
    """

    with pytest.raises(
        ValueError,
        match="Unsupported RAG outcome",
    ):
        record_rag_metrics(
            outcome="UNKNOWN",
            timings={"total_ms": 100.0},
        )