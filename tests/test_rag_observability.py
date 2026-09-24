from unittest.mock import patch

import pytest

from app.rag.rag_service import answer_question


SAMPLE_RESULTS = [
    {
        "document": "test.pdf",
        "page": 1,
        "chunk": 1,
        "content": "The IPO offer price is ₦525.00.",
        "final_score": 0.95,
    }
]


@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.embed_text")
def test_embedding_error_is_observed(
    mock_embed_text,
    mock_log_rag_error,
):
    """
    An embedding failure should be recorded as an
    embedding-stage RAG error and re-raised.
    """

    error = RuntimeError("Embedding failed")

    mock_embed_text.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Embedding failed",
    ):
        answer_question(
            "What is the IPO offer price?",
            request_id="test-request",
        )

    kwargs = mock_log_rag_error.call_args.kwargs

    assert kwargs["request_id"] == "test-request"
    assert kwargs["failed_stage"] == "embedding"
    assert kwargs["error"] is error
    assert kwargs["retrieved_chunks"] == 0
    assert kwargs["used_chunks"] == 0
    assert "embedding_ms" in kwargs["timings"]
    assert "total_ms" in kwargs["timings"]


@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
def test_retrieval_error_is_observed(
    mock_embed_text,
    mock_hybrid_search,
    mock_log_rag_error,
):
    """
    A retrieval failure should be recorded as a
    retrieval-stage RAG error and re-raised.
    """

    mock_embed_text.return_value = [0.1, 0.2]

    error = RuntimeError("Retrieval failed")

    mock_hybrid_search.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Retrieval failed",
    ):
        answer_question(
            "What is the IPO offer price?",
            request_id="test-request",
        )

    kwargs = mock_log_rag_error.call_args.kwargs

    assert kwargs["request_id"] == "test-request"
    assert kwargs["failed_stage"] == "retrieval"
    assert kwargs["error"] is error
    assert kwargs["retrieved_chunks"] == 0
    assert kwargs["used_chunks"] == 0
    assert "embedding_ms" in kwargs["timings"]
    assert "retrieval_ms" in kwargs["timings"]
    assert "total_ms" in kwargs["timings"]


@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.rerank_results")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
def test_reranking_error_is_observed(
    mock_embed_text,
    mock_hybrid_search,
    mock_rerank_results,
    mock_log_rag_error,
):
    """
    A reranking failure should be recorded as a
    reranking-stage RAG error and re-raised.
    """

    mock_embed_text.return_value = [0.1, 0.2]

    mock_hybrid_search.return_value = SAMPLE_RESULTS

    error = RuntimeError("Reranking failed")

    mock_rerank_results.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Reranking failed",
    ):
        answer_question(
            "What is the IPO offer price?",
            request_id="test-request",
        )

    kwargs = mock_log_rag_error.call_args.kwargs

    assert kwargs["request_id"] == "test-request"
    assert kwargs["failed_stage"] == "reranking"
    assert kwargs["error"] is error
    assert kwargs["retrieved_chunks"] == 1
    assert kwargs["used_chunks"] == 0
    assert "reranking_ms" in kwargs["timings"]
    assert "total_ms" in kwargs["timings"]


@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.get_openai_client")
@patch("app.rag.rag_service.rerank_results")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
def test_generation_error_is_observed(
    mock_embed_text,
    mock_hybrid_search,
    mock_rerank_results,
    mock_get_openai_client,
    mock_log_rag_error,
):
    """
    A generation failure should be recorded as a
    generation-stage RAG error and re-raised.
    """

    mock_embed_text.return_value = [0.1, 0.2]

    mock_hybrid_search.return_value = SAMPLE_RESULTS

    mock_rerank_results.return_value = SAMPLE_RESULTS

    error = RuntimeError("Generation failed")

    client = mock_get_openai_client.return_value
    client.responses.create.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Generation failed",
    ):
        answer_question(
            "What is the IPO offer price?",
            request_id="test-request",
        )

    kwargs = mock_log_rag_error.call_args.kwargs

    assert kwargs["request_id"] == "test-request"
    assert kwargs["failed_stage"] == "generation"
    assert kwargs["error"] is error
    assert kwargs["retrieved_chunks"] == 1
    assert kwargs["used_chunks"] == 1

    assert "embedding_ms" in kwargs["timings"]
    assert "retrieval_ms" in kwargs["timings"]
    assert "reranking_ms" in kwargs["timings"]
    assert "generation_ms" in kwargs["timings"]
    assert "total_ms" in kwargs["timings"]