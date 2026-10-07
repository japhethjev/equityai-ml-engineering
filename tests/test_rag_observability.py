from unittest.mock import patch

import pytest

from app.rag.rag_service import (
    RERANK_CANDIDATE_LIMIT,
    answer_question,
)


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
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_embedding_error_is_observed(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_log_rag_error,
):
    """
    An embedding failure should be recorded as an
    embedding-stage RAG error and re-raised.
    """

    mock_resolve_retrieval_scope.return_value = {
        "company": None,
        "selections": [],
        "documents": [],
        "document_ids": None,
        "filtered": False,
    }


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
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_retrieval_error_is_observed(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_hybrid_search,
    mock_log_rag_error,
):
    """
    A retrieval failure should be recorded as a
    retrieval-stage RAG error and re-raised.
    """

    mock_resolve_retrieval_scope.return_value = {
        "company": None,
        "selections": [],
        "documents": [],
        "document_ids": None,
        "filtered": False,
    }


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
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_reranking_error_is_observed(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_hybrid_search,
    mock_rerank_results,
    mock_log_rag_error,
):
    """
    A reranking failure should be recorded as a
    reranking-stage RAG error and re-raised.
    """

    mock_resolve_retrieval_scope.return_value = {
        "company": None,
        "selections": [],
        "documents": [],
        "document_ids": None,
        "filtered": False,
    }


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
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_generation_error_is_observed(
    mock_resolve_retrieval_scope,
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

    mock_resolve_retrieval_scope.return_value = {
        "company": None,
        "selections": [],
        "documents": [],
        "document_ids": None,
        "filtered": False,
    }


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

@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_answer_question_passes_scoped_document_ids_to_search(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_hybrid_search,
    mock_log_rag_error,
):
    """
    A report-aware question must constrain hybrid retrieval
    to the document IDs selected by retrieval scope.
    """

    document_id = (
        "11111111-1111-1111-1111-111111111111"
    )

    mock_resolve_retrieval_scope.return_value = {
        "company": {
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "exchange": "NASDAQ",
        },
        "selections": [],
        "documents": [],
        "document_ids": [document_id],
        "filtered": True,
    }

    mock_embed_text.return_value = [
        0.1,
        0.2,
    ]

    error = RuntimeError(
        "Stop after retrieval call"
    )

    mock_hybrid_search.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Stop after retrieval call",
    ):
        answer_question(
            "What was AAPL revenue in Q2 2026?",
            request_id="scoped-test",
        )

    mock_hybrid_search.assert_called_once_with(
        query="What was AAPL revenue in Q2 2026?",
        query_embedding=[0.1, 0.2],
        limit=RERANK_CANDIDATE_LIMIT,
        document_ids=[document_id],
    )


@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_answer_question_preserves_unrestricted_search(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_hybrid_search,
    mock_log_rag_error,
):
    """
    Legacy questions without issuer/report scope continue
    to use unrestricted hybrid retrieval.
    """

    mock_resolve_retrieval_scope.return_value = {
        "company": None,
        "selections": [],
        "documents": [],
        "document_ids": None,
        "filtered": False,
    }

    mock_embed_text.return_value = [
        0.1,
        0.2,
    ]

    error = RuntimeError(
        "Stop after retrieval call"
    )

    mock_hybrid_search.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Stop after retrieval call",
    ):
        answer_question(
            "What is the IPO offer price?",
            request_id="unrestricted-test",
        )

    mock_hybrid_search.assert_called_once_with(
        query="What is the IPO offer price?",
        query_embedding=[0.1, 0.2],
        limit=RERANK_CANDIDATE_LIMIT,
        document_ids=None,
    )


@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_comparison_passes_all_document_ids_to_search(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_hybrid_search,
    mock_log_rag_error,
):
    """
    A multi-period comparison must preserve every resolved
    document ID when hybrid retrieval is executed.
    """

    q2_2025_id = (
        "11111111-1111-1111-1111-111111111111"
    )

    q2_2026_id = (
        "22222222-2222-2222-2222-222222222222"
    )

    mock_resolve_retrieval_scope.return_value = {
        "company": {
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "exchange": "NASDAQ",
        },
        "selections": [],
        "documents": [],
        "document_ids": [
            q2_2025_id,
            q2_2026_id,
        ],
        "filtered": True,
    }

    mock_embed_text.return_value = [
        0.1,
        0.2,
    ]

    error = RuntimeError(
        "Stop after comparison retrieval"
    )

    mock_hybrid_search.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Stop after comparison retrieval",
    ):
        answer_question(
            (
                "Compare AAPL revenue in "
                "Q2 2025 and Q2 2026."
            ),
            request_id="comparison-test",
        )

    mock_hybrid_search.assert_called_once_with(
        query=(
            "Compare AAPL revenue in "
            "Q2 2025 and Q2 2026."
        ),
        query_embedding=[0.1, 0.2],
        limit=RERANK_CANDIDATE_LIMIT,
        document_ids=[
            q2_2025_id,
            q2_2026_id,
        ],
    )


@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.select_balanced_evidence")
@patch("app.rag.rag_service.rerank_results")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_comparison_uses_balanced_evidence_selection(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_hybrid_search,
    mock_rerank_results,
    mock_select_balanced_evidence,
    mock_log_rag_error,
):
    """
    The production RAG pipeline must pass comparison
    document IDs into balanced evidence selection.
    """

    document_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]

    mock_resolve_retrieval_scope.return_value = {
        "company": {
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "exchange": "NASDAQ",
        },
        "selections": [],
        "documents": [],
        "document_ids": document_ids,
        "filtered": True,
    }

    mock_embed_text.return_value = [0.1, 0.2]

    retrieved = [
        {
            "document": "q2-2025.pdf",
            "document_id": document_ids[0],
            "page": 1,
            "chunk": 1,
            "content": "2025 revenue",
            "semantic_similarity": 0.8,
        },
        {
            "document": "q2-2026.pdf",
            "document_id": document_ids[1],
            "page": 1,
            "chunk": 1,
            "content": "2026 revenue",
            "semantic_similarity": 0.9,
        },
    ]

    mock_hybrid_search.return_value = retrieved
    mock_rerank_results.return_value = retrieved

    # Stop the pipeline cleanly at evidence selection.
    error = RuntimeError(
        "Stop after balanced selection"
    )
    mock_select_balanced_evidence.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Stop after balanced selection",
    ):
        answer_question(
            "Compare AAPL revenue in Q2 2025 and Q2 2026.",
            request_id="balanced-comparison-test",
        )

    mock_select_balanced_evidence.assert_called_once_with(
        retrieved,
        document_ids=document_ids,
        limit=3,
    )


@patch("app.rag.rag_service.log_rag_error")
@patch("app.rag.rag_service.select_balanced_evidence")
@patch("app.rag.rag_service.rerank_results")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_multi_company_evidence_limit_scales_with_documents(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_hybrid_search,
    mock_rerank_results,
    mock_select_balanced_evidence,
    mock_log_rag_error,
):
    document_ids = [
        f"00000000-0000-0000-0000-{number:012d}"
        for number in range(1, 6)
    ]

    mock_resolve_retrieval_scope.return_value = {
        "company": None,
        "companies": [
            {"company_name": f"Company {number}"}
            for number in range(1, 6)
        ],
        "selections": [],
        "documents": [],
        "document_ids": document_ids,
        "filtered": True,
    }

    mock_embed_text.return_value = [0.1, 0.2]

    retrieved = [
        {
            "document": f"company-{number}.pdf",
            "document_id": document_id,
            "page": 1,
            "chunk": 1,
            "content": f"Company {number} evidence",
            "semantic_similarity": 0.9,
        }
        for number, document_id in enumerate(
            document_ids,
            start=1,
        )
    ]

    mock_hybrid_search.return_value = retrieved
    mock_rerank_results.return_value = retrieved

    error = RuntimeError("Stop after dynamic selection")
    mock_select_balanced_evidence.side_effect = error

    with pytest.raises(
        RuntimeError,
        match="Stop after dynamic selection",
    ):
        answer_question(
            "Compare Company 1, Company 2, Company 3, "
            "Company 4 and Company 5.",
            request_id="multi-company-limit-test",
        )

    mock_select_balanced_evidence.assert_called_once_with(
        retrieved,
        document_ids=document_ids,
        limit=5,
    )


@patch("app.rag.rag_service.record_rag_metrics")
@patch("app.rag.rag_service.log_rag_request")
@patch("app.rag.rag_service.get_openai_client")
@patch("app.rag.rag_service.rerank_results")
@patch("app.rag.rag_service.hybrid_search")
@patch("app.rag.rag_service.embed_text")
@patch("app.rag.rag_service.resolve_retrieval_scope")
def test_multi_company_prompt_enforces_comparison_contract(
    mock_resolve_retrieval_scope,
    mock_embed_text,
    mock_hybrid_search,
    mock_rerank_results,
    mock_get_openai_client,
    mock_log_rag_request,
    mock_record_rag_metrics,
):
    document_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
        "44444444-4444-4444-4444-444444444444",
    ]

    companies = [
        "Barclays Bank Plc",
        "HSBC Holdings plc",
        "Deutsche Bank",
        "Standard Chartered Bank",
    ]

    mock_resolve_retrieval_scope.return_value = {
        "company": None,
        "companies": [
            {"company_name": company}
            for company in companies
        ],
        "selections": [],
        "documents": [],
        "document_ids": document_ids,
        "filtered": True,
    }

    mock_embed_text.return_value = [0.1, 0.2]

    retrieved = [
        {
            "document": f"company-{number}.pdf",
            "document_id": document_id,
            "page": number,
            "chunk": 1,
            "content": (
                f"{company} reported profit before tax."
            ),
            "semantic_similarity": 0.9,
            "final_score": 0.9,
        }
        for number, (document_id, company) in enumerate(
            zip(document_ids, companies),
            start=1,
        )
    ]

    mock_hybrid_search.return_value = retrieved
    mock_rerank_results.return_value = retrieved

    client = mock_get_openai_client.return_value
    client.responses.create.return_value.output_text = (
        "Comparison answer"
    )

    answer_question(
        "Compare Barclays Bank Plc, HSBC Holdings plc, "
        "Deutsche Bank and Standard Chartered Bank using "
        "their 2025 annual reports.",
        request_id="multi-company-prompt-test",
    )

    prompt = client.responses.create.call_args.kwargs["input"]

    assert (
        "MULTI-COMPANY COMPARISON CONTROL:"
        in prompt
    )
    assert "ENABLED" in prompt
    assert (
        "Never use one company's evidence to answer "
        "for another company."
        in prompt
    )
    assert (
        "Markdown comparison table"
        in prompt
    )
    assert (
        "Source Document, and Page"
        in prompt
    )
    assert (
        "brief evidence-grounded comparative analysis"
        in prompt
    )
