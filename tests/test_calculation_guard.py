import pytest

from app.rag.rag_service import (
    looks_like_unrequested_calculation,
)


@pytest.mark.parametrize(
    "answer",
    [
        "I calculated the profit margin as 25%.",
        "The computed growth rate is 12%.",
        "Calculation: 500 / 400 - 1 = 25%.",
        "Revenue ÷ total assets = 0.8.",
    ],
)
def test_detects_explicit_calculations(answer):
    assert looks_like_unrequested_calculation(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Revenue increased during the financial year.",
        "The figures were derived from the annual report.",
        "No calculation was performed.",
        "The calculated figures were not available.",
        "The company reported revenue of NGN 1.38 trillion.",
    ],
)
def test_allows_reporting_and_explanatory_language(answer):
    assert not looks_like_unrequested_calculation(answer)


@pytest.mark.parametrize(
    "generated_answer,should_block",
    [
        (
            "Dangote Cement reported revenue of NGN 1.38 trillion. "
            "The figures were derived from the annual report.",
            False,
        ),
        (
            "I calculated the profit margin as 25%.",
            True,
        ),
    ],
)
def test_calculation_guard_in_answer_pipeline(
    generated_answer,
    should_block,
):
    from unittest.mock import patch

    from app.rag.rag_service import answer_question

    document_id = "11111111-1111-1111-1111-111111111111"

    evidence = {
        "document": "dangote-2021.pdf",
        "document_id": document_id,
        "page": 16,
        "chunk": 1,
        "content": (
            "Dangote Cement Plc consolidated financial "
            "statements for FY2021."
        ),
        "semantic_similarity": 0.9,
        "final_score": 0.9,
    }

    with (
        patch(
            "app.rag.rag_service.resolve_retrieval_scope"
        ) as mock_scope,
        patch(
            "app.rag.rag_service.embed_text"
        ) as mock_embed,
        patch(
            "app.rag.rag_service.hybrid_search"
        ) as mock_search,
        patch(
            "app.rag.rag_service.financial_metric_search"
        ) as mock_metric_search,
        patch(
            "app.rag.rag_service.rerank_results"
        ) as mock_rank,
        patch(
            "app.rag.rag_service.get_openai_client"
        ) as mock_client,
        patch("app.rag.rag_service.log_rag_request"),
        patch("app.rag.rag_service.record_rag_metrics"),
    ):
        mock_scope.return_value = {
            "company": None,
            "companies": [
                {"company_name": "Dangote Cement Plc"}
            ],
            "selections": [],
            "documents": [
                {
                    "document_id": document_id,
                    "document_name": "dangote-2021.pdf",
                    "company_name": "Dangote Cement Plc",
                    "fiscal_year": 2021,
                }
            ],
            "document_ids": [document_id],
            "filtered": True,
        }

        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = [evidence]
        mock_metric_search.return_value = []
        mock_rank.return_value = [evidence]

        mock_client.return_value.responses.create.return_value.output_text = (
            generated_answer
        )

        result = answer_question(
            "Analyze Dangote Cement financial performance "
            "and major risks in FY2021.",
            request_id="calculation-guard-integration",
        )

    assert result["sources"]
    assert result["sources"][0]["page"] == 16

    if should_block:
        assert "not authorised or verified" in result["answer"]
    else:
        assert result["answer"] == generated_answer
