from unittest.mock import patch

import pytest

from app.rag.rag_service import (
    ABSTENTION_TEXT,
    answer_question,
)


@pytest.mark.parametrize(
    "question,expected_mode",
    [
        (
            "Analyse Dangote Cement financial performance "
            "and major risks in FY2021.",
            "broad",
        ),
        (
            "What was Dangote Cement revenue in FY2021?",
            "exact",
        ),
        (
            "Compare Barclays Bank Plc and HSBC Holdings plc "
            "profit before tax for FY2025.",
            "comparison",
        ),
    ],
)
def test_evidence_sufficiency_prompt(
    question,
    expected_mode,
):
    document_ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]

    companies = (
        ["Barclays Bank Plc", "HSBC Holdings plc"]
        if expected_mode == "comparison"
        else ["Dangote Cement Plc"]
    )

    selected_ids = (
        document_ids
        if expected_mode == "comparison"
        else document_ids[:1]
    )

    documents = [
        {
            "document_id": document_id,
            "document_name": f"report-{i}.pdf",
            "company_name": company,
            "fiscal_year": (
                2025 if expected_mode == "comparison" else 2021
            ),
        }
        for i, (document_id, company) in enumerate(
            zip(selected_ids, companies),
            start=1,
        )
    ]

    retrieved = [
        {
            "document": document["document_name"],
            "document_id": document["document_id"],
            "page": 10,
            "chunk": 1,
            "content": (
                f"{document['company_name']} reported "
                "financial performance information."
            ),
            "semantic_similarity": 0.9,
            "final_score": 0.9,
        }
        for document in documents
    ]

    with (
        patch(
            "app.rag.rag_service.resolve_retrieval_scope"
        ) as mock_scope,
        patch("app.rag.rag_service.embed_text") as mock_embed,
        patch("app.rag.rag_service.hybrid_search") as mock_search,
        patch("app.rag.rag_service.financial_metric_search") as mock_metric_search,
        patch("app.rag.rag_service.rerank_results") as mock_rank,
        patch("app.rag.rag_service.get_openai_client") as mock_client,
        patch("app.rag.rag_service.log_rag_request"),
        patch("app.rag.rag_service.record_rag_metrics"),
    ):
        mock_scope.return_value = {
            "company": None,
            "companies": [
                {"company_name": company}
                for company in companies
            ],
            "selections": [],
            "documents": documents,
            "document_ids": selected_ids,
            "filtered": True,
        }

        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = retrieved
        mock_metric_search.return_value = []
        mock_rank.return_value = retrieved

        client = mock_client.return_value
        client.responses.create.return_value.output_text = (
            ABSTENTION_TEXT
        )

        answer_question(
            question,
            request_id=f"evidence-policy-{expected_mode}",
        )

        prompt = client.responses.create.call_args.kwargs["input"]

    assert "EVIDENCE SUFFICIENCY AND PROPORTIONATE ANSWERING" in prompt
    assert "answer the" in prompt.lower()
    assert "supported parts" in prompt
    assert "If no substantive requested component is supported" in prompt

    if expected_mode == "broad":
        assert "synthesise supported findings" in prompt

    elif expected_mode == "exact":
        assert "single-metric factual question" in prompt
        assert "NOT ALLOWED" in prompt

    elif expected_mode == "comparison":
        assert "MULTI-COMPANY COMPARISON CONTROL:" in prompt
        assert "Not verified from retrieved evidence" in prompt
        assert "Never use one company's evidence" in prompt
