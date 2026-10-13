import pytest

from app.rag.rag_service import (
    select_broad_financial_evidence,
)


def make_result(page, score, categories=None):
    return {
        "document_id": "doc-1",
        "page": page,
        "chunk": 1,
        "content": f"Evidence from page {page}",
        "final_score": score,
        "evidence_categories": categories or [],
    }


def test_broad_selection_covers_all_categories():
    ranked = [
        make_result(2, 0.99),
        make_result(6, 0.95),
        make_result(16, 0.90, ["profitability"]),
        make_result(18, 0.85, ["financial_position"]),
        make_result(21, 0.80, ["cash_flows"]),
        make_result(32, 0.75, ["risks"]),
    ]

    selected = select_broad_financial_evidence(
        ranked, limit=6
    )

    covered = {
        category
        for result in selected
        for category in result["evidence_categories"]
    }

    assert covered == {
        "profitability",
        "financial_position",
        "cash_flows",
        "risks",
    }

    assert len(selected) == 6


def test_broad_selection_deduplicates_passages():
    passage = make_result(
        16, 0.90, ["profitability"]
    )

    selected = select_broad_financial_evidence(
        [passage, passage.copy()],
        limit=8,
    )

    assert len(selected) == 1


def test_broad_selection_handles_missing_categories():
    ranked = [
        make_result(16, 0.90, ["profitability"]),
        make_result(18, 0.85, ["financial_position"]),
    ]

    selected = select_broad_financial_evidence(
        ranked, limit=8
    )

    assert len(selected) == 2


def test_broad_selection_rejects_invalid_limit():
    with pytest.raises(ValueError):
        select_broad_financial_evidence(
            [], limit=0
        )
