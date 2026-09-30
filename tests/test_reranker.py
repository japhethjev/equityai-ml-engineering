from app.rag.reranker import (
    select_balanced_evidence,
)


def make_result(
    document_id: str,
    score: float,
    content: str,
) -> dict:
    return {
        "document_id": document_id,
        "final_score": score,
        "content": content,
    }


def test_balanced_evidence_preserves_both_documents():
    first_id = "doc-2025"
    second_id = "doc-2026"

    ranked = [
        make_result(
            second_id,
            0.99,
            "2026 strongest result",
        ),
        make_result(
            second_id,
            0.95,
            "2026 second result",
        ),
        make_result(
            second_id,
            0.90,
            "2026 third result",
        ),
        make_result(
            first_id,
            0.80,
            "2025 strongest result",
        ),
        make_result(
            first_id,
            0.70,
            "2025 second result",
        ),
    ]

    selected = select_balanced_evidence(
        ranked,
        document_ids=[
            first_id,
            second_id,
        ],
        limit=3,
    )

    selected_ids = {
        result["document_id"]
        for result in selected
    }

    assert len(selected) == 3
    assert first_id in selected_ids
    assert second_id in selected_ids


def test_single_document_preserves_top_n():
    ranked = [
        make_result("doc-1", 0.9, "First"),
        make_result("doc-1", 0.8, "Second"),
        make_result("doc-1", 0.7, "Third"),
        make_result("doc-1", 0.6, "Fourth"),
    ]

    selected = select_balanced_evidence(
        ranked,
        document_ids=["doc-1"],
        limit=3,
    )

    assert selected == ranked[:3]


def test_unrestricted_retrieval_preserves_top_n():
    ranked = [
        make_result("doc-1", 0.9, "First"),
        make_result("doc-2", 0.8, "Second"),
        make_result("doc-3", 0.7, "Third"),
        make_result("doc-4", 0.6, "Fourth"),
    ]

    selected = select_balanced_evidence(
        ranked,
        document_ids=None,
        limit=3,
    )

    assert selected == ranked[:3]


def test_balanced_evidence_fills_remaining_slots_by_rank():
    first_id = "doc-2025"
    second_id = "doc-2026"

    ranked = [
        make_result(second_id, 0.99, "2026 first"),
        make_result(second_id, 0.95, "2026 second"),
        make_result(first_id, 0.80, "2025 first"),
        make_result(first_id, 0.70, "2025 second"),
    ]

    selected = select_balanced_evidence(
        ranked,
        document_ids=[
            first_id,
            second_id,
        ],
        limit=3,
    )

    assert selected[0]["document_id"] == first_id
    assert selected[1]["document_id"] == second_id
    assert selected[2]["content"] == "2026 second"
