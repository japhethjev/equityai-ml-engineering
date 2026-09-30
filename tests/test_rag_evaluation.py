from evaluation.evaluate_rag import (
    ABSTENTION_TEXT,
    answer_matches,
    extract_financial_values,
    extract_percentages,
    is_abstention,
    values_close,
)


def test_trillion_billion_equivalence():
    assert answer_matches(
        "Gross profit was ₦3,433 billion.",
        "₦3.433 trillion",
    )


def test_financial_rounding_tolerance():
    assert answer_matches(
        "Revenue was ₦19.13 trillion.",
        "₦19.135 trillion",
    )


def test_materially_wrong_financial_value():
    assert not answer_matches(
        "Revenue was ₦18.0 trillion.",
        "₦19.135 trillion",
    )


def test_percentage_match():
    assert answer_matches(
        "Gross margin was 17.9%.",
        "17.9%",
    )


def test_percentage_wrong_value():
    assert not answer_matches(
        "Gross margin was 16.5%.",
        "17.9%",
    )


def test_abstention_detection():
    assert is_abstention(
        ABSTENTION_TEXT
    )


def test_financial_value_extraction():
    values = extract_financial_values(
        "Operating profit was ₦3.254 trillion."
    )

    assert len(values) == 1

    assert values_close(
        values[0],
        3_254_000_000_000,
    )


def test_percentage_extraction():
    percentages = extract_percentages(
        "Offer costs were 1.93%."
    )

    assert percentages == [1.93]

def test_compare_authorizes_comparison_intent():
    from app.rag.rag_service import (
        explicitly_requests_comparison,
    )

    assert explicitly_requests_comparison(
        "Compare revenue in FY2024 and FY2025."
    )


def test_versus_authorizes_comparison_intent():
    from app.rag.rag_service import (
        explicitly_requests_comparison,
    )

    assert explicitly_requests_comparison(
        "FY2024 versus FY2025 revenue"
    )


def test_growth_word_alone_does_not_authorize_comparison():
    from app.rag.rag_service import (
        explicitly_requests_comparison,
    )

    assert not explicitly_requests_comparison(
        "What revenue growth did the company report?"
    )


def test_increase_word_alone_does_not_authorize_comparison():
    from app.rag.rag_service import (
        explicitly_requests_comparison,
    )

    assert not explicitly_requests_comparison(
        "What increase in revenue did management report?"
    )


def test_single_period_factual_question_not_comparison():
    from app.rag.rag_service import (
        explicitly_requests_comparison,
    )

    assert not explicitly_requests_comparison(
        "What was revenue in Q2 2026?"
    )
