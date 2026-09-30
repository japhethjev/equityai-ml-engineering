import pytest

from app.rag.company_selector import (
    AmbiguousCompanyError,
    resolve_company_from_question,
)


APPLE = {
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "exchange": "NASDAQ",
    "market": "US",
    "country": "United States",
}

GTCO = {
    "company_name": "Guaranty Trust Holding Company Plc",
    "ticker": "GTCO",
    "exchange": "NGX",
    "market": "Nigeria",
    "country": "Nigeria",
}

ABC_NASDAQ = {
    "company_name": "ABC Technologies Inc.",
    "ticker": "ABC",
    "exchange": "NASDAQ",
    "market": "US",
    "country": "United States",
}

ABC_LSE = {
    "company_name": "ABC Holdings Plc",
    "ticker": "ABC",
    "exchange": "LSE",
    "market": "UK",
    "country": "United Kingdom",
}

CAT = {
    "company_name": "Caterpillar Inc.",
    "ticker": "CAT",
    "exchange": "NYSE",
    "market": "US",
    "country": "United States",
}


def test_resolves_company_by_ticker():
    result = resolve_company_from_question(
        "What was AAPL revenue in Q2 2026?",
        [APPLE, GTCO],
    )

    assert result == APPLE


def test_resolves_company_by_full_name():
    result = resolve_company_from_question(
        "What was Apple Inc. revenue in FY2025?",
        [APPLE, GTCO],
    )

    assert result == APPLE


def test_duplicate_ticker_is_ambiguous():
    with pytest.raises(
        AmbiguousCompanyError
    ):
        resolve_company_from_question(
            "What was ABC revenue in FY2025?",
            [ABC_NASDAQ, ABC_LSE],
        )


def test_exchange_disambiguates_duplicate_ticker():
    result = resolve_company_from_question(
        "What was ABC NASDAQ revenue in FY2025?",
        [ABC_NASDAQ, ABC_LSE],
    )

    assert result == ABC_NASDAQ


def test_unknown_company_returns_none():
    result = resolve_company_from_question(
        "What was XYZ revenue in FY2025?",
        [APPLE, GTCO],
    )

    assert result is None


def test_ticker_does_not_match_inside_word():
    result = resolve_company_from_question(
        "Calculate the operating margin.",
        [CAT],
    )

    assert result is None
