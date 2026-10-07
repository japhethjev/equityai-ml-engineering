import pytest

from app.rag.company_selector import (
    AmbiguousCompanyError,
    resolve_company_from_question,
    resolve_companies_from_question,
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


def test_unique_partial_company_name_resolves_tickerless_issuer():
    companies = [
        {
            "company_name": "Dangote Petroleum Refinery FZE",
            "ticker": None,
            "exchange": None,
            "market": None,
            "country": "Nigeria",
        }
    ]

    result = resolve_company_from_question(
        (
            "Show Dangote Petroleum Refinery revenue "
            "for the last 2 years."
        ),
        companies,
    )

    assert result is not None
    assert (
        result["company_name"]
        == "Dangote Petroleum Refinery FZE"
    )
    assert result["ticker"] is None


def test_partial_company_name_does_not_guess_when_ambiguous():
    companies = [
        {
            "company_name": "Dangote Petroleum Refinery FZE",
            "ticker": None,
            "exchange": None,
        },
        {
            "company_name": "Dangote Cement Plc",
            "ticker": "DANGCEM",
            "exchange": "NGX",
        },
    ]

    with pytest.raises(AmbiguousCompanyError):
        resolve_company_from_question(
            "Show Dangote revenue for the last 2 years.",
            companies,
        )


def test_resolves_multiple_explicit_companies():
    barclays = {
        "company_name": "Barclays Bank Plc",
        "ticker": None,
        "exchange": "LSE",
    }
    hsbc = {
        "company_name": "HSBC Holdings plc",
        "ticker": "HSBA",
        "exchange": "LSE",
    }
    deutsche = {
        "company_name": "Deutsche Bank",
        "ticker": "DBK",
        "exchange": "XETRA",
    }
    standard_chartered = {
        "company_name": "Standard Chartered Bank",
        "ticker": "STAN",
        "exchange": "LSE",
    }

    result = resolve_companies_from_question(
        (
            "Compare Barclays Bank Plc, HSBC Holdings plc, "
            "Deutsche Bank and Standard Chartered Bank "
            "using their 2025 annual reports."
        ),
        [
            barclays,
            hsbc,
            deutsche,
            standard_chartered,
        ],
    )

    assert result == [
        barclays,
        hsbc,
        deutsche,
        standard_chartered,
    ]


def test_multi_company_resolver_does_not_expand_ambiguous_partial_name():
    companies = [
        {
            "company_name": "Dangote Petroleum Refinery FZE",
            "ticker": None,
            "exchange": None,
        },
        {
            "company_name": "Dangote Cement Plc",
            "ticker": "DANGCEM",
            "exchange": "NGX",
        },
    ]

    result = resolve_companies_from_question(
        "Compare Dangote revenue in 2025.",
        companies,
    )

    assert result == []
