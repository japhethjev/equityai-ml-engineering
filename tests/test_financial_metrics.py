import pytest

from app.rag.financial_metrics import (
    get_metric,
    get_metric_group,
    resolve_metric,
)


def test_get_reported_metric():
    metric = get_metric("revenue")

    assert metric.key == "revenue"
    assert metric.kind == "reported"
    assert metric.required_inputs == ()


def test_get_derived_margin():
    metric = get_metric(
        "gross_profit_margin"
    )

    assert metric.kind == "derived"
    assert metric.required_inputs == (
        "gross_profit",
        "revenue",
    )
    assert metric.output_unit == "%"


def test_net_margin_uses_pat_and_revenue():
    metric = get_metric(
        "net_profit_margin"
    )

    assert metric.required_inputs == (
        "profit_after_tax",
        "revenue",
    )


def test_debt_to_equity_inputs():
    metric = get_metric(
        "debt_to_equity"
    )

    assert metric.required_inputs == (
        "total_debt",
        "total_equity",
    )


def test_current_ratio_inputs():
    metric = get_metric(
        "current_ratio"
    )

    assert metric.required_inputs == (
        "current_assets",
        "current_liabilities",
    )


def test_operating_cash_flow_is_reported():
    metric = get_metric(
        "operating_cash_flow"
    )

    assert metric.kind == "reported"


def test_resolve_gross_margin_alias():
    metric = resolve_metric(
        "gross margin"
    )

    assert metric is not None
    assert metric.key == (
        "gross_profit_margin"
    )


def test_resolve_net_income_alias():
    metric = resolve_metric(
        "net income"
    )

    assert metric is not None
    assert metric.key == "profit_after_tax"


def test_resolve_debt_equity_alias():
    metric = resolve_metric(
        "debt-to-equity"
    )

    assert metric is not None
    assert metric.key == "debt_to_equity"


def test_resolve_unknown_metric_returns_none():
    assert (
        resolve_metric(
            "completely unknown metric"
        )
        is None
    )


def test_liquidity_group():
    metrics = get_metric_group(
        "liquidity"
    )

    keys = tuple(
        metric.key
        for metric in metrics
    )

    assert keys == (
        "current_ratio",
        "quick_ratio",
        "cash_ratio",
    )


def test_profitability_group():
    metrics = get_metric_group(
        "profitability"
    )

    keys = {
        metric.key
        for metric in metrics
    }

    assert "gross_profit_margin" in keys
    assert "operating_profit_margin" in keys
    assert "net_profit_margin" in keys


def test_unknown_metric_raises():
    with pytest.raises(
        KeyError,
        match="Unknown financial metric",
    ):
        get_metric(
            "unknown_metric"
        )


def test_unknown_group_raises():
    with pytest.raises(
        KeyError,
        match=(
            "Unknown financial metric group"
        ),
    ):
        get_metric_group(
            "unknown_group"
        )
