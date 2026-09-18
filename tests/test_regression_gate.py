from evaluation.compare_regression import (
    classify_run,
    find_case_regressions,
    metric_change,
)


def make_report(
    total_cases=30,
    cases_completed=30,
    evaluation_failures=0,
):
    return {
        "summary": {
            "total_cases": total_cases,
            "cases_completed": cases_completed,
            "evaluation_failures": evaluation_failures,
        },
        "cases": [],
    }


def test_valid_evaluation_run():
    report = make_report()

    status, reason = classify_run(report)

    assert status == "VALID"
    assert reason is None


def test_infrastructure_failure_is_invalid():
    report = make_report(
        cases_completed=0,
        evaluation_failures=30,
    )

    status, reason = classify_run(report)

    assert status == "INVALID"
    assert reason is not None


def test_incomplete_run_is_invalid():
    report = make_report(
        cases_completed=29,
    )

    status, reason = classify_run(report)

    assert status == "INVALID"
    assert reason is not None


def test_metric_regression_delta():
    change = metric_change(
        100.0,
        95.0,
    )

    assert change == -5.0


def test_stable_metric_delta():
    change = metric_change(
        100.0,
        100.0,
    )

    assert change == 0.0


def test_case_regression_detected():
    baseline = {
        "cases": [
            {
                "id": "dangote_001",
                "passed": True,
            }
        ]
    }

    current = {
        "cases": [
            {
                "id": "dangote_001",
                "passed": False,
            }
        ]
    }

    regressions = find_case_regressions(
        baseline,
        current,
    )

    assert regressions == [
        "dangote_001"
    ]


def test_stable_case_not_regression():
    baseline = {
        "cases": [
            {
                "id": "dangote_001",
                "passed": True,
            }
        ]
    }

    current = {
        "cases": [
            {
                "id": "dangote_001",
                "passed": True,
            }
        ]
    }

    regressions = find_case_regressions(
        baseline,
        current,
    )

    assert regressions == []