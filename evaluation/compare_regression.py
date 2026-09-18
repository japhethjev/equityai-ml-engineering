import json
import sys
from pathlib import Path


BASELINE_PATH = Path("evaluation/baseline.json")
LATEST_PATH = Path("evaluation/results/latest.json")

METRICS = [
    "overall_pass_rate",
    "answer_accuracy",
    "retrieval_hit_rate",
    "top1_retrieval_accuracy",
    "citation_accuracy",
    "abstention_accuracy",
]


def load_report(path: Path) -> dict:
    """
    Load an evaluation report from disk.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def format_metric(value) -> str:
    """
    Format percentage metrics consistently.
    """

    if value is None:
        return "N/A"

    return f"{value:.1f}%"


def metric_change(
    baseline_value,
    current_value,
):
    """
    Calculate percentage-point change.
    """

    if (
        baseline_value is None
        or current_value is None
    ):
        return None

    return current_value - baseline_value


def format_change(change) -> str:
    """
    Format a percentage-point delta.
    """

    if change is None:
        return "N/A"

    return f"{change:+.1f} pp"


def classify_run(report: dict) -> tuple[str, str | None]:
    """
    Determine whether an evaluation run is valid.

    Infrastructure/execution failures must not be
    interpreted as model-quality regressions.
    """

    summary = report.get(
        "summary",
        {},
    )

    total_cases = summary.get(
        "total_cases",
        0,
    )

    completed = summary.get(
        "cases_completed",
        0,
    )

    evaluation_failures = summary.get(
        "evaluation_failures",
        0,
    )

    if evaluation_failures > 0:
        return (
            "INVALID",
            (
                f"{evaluation_failures} evaluation "
                "case(s) failed during execution."
            ),
        )

    if total_cases <= 0:
        return (
            "INVALID",
            "Benchmark contains no evaluation cases.",
        )

    if completed != total_cases:
        return (
            "INVALID",
            (
                f"Only {completed}/{total_cases} "
                "cases completed."
            ),
        )

    return (
        "VALID",
        None,
    )


def case_map(report: dict) -> dict:
    """
    Map evaluation case IDs to their results.
    """

    return {
        case["id"]: case
        for case in report.get(
            "cases",
            []
        )
        if case.get("id")
    }


def find_case_regressions(
    baseline: dict,
    current: dict,
) -> list[str]:
    """
    Find cases that passed in the baseline but
    fail in the current evaluation.
    """

    baseline_cases = case_map(
        baseline
    )

    current_cases = case_map(
        current
    )

    regressions = []

    for case_id, baseline_case in (
        baseline_cases.items()
    ):
        current_case = (
            current_cases.get(
                case_id
            )
        )

        if current_case is None:
            regressions.append(
                f"{case_id} (missing)"
            )
            continue

        baseline_passed = (
            baseline_case.get(
                "passed"
            )
        )

        current_passed = (
            current_case.get(
                "passed"
            )
        )

        if (
            baseline_passed is True
            and current_passed
            is not True
        ):
            regressions.append(
                case_id
            )

    return regressions


def find_new_failures(
    baseline: dict,
    current: dict,
) -> list[str]:
    """
    Find current failing cases that were not
    already failing in the baseline.
    """

    baseline_cases = case_map(
        baseline
    )

    new_failures = []

    for case in current.get(
        "cases",
        []
    ):
        if case.get("passed") is True:
            continue

        case_id = case.get("id")

        baseline_case = (
            baseline_cases.get(
                case_id
            )
        )

        if (
            baseline_case is None
            or baseline_case.get(
                "passed"
            )
            is True
        ):
            new_failures.append(
                case_id
            )

    return new_failures


def compare_categories(
    baseline: dict,
    current: dict,
) -> list[str]:
    """
    Detect category-level pass-rate regressions.
    """

    baseline_categories = (
        baseline.get(
            "categories",
            {}
        )
    )

    current_categories = (
        current.get(
            "categories",
            {}
        )
    )

    regressions = []

    for category, baseline_metrics in (
        baseline_categories.items()
    ):
        current_metrics = (
            current_categories.get(
                category
            )
        )

        if current_metrics is None:
            regressions.append(
                f"{category} (missing)"
            )
            continue

        baseline_rate = (
            baseline_metrics.get(
                "pass_rate"
            )
        )

        current_rate = (
            current_metrics.get(
                "pass_rate"
            )
        )

        if (
            baseline_rate is not None
            and current_rate is not None
            and current_rate
            < baseline_rate
        ):
            regressions.append(
                category
            )

    return regressions


def main() -> int:
    try:
        baseline = load_report(
            BASELINE_PATH
        )

        current = load_report(
            LATEST_PATH
        )

    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ) as error:
        print()
        print("REGRESSION ANALYSIS")
        print("=" * 60)
        print("STATUS: INVALID")
        print(f"Reason: {error}")
        print("=" * 60)

        return 2

    baseline_status, baseline_reason = (
        classify_run(
            baseline
        )
    )

    current_status, current_reason = (
        classify_run(
            current
        )
    )

    print()
    print("EQUITYAI RAG REGRESSION ANALYSIS")
    print("=" * 60)

    if baseline_status != "VALID":
        print(
            "BASELINE STATUS: INVALID"
        )
        print(
            f"Reason: {baseline_reason}"
        )
        print(
            "REGRESSION STATUS: "
            "NOT EVALUATED"
        )
        print("=" * 60)

        return 2

    if current_status != "VALID":
        print(
            "BENCHMARK STATUS: INVALID"
        )
        print(
            "INFRASTRUCTURE/EXECUTION "
            "FAILURE: YES"
        )
        print(
            f"Reason: {current_reason}"
        )
        print(
            "MODEL REGRESSION: "
            "NOT EVALUATED"
        )
        print("=" * 60)

        return 2

    baseline_summary = (
        baseline["summary"]
    )

    current_summary = (
        current["summary"]
    )

    metric_regressions = []

    print("BENCHMARK STATUS: VALID")
    print("-" * 60)

    for metric in METRICS:
        baseline_value = (
            baseline_summary.get(
                metric
            )
        )

        current_value = (
            current_summary.get(
                metric
            )
        )

        change = metric_change(
            baseline_value,
            current_value,
        )

        if (
            baseline_value is not None
            and current_value is not None
            and current_value
            < baseline_value
        ):
            status = "REGRESSION"

            metric_regressions.append(
                metric
            )
        else:
            status = "STABLE"

        print(
            f"{metric}:"
        )
        print(
            "  Baseline: "
            f"{format_metric(baseline_value)}"
        )
        print(
            "  Current:  "
            f"{format_metric(current_value)}"
        )
        print(
            "  Change:   "
            f"{format_change(change)}"
        )
        print(
            f"  Status:   {status}"
        )

    case_regressions = (
        find_case_regressions(
            baseline,
            current,
        )
    )

    new_failures = (
        find_new_failures(
            baseline,
            current,
        )
    )

    category_regressions = (
        compare_categories(
            baseline,
            current,
        )
    )

    print("-" * 60)

    print(
        "Case regressions: "
        f"{len(case_regressions)}"
    )

    if case_regressions:
        for case_id in (
            case_regressions
        ):
            print(
                f"  - {case_id}"
            )

    print(
        "Category regressions: "
        f"{len(category_regressions)}"
    )

    if category_regressions:
        for category in (
            category_regressions
        ):
            print(
                f"  - {category}"
            )

    regression_detected = any(
        [
            metric_regressions,
            case_regressions,
            new_failures,
            category_regressions,
        ]
    )

    print("-" * 60)

    if regression_detected:
        print(
            "MODEL REGRESSION: YES"
        )
        print(
            "REGRESSION STATUS: FAILED"
        )

        if metric_regressions:
            print(
                "Regressed metrics: "
                + ", ".join(
                    metric_regressions
                )
            )

        if new_failures:
            print(
                "New failing cases: "
                + ", ".join(
                    new_failures
                )
            )

        print("=" * 60)

        # Exit code 1 = genuine regression.
        return 1

    print(
        "MODEL REGRESSION: NO"
    )
    print(
        "REGRESSION STATUS: PASSED"
    )
    print("=" * 60)

    # Exit code 0 = successful regression gate.
    return 0


if __name__ == "__main__":
    sys.exit(main())