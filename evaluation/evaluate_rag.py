import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.rag.rag_service import answer_question


DATASET_PATH = Path("evaluation/rag_eval_dataset.json")
RESULTS_DIR = Path("evaluation/results")
LATEST_JSON_PATH = RESULTS_DIR / "latest.json"
LATEST_CSV_PATH = RESULTS_DIR / "latest.csv"

ABSTENTION_TEXT = (
    "I could not find sufficient evidence "
    "in the provided documents."
)

# Maximum relative numerical difference accepted for
# approximate financial answers.
NUMERIC_REL_TOLERANCE = 0.005


def load_dataset() -> list[dict]:
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_financial_text(
    text: str | None,
) -> str:
    """
    Normalize common financial notation for textual comparison.
    """

    if text is None:
        return ""

    text = text.lower().strip()

    text = text.replace("ngn", "₦")

    text = re.sub(
        r"\btn\b",
        "trillion",
        text,
    )

    text = re.sub(
        r"\bbn\b",
        "billion",
        text,
    )

    text = re.sub(
        r"\bmn\b",
        "million",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def is_abstention(answer: str) -> bool:
    """
    Determine whether the model correctly abstained.
    """

    return (
        ABSTENTION_TEXT.lower()
        in answer.lower().strip()
    )


def unit_multiplier(
    unit: str | None,
) -> float:
    """
    Convert financial magnitude words into a common scale.
    """

    if not unit:
        return 1.0

    unit = unit.lower()

    multipliers = {
        "trillion": 1_000_000_000_000,
        "tn": 1_000_000_000_000,
        "billion": 1_000_000_000,
        "bn": 1_000_000_000,
        "million": 1_000_000,
        "mn": 1_000_000,
    }

    return multipliers.get(
        unit,
        1.0,
    )


def extract_financial_values(
    text: str | None,
) -> list[float]:
    """
    Extract currency-denominated financial values and convert
    them to a common base-unit representation.

    Examples:
        ₦3,433 billion -> 3.433e12
        ₦3.433 trillion -> 3.433e12
        ₦2.50tn -> 2.50e12
    """

    if not text:
        return []

    pattern = re.compile(
        r"(?:₦|ngn)\s*"
        r"\(?\s*"
        r"([\d,]+(?:\.\d+)?)"
        r"\s*\)?"
        r"\s*"
        r"(trillion|billion|million|tn|bn|mn)?",
        re.IGNORECASE,
    )

    values = []

    for match in pattern.finditer(text):
        number_text = (
            match.group(1)
            .replace(",", "")
        )

        try:
            number = float(number_text)
        except ValueError:
            continue

        multiplier = unit_multiplier(
            match.group(2)
        )

        values.append(
            number * multiplier
        )

    return values


def extract_percentages(
    text: str | None,
) -> list[float]:
    """
    Extract percentage values from text.
    """

    if not text:
        return []

    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*%"
    )

    return [
        float(match.group(1))
        for match in pattern.finditer(text)
    ]


def values_close(
    actual: float,
    expected: float,
) -> bool:
    """
    Compare numerical values using a controlled
    relative tolerance.
    """

    if actual == expected:
        return True

    denominator = max(
        abs(expected),
        1.0,
    )

    relative_error = (
        abs(actual - expected)
        / denominator
    )

    return (
        relative_error
        <= NUMERIC_REL_TOLERANCE
    )


def numeric_answer_match(
    answer: str,
    expected_answer: str,
) -> bool:
    """
    Compare financial numbers after converting units.
    """

    expected_values = (
        extract_financial_values(
            expected_answer
        )
    )

    actual_values = (
        extract_financial_values(
            answer
        )
    )

    if expected_values:
        for expected in expected_values:
            for actual in actual_values:
                if values_close(
                    actual,
                    expected,
                ):
                    return True

    expected_percentages = (
        extract_percentages(
            expected_answer
        )
    )

    actual_percentages = (
        extract_percentages(
            answer
        )
    )

    if expected_percentages:
        for expected in expected_percentages:
            for actual in actual_percentages:
                if values_close(
                    actual,
                    expected,
                ):
                    return True

    return False


def answer_matches(
    answer: str,
    expected_answer: str | None,
) -> bool:
    """
    Evaluate an answer using normalized text and
    numerical financial equivalence.
    """

    if expected_answer is None:
        return False

    normalized_answer = (
        normalize_financial_text(
            answer
        )
    )

    normalized_expected = (
        normalize_financial_text(
            expected_answer
        )
    )

    if (
        normalized_expected
        in normalized_answer
    ):
        return True

    return numeric_answer_match(
        answer,
        expected_answer,
    )


def evaluate_case(case: dict) -> dict:
    """
    Run and score one RAG evaluation case.
    """

    result = answer_question(
        case["question"]
    )

    answer = result.get(
        "answer",
        "",
    )

    sources = result.get(
        "sources",
        [],
    )

    answerable = case["answerable"]

    evaluation = {
        "id": case["id"],
        "category": case.get(
            "category",
            "uncategorized",
        ),
        "question": case["question"],
        "answerable": answerable,
        "answer": answer,
        "expected_answer": case.get(
            "expected_answer"
        ),
        "expected_document": case.get(
            "expected_document"
        ),
        "expected_page": case.get(
            "expected_page"
        ),
        "answer_match": None,
        "retrieval_hit": None,
        "top1_hit": None,
        "citation_match": None,
        "abstention_match": None,
        "passed": False,
        "sources": sources,
    }

    if not answerable:
        evaluation["abstention_match"] = (
            is_abstention(answer)
        )

        evaluation["passed"] = (
            evaluation[
                "abstention_match"
            ]
        )

        return evaluation

    expected_document = (
        case["expected_document"]
    )

    expected_page = (
        case["expected_page"]
    )

    evaluation["answer_match"] = (
        answer_matches(
            answer,
            case["expected_answer"],
        )
    )

    evaluation["retrieval_hit"] = any(
        source.get("document")
        == expected_document
        and source.get("page")
        == expected_page
        for source in sources
    )

    if sources:
        top_source = sources[0]

        evaluation["top1_hit"] = (
            top_source.get("document")
            == expected_document
            and top_source.get("page")
            == expected_page
        )
    else:
        evaluation["top1_hit"] = False

    normalized_answer = (
        answer.lower()
    )

    document_name = (
        expected_document.lower()
    )

    page_patterns = [
        f"page {expected_page}",
        f"page: {expected_page}",
        f"p.{expected_page}",
        f"p. {expected_page}",
    ]

    evaluation["citation_match"] = (
        document_name
        in normalized_answer
        and any(
            pattern in normalized_answer
            for pattern in page_patterns
        )
    )

    evaluation["passed"] = all(
        [
            evaluation["answer_match"],
            evaluation["retrieval_hit"],
            evaluation["top1_hit"],
            evaluation["citation_match"],
        ]
    )

    return evaluation


def calculate_rate(
    results: list[dict],
    metric: str,
) -> float | None:
    """
    Calculate a percentage while ignoring
    non-applicable None values.
    """

    applicable = [
        result[metric]
        for result in results
        if result.get(metric)
        is not None
    ]

    if not applicable:
        return None

    return (
        sum(applicable)
        / len(applicable)
        * 100
    )


def format_rate(
    rate: float | None,
) -> str:
    if rate is None:
        return "N/A"

    return f"{rate:.1f}%"


def build_category_metrics(
    results: list[dict],
) -> dict:
    """
    Build machine-readable performance metrics
    for every benchmark category.
    """

    grouped = defaultdict(list)

    for result in results:
        grouped[
            result["category"]
        ].append(result)

    category_metrics = {}

    for category in sorted(grouped):
        category_results = (
            grouped[category]
        )

        category_metrics[category] = {
            "cases": len(
                category_results
            ),
            "passed": sum(
                result["passed"]
                for result
                in category_results
            ),
            "pass_rate": calculate_rate(
                category_results,
                "passed",
            ),
            "answer_accuracy": calculate_rate(
                category_results,
                "answer_match",
            ),
            "retrieval_hit_rate": calculate_rate(
                category_results,
                "retrieval_hit",
            ),
            "top1_accuracy": calculate_rate(
                category_results,
                "top1_hit",
            ),
            "citation_accuracy": calculate_rate(
                category_results,
                "citation_match",
            ),
            "abstention_accuracy": calculate_rate(
                category_results,
                "abstention_match",
            ),
        }

    return category_metrics


def build_summary(
    dataset: list[dict],
    results: list[dict],
    failures: int,
) -> dict:
    """
    Build overall regression metrics.
    """

    answerable_count = sum(
        1
        for case in dataset
        if case["answerable"]
    )

    unanswerable_count = (
        len(dataset)
        - answerable_count
    )

    passed_cases = sum(
        result["passed"]
        for result in results
    )

    failed_cases = (
        len(results)
        - passed_cases
    )

    return {
        "total_cases": len(dataset),
        "cases_completed": len(results),
        "evaluation_failures": failures,
        "answerable_cases": answerable_count,
        "unanswerable_cases": (
            unanswerable_count
        ),
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "overall_pass_rate": (
            passed_cases
            / len(results)
            * 100
            if results
            else 0.0
        ),
        "answer_accuracy": calculate_rate(
            results,
            "answer_match",
        ),
        "retrieval_hit_rate": calculate_rate(
            results,
            "retrieval_hit",
        ),
        "top1_retrieval_accuracy": (
            calculate_rate(
                results,
                "top1_hit",
            )
        ),
        "citation_accuracy": calculate_rate(
            results,
            "citation_match",
        ),
        "abstention_accuracy": (
            calculate_rate(
                results,
                "abstention_match",
            )
        ),
    }


def save_json_report(
    summary: dict,
    category_metrics: dict,
    results: list[dict],
) -> None:
    """
    Save the complete regression report as JSON.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "benchmark": (
            DATASET_PATH.as_posix()
        ),
        "numeric_relative_tolerance": (
            NUMERIC_REL_TOLERANCE
        ),
        "summary": summary,
        "categories": category_metrics,
        "cases": results,
    }

    with LATEST_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_csv_report(
    results: list[dict],
) -> None:
    """
    Save case-level regression results as CSV.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "id",
        "category",
        "question",
        "answerable",
        "expected_answer",
        "answer",
        "answer_match",
        "retrieval_hit",
        "top1_hit",
        "citation_match",
        "abstention_match",
        "passed",
    ]

    with LATEST_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field: result.get(
                        field
                    )
                    for field
                    in fieldnames
                }
            )


def print_category_summary(
    category_metrics: dict,
) -> None:
    """
    Print performance for each benchmark category.
    """

    print()
    print("PERFORMANCE BY CATEGORY")
    print("=" * 60)

    for category, metrics in (
        category_metrics.items()
    ):
        print(f"{category}:")
        print(
            f"  Cases: "
            f"{metrics['cases']}"
        )
        print(
            f"  Passed: "
            f"{metrics['passed']}"
        )
        print(
            "  Pass rate: "
            f"{format_rate(metrics['pass_rate'])}"
        )

        if (
            metrics["answer_accuracy"]
            is not None
        ):
            print(
                "  Answer accuracy: "
                f"{format_rate(metrics['answer_accuracy'])}"
            )

        if (
            metrics[
                "retrieval_hit_rate"
            ]
            is not None
        ):
            print(
                "  Retrieval hit rate: "
                f"{format_rate(metrics['retrieval_hit_rate'])}"
            )

        if (
            metrics[
                "abstention_accuracy"
            ]
            is not None
        ):
            print(
                "  Abstention accuracy: "
                f"{format_rate(metrics['abstention_accuracy'])}"
            )

        print("-" * 60)


def main() -> None:
    dataset = load_dataset()

    results = []
    failures = 0

    print(
        f"\nRunning {len(dataset)} "
        "RAG evaluation cases...\n"
    )

    for case in dataset:
        print(
            f"Evaluating: {case['id']}"
        )
        print(
            f"Category: "
            f"{case.get('category', 'uncategorized')}"
        )
        print(
            f"Question: "
            f"{case['question']}"
        )

        case_type = (
            "ANSWERABLE"
            if case["answerable"]
            else "UNANSWERABLE"
        )

        print(
            f"Case type: {case_type}"
        )

        try:
            evaluation = (
                evaluate_case(case)
            )

            results.append(
                evaluation
            )

            print(
                f"Answer: "
                f"{evaluation['answer']}"
            )

            if case["answerable"]:
                print(
                    "Expected answer: "
                    f"{case['expected_answer']}"
                )
                print(
                    "Answer match: "
                    f"{evaluation['answer_match']}"
                )
                print(
                    "Retrieval hit: "
                    f"{evaluation['retrieval_hit']}"
                )
                print(
                    "Top-1 hit: "
                    f"{evaluation['top1_hit']}"
                )
                print(
                    "Citation match: "
                    f"{evaluation['citation_match']}"
                )
            else:
                print(
                    "Expected behaviour: "
                    "ABSTAIN"
                )
                print(
                    "Abstention match: "
                    f"{evaluation['abstention_match']}"
                )

            print(
                f"Case passed: "
                f"{evaluation['passed']}"
            )

        except Exception as error:
            failures += 1

            print(
                "Evaluation error: "
                f"{error}"
            )

        print("-" * 60)

    summary = build_summary(
        dataset,
        results,
        failures,
    )

    category_metrics = (
        build_category_metrics(
            results
        )
    )

    save_json_report(
        summary,
        category_metrics,
        results,
    )

    save_csv_report(
        results
    )

    print()
    print("EQUITYAI RAG EVALUATION")
    print("=" * 60)

    print(
        f"Cases completed: "
        f"{summary['cases_completed']}/"
        f"{summary['total_cases']}"
    )

    print(
        f"Evaluation failures: "
        f"{summary['evaluation_failures']}"
    )

    print(
        f"Passed cases: "
        f"{summary['passed_cases']}"
    )

    print(
        f"Failed cases: "
        f"{summary['failed_cases']}"
    )

    print(
        "Overall pass rate: "
        f"{format_rate(summary['overall_pass_rate'])}"
    )

    print("-" * 60)

    print(
        "Answer accuracy: "
        f"{format_rate(summary['answer_accuracy'])}"
    )

    print(
        "Retrieval hit rate: "
        f"{format_rate(summary['retrieval_hit_rate'])}"
    )

    print(
        "Top-1 retrieval accuracy: "
        f"{format_rate(summary['top1_retrieval_accuracy'])}"
    )

    print(
        "Citation accuracy: "
        f"{format_rate(summary['citation_accuracy'])}"
    )

    print(
        "Abstention accuracy: "
        f"{format_rate(summary['abstention_accuracy'])}"
    )

    print("=" * 60)

    print_category_summary(
        category_metrics
    )

    print()
    print("REGRESSION REPORTS")
    print("=" * 60)
    print(
        f"JSON: {LATEST_JSON_PATH}"
    )
    print(
        f"CSV:  {LATEST_CSV_PATH}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()