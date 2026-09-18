import json
import re
from collections import defaultdict
from pathlib import Path

from app.rag.rag_service import answer_question


DATASET_PATH = Path("evaluation/rag_eval_dataset.json")

ABSTENTION_TEXT = (
    "I could not find sufficient evidence "
    "in the provided documents."
)

# Relative tolerance for approximate financial answers.
# 0.5% allows reasonable display rounding while remaining strict.
NUMERIC_REL_TOLERANCE = 0.005


def load_dataset() -> list[dict]:
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_financial_text(text: str | None) -> str:
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


def unit_multiplier(unit: str | None) -> float:
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

    return multipliers.get(unit, 1.0)


def extract_financial_values(text: str | None) -> list[float]:
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


def extract_percentages(text: str | None) -> list[float]:
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

    This allows economically equivalent representations such as:

        ₦3,433 billion
        ₦3.433 trillion

    and reasonable rounding such as:

        ₦2.50 trillion
        ₦2.504 trillion
    """

    expected_values = extract_financial_values(
        expected_answer
    )

    actual_values = extract_financial_values(
        answer
    )

    if expected_values:
        for expected in expected_values:
            for actual in actual_values:
                if values_close(
                    actual,
                    expected,
                ):
                    return True

    expected_percentages = extract_percentages(
        expected_answer
    )

    actual_percentages = extract_percentages(
        answer
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
    Evaluate an answer using both normalized text
    and numerical financial equivalence.
    """

    if expected_answer is None:
        return False

    normalized_answer = (
        normalize_financial_text(answer)
    )

    normalized_expected = (
        normalize_financial_text(
            expected_answer
        )
    )

    # First use the existing textual comparison.
    if normalized_expected in normalized_answer:
        return True

    # Then check numerical equivalence.
    if numeric_answer_match(
        answer,
        expected_answer,
    ):
        return True

    return False


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
        "answer_match": None,
        "retrieval_hit": None,
        "top1_hit": None,
        "citation_match": None,
        "abstention_match": None,
    }

    if not answerable:
        evaluation["abstention_match"] = (
            is_abstention(answer)
        )

        return evaluation

    expected_document = case[
        "expected_document"
    ]

    expected_page = case[
        "expected_page"
    ]

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

    # Current answer-generation contract requires
    # the answer itself to identify document/page.
    normalized_answer = answer.lower()

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


def print_category_summary(
    results: list[dict],
) -> None:
    """
    Report performance separately for each
    benchmark category.
    """

    grouped = defaultdict(list)

    for result in results:
        grouped[
            result["category"]
        ].append(result)

    print()
    print("PERFORMANCE BY CATEGORY")
    print("=" * 60)

    for category in sorted(grouped):
        category_results = grouped[
            category
        ]

        answer_rate = calculate_rate(
            category_results,
            "answer_match",
        )

        abstention_rate = calculate_rate(
            category_results,
            "abstention_match",
        )

        retrieval_rate = calculate_rate(
            category_results,
            "retrieval_hit",
        )

        print(
            f"{category}:"
        )
        print(
            f"  Cases: "
            f"{len(category_results)}"
        )

        if answer_rate is not None:
            print(
                "  Answer accuracy: "
                f"{format_rate(answer_rate)}"
            )

        if abstention_rate is not None:
            print(
                "  Abstention accuracy: "
                f"{format_rate(abstention_rate)}"
            )

        if retrieval_rate is not None:
            print(
                "  Retrieval hit rate: "
                f"{format_rate(retrieval_rate)}"
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
            f"Question: {case['question']}"
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
            evaluation = evaluate_case(
                case
            )

            results.append(evaluation)

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

        except Exception as error:
            failures += 1

            print(
                "Evaluation error: "
                f"{error}"
            )

        print("-" * 60)

    answerable_count = sum(
        1
        for case in dataset
        if case["answerable"]
    )

    unanswerable_count = (
        len(dataset)
        - answerable_count
    )

    print()
    print("EQUITYAI RAG EVALUATION")
    print("=" * 60)
    print(
        f"Cases completed: "
        f"{len(results)}/{len(dataset)}"
    )
    print(
        f"Evaluation failures: "
        f"{failures}"
    )
    print(
        f"Answerable cases: "
        f"{answerable_count}"
    )
    print(
        f"Unanswerable cases: "
        f"{unanswerable_count}"
    )
    print("-" * 60)

    print(
        "Answer accuracy: "
        f"{format_rate(calculate_rate(results, 'answer_match'))}"
    )

    print(
        "Retrieval hit rate: "
        f"{format_rate(calculate_rate(results, 'retrieval_hit'))}"
    )

    print(
        "Top-1 retrieval accuracy: "
        f"{format_rate(calculate_rate(results, 'top1_hit'))}"
    )

    print(
        "Citation accuracy: "
        f"{format_rate(calculate_rate(results, 'citation_match'))}"
    )

    print(
        "Abstention accuracy: "
        f"{format_rate(calculate_rate(results, 'abstention_match'))}"
    )

    print("=" * 60)

    print_category_summary(
        results
    )


if __name__ == "__main__":
    main()