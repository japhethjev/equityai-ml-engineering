import json
import re
from pathlib import Path

from app.rag.rag_service import answer_question


DATASET_PATH = Path(
    "evaluation/rag_eval_dataset.json"
)

ABSTENTION_TEXT = (
    "I could not find sufficient evidence "
    "in the provided documents."
)


def load_dataset() -> list[dict]:
    """
    Load the RAG evaluation benchmark.
    """

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_financial_text(
    text: str,
) -> str:
    """
    Normalize common financial notation so that
    semantically equivalent financial answers
    can be compared consistently.

    Examples:
    ₦2.15tn -> ₦2.15 trillion
    4.10bn -> 4.10 billion
    NGN 525 -> ₦525
    """

    normalized = text.lower().strip()

    # Normalize currency representation.
    normalized = normalized.replace(
        "ngn",
        "₦",
    )

    # Remove unnecessary space between
    # naira symbol and number.
    normalized = re.sub(
        r"₦\s+(?=\d)",
        "₦",
        normalized,
    )

    # Normalize trillion abbreviations.
    normalized = re.sub(
        r"(?<=\d)\s*tn\b",
        " trillion",
        normalized,
    )

    # Normalize billion abbreviations.
    normalized = re.sub(
        r"(?<=\d)\s*bn\b",
        " billion",
        normalized,
    )

    # Normalize million abbreviations.
    normalized = re.sub(
        r"(?<=\d)\s*mn\b",
        " million",
        normalized,
    )

    # Normalize whitespace.
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized


def is_abstention(
    answer: str,
) -> bool:
    """
    Determine whether the RAG system correctly
    declined to answer because sufficient
    evidence was unavailable.
    """

    normalized_answer = (
        answer.lower().strip()
    )

    normalized_abstention = (
        ABSTENTION_TEXT.lower()
    )

    return (
        normalized_abstention
        in normalized_answer
    )


def evaluate_case(case: dict) -> dict:
    """
    Run one evaluation question through the
    production RAG pipeline.

    Answerable and unanswerable questions are
    evaluated differently.
    """

    result = answer_question(
        case["question"]
    )

    answer = result["answer"]
    sources = result["sources"]

    answerable = case["answerable"]

    # -----------------------------------------
    # Unanswerable question
    # -----------------------------------------

    if not answerable:

        abstention_match = is_abstention(
            answer
        )

        return {
            "id": case["id"],
            "question": case["question"],
            "answerable": False,
            "answer": answer,
            "answer_match": None,
            "retrieval_hit": None,
            "top1_hit": None,
            "citation_match": None,
            "abstention_match": (
                abstention_match
            ),
            "sources": sources,
        }

    # -----------------------------------------
    # Answerable question
    # -----------------------------------------

    normalized_expected = (
        normalize_financial_text(
            case["expected_answer"]
        )
    )

    normalized_answer = (
        normalize_financial_text(
            answer
        )
    )

    # -----------------------------------------
    # Answer accuracy
    # -----------------------------------------

    answer_match = (
        normalized_expected
        in normalized_answer
    )

    # -----------------------------------------
    # Retrieval hit
    #
    # Correct document + page appears anywhere
    # in the returned sources.
    # -----------------------------------------

    retrieval_hit = any(
        source["document"]
        == case["expected_document"]
        and source["page"]
        == case["expected_page"]
        for source in sources
    )

    # -----------------------------------------
    # Top-1 retrieval accuracy
    # -----------------------------------------

    top1_hit = False

    if sources:

        top_source = sources[0]

        top1_hit = (
            top_source["document"]
            == case["expected_document"]
            and top_source["page"]
            == case["expected_page"]
        )

    # -----------------------------------------
    # Citation accuracy
    # -----------------------------------------

    citation_match = (
        case["expected_document"]
        in answer
    )

    return {
        "id": case["id"],
        "question": case["question"],
        "answerable": True,
        "answer": answer,
        "expected_answer": case[
            "expected_answer"
        ],
        "normalized_answer": (
            normalized_answer
        ),
        "normalized_expected": (
            normalized_expected
        ),
        "answer_match": answer_match,
        "retrieval_hit": retrieval_hit,
        "top1_hit": top1_hit,
        "citation_match": citation_match,
        "abstention_match": None,
        "sources": sources,
    }


def calculate_rate(
    results: list[dict],
    metric: str,
) -> float:
    """
    Calculate percentage success rate for
    a boolean evaluation metric.

    Results where the metric is None are
    excluded from the denominator.
    """

    applicable_results = [
        result
        for result in results
        if result.get(metric) is not None
    ]

    if not applicable_results:
        return 0.0

    successful = sum(
        bool(result[metric])
        for result in applicable_results
    )

    return (
        successful
        / len(applicable_results)
        * 100
    )


def count_applicable(
    results: list[dict],
    metric: str,
) -> int:
    """
    Count cases to which a metric applies.
    """

    return sum(
        result.get(metric) is not None
        for result in results
    )


def main():
    """
    Run the complete EquityAI RAG evaluation.
    """

    dataset = load_dataset()

    print(
        f"\nRunning {len(dataset)} "
        "RAG evaluation cases...\n"
    )

    results = []

    failed_cases = 0

    for case in dataset:

        print(
            f"Evaluating: {case['id']}"
        )

        print(
            f"Question: {case['question']}"
        )

        print(
            "Case type: "
            + (
                "ANSWERABLE"
                if case["answerable"]
                else "UNANSWERABLE"
            )
        )

        try:

            result = evaluate_case(case)

            results.append(result)

            print(
                f"Answer: {result['answer']}"
            )

            if result["answerable"]:

                print(
                    "Expected answer: "
                    f"{result['expected_answer']}"
                )

                print(
                    "Answer match: "
                    f"{result['answer_match']}"
                )

                print(
                    "Retrieval hit: "
                    f"{result['retrieval_hit']}"
                )

                print(
                    "Top-1 hit: "
                    f"{result['top1_hit']}"
                )

                print(
                    "Citation match: "
                    f"{result['citation_match']}"
                )

            else:

                print(
                    "Expected behaviour: "
                    "ABSTAIN"
                )

                print(
                    "Abstention match: "
                    f"{result['abstention_match']}"
                )

        except Exception as error:

            failed_cases += 1

            print(
                f"Evaluation failed: {error}"
            )

        print("-" * 60)

    if not results:

        print(
            "\nNo evaluation cases "
            "completed successfully."
        )

        return

    # -----------------------------------------
    # Calculate aggregate metrics
    # -----------------------------------------

    answer_accuracy = calculate_rate(
        results,
        "answer_match",
    )

    retrieval_hit_rate = calculate_rate(
        results,
        "retrieval_hit",
    )

    top1_accuracy = calculate_rate(
        results,
        "top1_hit",
    )

    citation_accuracy = calculate_rate(
        results,
        "citation_match",
    )

    abstention_accuracy = calculate_rate(
        results,
        "abstention_match",
    )

    answerable_cases = count_applicable(
        results,
        "answer_match",
    )

    unanswerable_cases = count_applicable(
        results,
        "abstention_match",
    )

    # -----------------------------------------
    # Print evaluation summary
    # -----------------------------------------

    print("\nEQUITYAI RAG EVALUATION")
    print("=" * 60)

    print(
        f"Cases completed: "
        f"{len(results)}/{len(dataset)}"
    )

    print(
        f"Evaluation failures: "
        f"{failed_cases}"
    )

    print(
        f"Answerable cases: "
        f"{answerable_cases}"
    )

    print(
        f"Unanswerable cases: "
        f"{unanswerable_cases}"
    )

    print("-" * 60)

    print(
        f"Answer accuracy: "
        f"{answer_accuracy:.1f}%"
    )

    print(
        f"Retrieval hit rate: "
        f"{retrieval_hit_rate:.1f}%"
    )

    print(
        f"Top-1 retrieval accuracy: "
        f"{top1_accuracy:.1f}%"
    )

    print(
        f"Citation accuracy: "
        f"{citation_accuracy:.1f}%"
    )

    print(
        f"Abstention accuracy: "
        f"{abstention_accuracy:.1f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()