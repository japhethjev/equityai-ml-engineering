import re


def normalize_text(text: str) -> str:
    """Lowercase text and remove most punctuation for comparison."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def keyword_overlap_score(query: str, content: str) -> float:
    """
    Measure how many important query words appear in the content.
    """

    query_words = set(normalize_text(query).split())
    content_words = set(normalize_text(content).split())

    if not query_words:
        return 0.0

    overlap = query_words.intersection(content_words)

    return len(overlap) / len(query_words)


def exact_phrase_bonus(query: str, content: str) -> float:
    """
    Add extra score when important phrases appear exactly.
    """

    query_normalized = normalize_text(query)
    content_normalized = normalize_text(content)

    bonus = 0.0

    important_phrases = [
        "offer price",
        "ipo offer price",
        "gross proceeds",
        "net proceeds",
        "market cap",
        "price book",
        "net margin",
    ]

    for phrase in important_phrases:
        if phrase in query_normalized and phrase in content_normalized:
            bonus += 0.4

    return bonus


def rerank_results(
    query: str,
    results: list[dict],
) -> list[dict]:
    """
    Rerank retrieved chunks using:
    - semantic similarity
    - keyword overlap
    - exact phrase matching
    """

    reranked = []

    for result in results:
        semantic_score = float(
            result.get(
                "semantic_similarity",
                result.get("similarity", 0.0),
            )
        )

        keyword_score = keyword_overlap_score(
            query,
            result["content"],
        )

        phrase_bonus = exact_phrase_bonus(
            query,
            result["content"],
        )

        final_score = (
            0.55 * semantic_score
            + 0.30 * keyword_score
            + 0.15 * phrase_bonus
        )

        updated_result = result.copy()
        updated_result["keyword_overlap"] = keyword_score
        updated_result["phrase_bonus"] = phrase_bonus
        updated_result["final_score"] = final_score

        reranked.append(updated_result)

    return sorted(
        reranked,
        key=lambda x: x["final_score"],
        reverse=True,
    )