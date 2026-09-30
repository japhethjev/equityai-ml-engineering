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

def select_balanced_evidence(
    ranked_results: list[dict],
    document_ids: list[str] | None = None,
    limit: int = 3,
) -> list[dict]:
    """
    Select final evidence while preserving document coverage.

    For multi-document retrieval, the highest-ranked available
    chunk from each requested document is selected first.
    Remaining slots are then filled by global reranking order.

    Single-document and unrestricted retrieval preserve the
    existing top-N behaviour.
    """

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    if not ranked_results:
        return []

    if not document_ids or len(document_ids) <= 1:
        return ranked_results[:limit]

    # Preserve requested document order while removing duplicates.
    requested_ids = list(
        dict.fromkeys(
            str(document_id)
            for document_id in document_ids
        )
    )

    selected = []
    selected_positions = set()

    # First pass: best available chunk from each document.
    for document_id in requested_ids:
        if len(selected) >= limit:
            break

        for position, result in enumerate(
            ranked_results
        ):
            result_document_id = result.get(
                "document_id"
            )

            if (
                result_document_id is not None
                and str(result_document_id)
                == document_id
            ):
                selected.append(result)
                selected_positions.add(position)
                break

    # Second pass: fill remaining capacity using global rank.
    for position, result in enumerate(
        ranked_results
    ):
        if len(selected) >= limit:
            break

        if position in selected_positions:
            continue

        selected.append(result)
        selected_positions.add(position)

    return selected
