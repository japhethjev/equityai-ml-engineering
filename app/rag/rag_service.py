import os

from openai import OpenAI

from app.rag.embedding_service import embed_text
from app.rag.vector_store import hybrid_search
from app.rag.reranker import rerank_results


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client only when answer generation
    actually requires one.

    This allows application modules and automated tests
    to load without requiring OpenAI credentials.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    return OpenAI(api_key=api_key)


def answer_question(query: str) -> dict:
    """
    Answer a user question using retrieved
    document evidence only.
    """

    query_embedding = embed_text(query)

    retrieved = hybrid_search(
        query=query,
        query_embedding=query_embedding,
        limit=5,
    )

    ranked = rerank_results(
        query,
        retrieved,
    )

    top_results = ranked[:3]

    context = "\n\n".join(
        [
            (
                f"Document: {r['document']}\n"
                f"Page: {r['page']}\n"
                f"Content: {r['content']}"
            )
            for r in top_results
        ]
    )

    prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the evidence provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or numbers.
3. If the evidence does not contain the answer, say:
   "I could not find sufficient evidence in the provided documents."
4. Give a concise answer.
5. State the document and page used as the source.

USER QUESTION:
{query}

EVIDENCE:
{context}
"""

    # Create OpenAI client only when generation is required.
    client = get_openai_client()

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    return {
        "answer": response.output_text,
        "sources": [
            {
                "document": r["document"],
                "page": r["page"],
                "chunk": r["chunk"],
                "score": r["final_score"],
            }
            for r in top_results
        ],
    }