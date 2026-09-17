import sys
from pathlib import Path

from app.rag.document_loader import load_pdf
from app.rag.text_chunker import chunk_pages
from app.rag.embedding_service import embed_text
from app.rag.vector_store import (
    chunk_exists,
    store_chunks,
)


def ingest_document(
    file_path: str,
    document_name: str | None = None,
) -> dict:
    """
    Ingest a PDF into the EquityAI vector database.

    Workflow:
    1. Load PDF pages.
    2. Preserve the original document filename.
    3. Split pages into chunks.
    4. Check which chunks already exist.
    5. Generate embeddings only for new chunks.
    6. Store new chunks in PostgreSQL/pgvector.

    Returns ingestion statistics.
    """

    print(f"Loading document: {file_path}")

    pages = load_pdf(file_path)

    print(f"Loaded {len(pages)} pages")

    # -----------------------------------------
    # Preserve original document filename
    # -----------------------------------------

    if document_name:
        safe_document_name = Path(
            document_name
        ).name

        for page in pages:
            page["document"] = safe_document_name

    chunks = chunk_pages(pages)

    print(f"Created {len(chunks)} chunks")

    # -----------------------------------------
    # Check for duplicates before embedding
    # -----------------------------------------

    print("Checking for existing chunks...")

    new_chunks = []
    existing_count = 0

    for chunk in chunks:

        exists = chunk_exists(
            document=chunk["document"],
            page=chunk["page"],
            chunk=chunk["chunk"],
        )

        if exists:
            existing_count += 1
        else:
            new_chunks.append(chunk)

    print(f"New chunks: {len(new_chunks)}")
    print(f"Existing chunks: {existing_count}")

    # -----------------------------------------
    # Stop if everything already exists
    # -----------------------------------------

    if not new_chunks:
        print("No new chunks to embed.")
        print("Ingestion complete.")

        return {
            "document": (
                Path(document_name).name
                if document_name
                else Path(file_path).name
            ),
            "total_chunks": len(chunks),
            "new_chunks": 0,
            "existing_chunks": existing_count,
            "inserted": 0,
            "skipped": existing_count,
        }

    # -----------------------------------------
    # Generate embeddings for new chunks only
    # -----------------------------------------

    print("Creating embeddings...")

    embeddings = []

    for index, chunk in enumerate(
        new_chunks,
        start=1,
    ):

        print(
            f"Embedding chunk "
            f"{index}/{len(new_chunks)}"
        )

        embedding = embed_text(
            chunk["text"]
        )

        embeddings.append(embedding)

    # -----------------------------------------
    # Store new chunks
    # -----------------------------------------

    print(
        "Storing new chunks in PostgreSQL..."
    )

    result = store_chunks(
        new_chunks,
        embeddings,
    )

    print("Ingestion complete.")
    print(
        f"Inserted: {result['inserted']}"
    )
    print(
        "Skipped duplicates during insert: "
        f"{result['skipped']}"
    )

    return {
        "document": (
            Path(document_name).name
            if document_name
            else Path(file_path).name
        ),
        "total_chunks": len(chunks),
        "new_chunks": len(new_chunks),
        "existing_chunks": existing_count,
        "inserted": result["inserted"],
        "skipped": (
            existing_count
            + result["skipped"]
        ),
    }


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python -m app.rag.ingest "
            "<path-to-pdf>"
        )
        sys.exit(1)

    ingest_document(
        sys.argv[1]
    )