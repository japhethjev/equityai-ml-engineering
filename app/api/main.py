import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel

from app.rag.ingest import ingest_document
from app.rag.rag_service import answer_question
from app.rag.vector_store import (
    delete_document,
    get_connection,
    list_documents,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("equityai-api")


app = FastAPI(
    title="EquityAI RAG API",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


# =========================================================
# HEALTH CHECK
# =========================================================


@app.get("/health")
def health():
    """
    Check that both the API and PostgreSQL database
    are operational.
    """

    try:
        connection = get_connection()

        with connection.cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()

        connection.close()

        return {
            "status": "healthy",
            "api": "up",
            "database": "up",
        }

    except Exception:
        logger.exception(
            "Database health check failed"
        )

        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "api": "up",
                "database": "down",
            },
        )


# =========================================================
# RAG QUESTION ENDPOINT
# =========================================================


@app.post("/ask")
def ask(request: QuestionRequest):
    """
    Ask a question using the EquityAI RAG pipeline.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    logger.info("Received question")

    try:
        result = answer_question(question)

        logger.info(
            "Question answered successfully"
        )

        return result

    except Exception:
        logger.exception(
            "Failed to answer question"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is "
                "temporarily unavailable."
            ),
        )


# =========================================================
# DOCUMENT UPLOAD
# =========================================================


@app.post("/documents/upload")
def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload and ingest a PDF document.

    The document is temporarily saved,
    processed by the ingestion pipeline,
    and then removed.

    Existing chunks are detected before
    embeddings are generated.
    """

    filename = file.filename or ""

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename.",
        )

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # Strip directory components from the supplied filename.
    filename = Path(filename).name

    logger.info(
        "Received document upload: %s",
        filename,
    )

    temp_path = None

    try:
        # ---------------------------------------------
        # Save uploaded PDF temporarily
        # ---------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = temp_file.name

        # ---------------------------------------------
        # Run ingestion and capture statistics
        # ---------------------------------------------

        result = ingest_document(
            temp_path,
            document_name=filename,
        )

        logger.info(
            "Document ingested successfully: %s",
            filename,
        )

        # ---------------------------------------------
        # Return ingestion statistics
        # ---------------------------------------------

        return {
            "status": "success",
            "document": result["document"],
            "total_chunks": result["total_chunks"],
            "new_chunks": result["new_chunks"],
            "existing_chunks": result["existing_chunks"],
            "inserted": result["inserted"],
            "skipped": result["skipped"],
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Document ingestion failed: %s",
            filename,
        )

        raise HTTPException(
            status_code=503,
            detail="Document ingestion failed.",
        )

    finally:
        file.file.close()

        if temp_path:
            temp_file_path = Path(temp_path)

            if temp_file_path.exists():
                temp_file_path.unlink()


# =========================================================
# DOCUMENT INVENTORY
# =========================================================


@app.get("/documents")
def get_documents():
    """
    Return all documents currently indexed
    in the EquityAI knowledge base.
    """

    try:
        documents = list_documents()

        logger.info(
            "Retrieved document inventory: %s documents",
            len(documents),
        )

        return {
            "total_documents": len(documents),
            "documents": documents,
        }

    except Exception:
        logger.exception(
            "Failed to retrieve documents"
        )

        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve documents.",
        )


# =========================================================
# DOCUMENT DELETION
# =========================================================


@app.delete("/documents/{document_name}")
def remove_document(
    document_name: str,
):
    """
    Delete a document and all associated chunks
    and embeddings from the knowledge base.
    """

    # Strip directory components for safety.
    document_name = Path(document_name).name

    try:
        deleted_chunks = delete_document(
            document_name
        )

        if deleted_chunks == 0:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        logger.info(
            "Deleted document: %s (%s chunks)",
            document_name,
            deleted_chunks,
        )

        return {
            "status": "success",
            "document": document_name,
            "deleted_chunks": deleted_chunks,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to delete document: %s",
            document_name,
        )

        raise HTTPException(
            status_code=503,
            detail="Unable to delete document.",
        )