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
from app.rag.vector_store import get_connection


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

    logger.info(
        "Received document upload: %s",
        filename,
    )

    temp_path = None

    try:
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = temp_file.name

        # Run ingestion and capture statistics
        result = ingest_document(
            temp_path,
            document_name=filename,
        )

        logger.info(
            "Document ingested successfully: %s",
            filename,
        )

        # Return ingestion statistics
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