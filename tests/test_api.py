from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


@patch("app.api.main.get_connection")
def test_health(mock_get_connection):
    """
    The health endpoint should report that both
    the API and database are operational.
    """

    mock_connection = mock_get_connection.return_value

    mock_cursor = (
        mock_connection
        .cursor
        .return_value
        .__enter__
        .return_value
    )

    mock_cursor.fetchone.return_value = (1,)

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "api": "up",
        "database": "up",
    }

    mock_cursor.execute.assert_called_once_with(
        "SELECT 1;"
    )

    mock_connection.close.assert_called_once()


def test_empty_question():
    """
    An empty question should be rejected.
    """

    response = client.post(
        "/ask",
        json={
            "question": "   "
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Question cannot be empty."
    }


@patch("app.api.main.answer_question")
def test_valid_question(mock_answer_question):
    """
    A valid question should return a successful
    RAG response.

    The RAG service is mocked so this test does
    not call OpenAI or PostgreSQL.
    """

    mock_answer_question.return_value = {
        "answer": "The IPO offer price is ₦525.00.",
        "sources": [
            {
                "document": "dangote_refinery_valuation.pdf",
                "page": 1,
                "chunk": 1,
                "score": 0.525,
            }
        ],
    }

    response = client.post(
        "/ask",
        json={
            "question": "What is the IPO offer price?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["answer"]
        == "The IPO offer price is ₦525.00."
    )

    assert len(data["sources"]) == 1

    assert (
        data["sources"][0]["document"]
        == "dangote_refinery_valuation.pdf"
    )

    mock_answer_question.assert_called_once_with(
        "What is the IPO offer price?"
    )


@patch("app.api.main.list_documents")
def test_list_documents(mock_list_documents):
    """
    The documents endpoint should return all
    indexed documents and their metadata.
    """

    mock_list_documents.return_value = [
        {
            "document": "dangote_refinery_valuation.pdf",
            "pages": 2,
            "chunks": 9,
        },
        {
            "document": "equityai_test_company.pdf",
            "pages": 1,
            "chunks": 1,
        },
    ]

    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert data["total_documents"] == 2

    assert len(data["documents"]) == 2

    assert (
        data["documents"][0]["document"]
        == "dangote_refinery_valuation.pdf"
    )

    assert data["documents"][0]["pages"] == 2
    assert data["documents"][0]["chunks"] == 9

    mock_list_documents.assert_called_once_with()


@patch("app.api.main.delete_document")
def test_delete_document(mock_delete_document):
    """
    Deleting an existing document should return
    the number of deleted chunks.
    """

    mock_delete_document.return_value = 1

    response = client.delete(
        "/documents/equityai_test_company.pdf"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "success",
        "document": "equityai_test_company.pdf",
        "deleted_chunks": 1,
    }

    mock_delete_document.assert_called_once_with(
        "equityai_test_company.pdf"
    )


@patch("app.api.main.delete_document")
def test_delete_missing_document(
    mock_delete_document,
):
    """
    Deleting a document that does not exist
    should return HTTP 404.
    """

    mock_delete_document.return_value = 0

    response = client.delete(
        "/documents/missing_document.pdf"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Document not found."
    }

    mock_delete_document.assert_called_once_with(
        "missing_document.pdf"
    )