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

    # Create a mock database connection
    mock_connection = mock_get_connection.return_value

    # Create a mock database cursor
    mock_cursor = (
        mock_connection
        .cursor
        .return_value
        .__enter__
        .return_value
    )

    mock_cursor.fetchone.return_value = (1,)

    # Call the health endpoint
    response = client.get("/health")

    # Check HTTP response
    assert response.status_code == 200

    # Check returned health information
    assert response.json() == {
        "status": "healthy",
        "api": "up",
        "database": "up",
    }

    # Confirm that the database was actually checked
    mock_cursor.execute.assert_called_once_with(
        "SELECT 1;"
    )

    # Confirm connection was closed
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