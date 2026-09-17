import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client only when it is needed.

    This avoids requiring OpenAI credentials merely
    to import the module, which allows unit tests and
    CI pipelines to run without external API access.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    return OpenAI(api_key=api_key)


def embed_text(text: str) -> list[float]:
    """
    Convert text into an embedding vector.
    """

    if not text.strip():
        raise ValueError(
            "Text cannot be empty"
        )

    client = get_openai_client()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding