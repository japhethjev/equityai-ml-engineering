import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: str) -> list[float]:
    """Convert text into an embedding vector."""

    if not text.strip():
        raise ValueError("Text cannot be empty")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding