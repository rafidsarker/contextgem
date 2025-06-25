"""Wrapper around OpenAI embedding models."""

from typing import Iterable, List

import openai


class OpenAIEmbedder:
    """Embed text chunks using an OpenAI embedding model."""

    def __init__(self, model: str = "text-embedding-ada-002") -> None:
        self.model = model

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Return embeddings for a list of texts."""
        response = openai.embeddings.create(model=self.model, input=list(texts))
        return [record.embedding for record in response.data]
