"""FAISS vector store for similarity search."""

from __future__ import annotations

from typing import Iterable, List

import faiss
import numpy as np


class FaissVectorStore:
    """In-memory FAISS index for dense vector similarity."""

    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: List[str] = []

    def add(self, embeddings: Iterable[List[float]], chunks: Iterable[str]) -> None:
        arr = np.array(list(embeddings)).astype("float32")
        self.index.add(arr)
        self.chunks.extend(list(chunks))

    def search(self, embedding: List[float], k: int = 5) -> List[str]:
        query = np.array(embedding, dtype="float32")[None, :]
        distances, indices = self.index.search(query, k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
