"""Pipeline for indexing documents and retrieving relevant chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ..chunkers.page_chunker import PageChunker
from ..embedders.openai_embedder import OpenAIEmbedder
from ..loaders.pdf_loader import PDFLoader
from ..store.faiss_store import FaissVectorStore


class RetrievalPipeline:
    """Index documents and retrieve relevant chunks via embeddings."""

    def __init__(
        self,
        loader: PDFLoader,
        chunker: PageChunker,
        embedder: OpenAIEmbedder,
        store: FaissVectorStore,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def index_document(self, path: Path) -> bytes:
        """Load the PDF, chunk its text, embed the chunks, and store them."""
        text = self.loader.load_text(path)
        chunks = self.chunker.chunk(text)
        embeddings = self.embedder.embed(chunks)
        self.store.add(embeddings, chunks)
        return self.loader.load_bytes(path)

    def get_relevant_chunks(self, query: str, k: int = 5) -> List[str]:
        """Retrieve the top ``k`` most similar chunks for the query."""
        embedding = self.embedder.embed([query])[0]
        return self.store.search(embedding, k)
