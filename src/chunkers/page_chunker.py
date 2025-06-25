"""Chunkers for splitting documents."""

from typing import List


class PageChunker:
    """Split extracted text into page-based chunks."""

    def chunk(self, text: str) -> List[str]:
        """Return a list of page chunks using form feed delimiters."""
        return [page.strip() for page in text.split("\f") if page.strip()]
