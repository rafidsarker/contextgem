"""PDF loading utilities."""

from pathlib import Path


class PDFLoader:
    """Load PDF files from disk and return raw bytes and extracted text."""

    def load_bytes(self, path: Path) -> bytes:
        """Return the raw bytes from the given PDF path."""
        with path.open("rb") as fh:
            return fh.read()

    def load_text(self, path: Path) -> str:
        """Extract text from the PDF using ``pdfminer.six``."""
        from pdfminer.high_level import extract_text

        return extract_text(str(path))
