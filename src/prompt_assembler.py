"""Helpers to build prompts for Gemini."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def build_extraction_prompt(chunks: Iterable[str], schema: Dict[str, Any]) -> str:
    """Assemble extraction prompt from context chunks and schema."""
    joined = "\n".join(chunks)
    return (
        "Extract the following fields as JSON matching this schema:\n"
        f"{schema}\n\nContext:\n{joined}"
    )


def build_classification_prompt(
    chunks: Iterable[str], data: Dict[str, Any], schema: Dict[str, Any]
) -> str:
    """Assemble classification prompt from data and optional context."""
    context = "\n".join(chunks)
    return (
        "Classify the extracted data into allowed values as JSON matching this schema:\n"
        f"{schema}\n\nData:\n{data}\n\nContext:\n{context}"
    )
