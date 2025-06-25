"""Service wrapper for Google Gemini classification calls."""

from __future__ import annotations

from typing import Any

import google.generativeai as genai


def classify_json(model: str, prompt: str, **kwargs: Any) -> str:
    """Call Gemini to classify JSON content."""
    return genai.generate_text(model=model, prompt=prompt, **kwargs).text
