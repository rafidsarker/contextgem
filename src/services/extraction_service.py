"""Service wrapper for Google Gemini extraction calls."""

from __future__ import annotations

from typing import Any, Dict

import google.generativeai as genai


def extract_json(model: str, prompt: str, **kwargs: Any) -> str:
    """Call Gemini to generate JSON content."""
    return genai.generate_text(model=model, prompt=prompt, **kwargs).text
