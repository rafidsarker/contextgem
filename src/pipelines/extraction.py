"""Extraction pipeline leveraging retrieved context and Gemini."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, create_model

from ..model_builder import build_model_from_schema, extraction_schema_to_json
from ..pipelines.retrieval import RetrievalPipeline
from ..prompt_assembler import build_extraction_prompt
from ..services.extraction_service import extract_json


class ExtractionPipeline:
    """Run schema-driven extraction against a PDF."""

    def __init__(self, retrieval: RetrievalPipeline, model: str = "gemini-pro") -> None:
        self.retrieval = retrieval
        self.model = model

    def run(self, pdf_path: Path, schema: Dict[str, Any]) -> BaseModel:
        """Return validated extraction results."""
        self.retrieval.index_document(pdf_path)
        json_schema = extraction_schema_to_json(schema)
        prompt = build_extraction_prompt(self.retrieval.store.chunks, json_schema)
        response = extract_json(self.model, prompt)
        model_cls = build_model_from_schema(json_schema)
        return model_cls.model_validate_json(response)
