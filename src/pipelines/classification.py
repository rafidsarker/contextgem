"""Classification pipeline using Gemini with retrieved context."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from ..model_builder import build_model_from_schema
from ..pipelines.retrieval import RetrievalPipeline
from ..prompt_assembler import build_classification_prompt
from ..services.classification_service import classify_json


class ClassificationPipeline:
    """Run schema-driven classification of extracted data."""

    def __init__(
        self, retrieval: RetrievalPipeline | None = None, model: str = "gemini-pro"
    ) -> None:
        self.retrieval = retrieval
        self.model = model

    def run(self, extracted: List[Dict[str, Any]], schema: Dict[str, Any]) -> BaseModel:
        """Return validated classification results."""
        field_map = {item["field"]: item["extracted_value"] for item in extracted}

        chunks = []
        if self.retrieval:
            chunks = self.retrieval.get_relevant_chunks(
                " ".join(map(str, field_map.values()))
            )

        prompt = build_classification_prompt(chunks, field_map, schema)
        response = classify_json(self.model, prompt)
        model_cls = build_model_from_schema(schema)
        return model_cls.model_validate_json(response)
