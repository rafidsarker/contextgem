"""Utilities for building Pydantic models from JSON Schema."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel, create_model

JSON_TYPE_MAPPING = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def extraction_schema_to_json(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a custom extraction YAML schema into the output JSON Schema."""

    document_type = schema.get("documentTypes", [{}])[0]
    title = document_type.get("name", "ExtractionResult")

    field_names: List[str] = []
    for extractor in document_type.get("extractors", []):
        if extractor.get("field"):
            field_names.append(extractor["field"])

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": title,
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "ordinal": {"type": "integer"},
                "field": {"type": "string", "enum": field_names},
                "extracted_value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "array"},
                        {"type": "object"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
                "confidence": {"type": "number"},
                "justification": {"type": "string"},
                "location": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "bbox": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "width": {"type": "number"},
                                "height": {"type": "number"},
                            },
                            "required": ["x", "y", "width", "height"],
                        },
                    },
                    "required": ["page", "bbox"],
                },
            },
            "required": [
                "field",
                "extracted_value",
                "confidence",
                "justification",
                "location",
            ],
        },
    }


def _schema_to_type(name: str, schema: Dict[str, Any]) -> Any:
    """Recursively convert JSON Schema to a Python type."""

    if "oneOf" in schema:
        # Simplify ``oneOf`` as ``Any`` for validation purposes.
        return Any

    schema_type = schema.get("type", "string")

    if schema_type == "object":
        fields: Dict[str, Tuple[Any, Any]] = {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for prop_name, prop_schema in properties.items():
            field_type = _schema_to_type(prop_name.capitalize(), prop_schema)
            default = ... if prop_name in required else None
            fields[prop_name] = (field_type, default)
        return create_model(name.capitalize(), **fields)

    if schema_type == "array":
        item_type = _schema_to_type(f"{name}Item", schema.get("items", {}))
        return List[item_type]

    return JSON_TYPE_MAPPING.get(schema_type, Any)


def build_model_from_schema(schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from a JSON Schema dictionary."""

    root_type = _schema_to_type(schema.get("title", "Root"), schema)

    if isinstance(root_type, type) and issubclass(root_type, BaseModel):
        return root_type

    return create_model(schema.get("title", "Root"), __root__=(root_type, ...))
