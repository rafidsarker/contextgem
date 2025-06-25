"""CLI entry point for running the extraction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chunkers.page_chunker import PageChunker
from src.embedders.openai_embedder import OpenAIEmbedder
from src.loaders.pdf_loader import PDFLoader
from src.pipelines.extraction import ExtractionPipeline
from src.pipelines.retrieval import RetrievalPipeline
from src.store.faiss_store import FaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction pipeline")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument("schema", type=Path, help="Path to extraction schema YAML")
    parser.add_argument("output", type=Path, help="Path to write JSON output")
    parser.add_argument("--model", default="gemini-pro", help="Gemini model id")
    args = parser.parse_args()

    loader = PDFLoader()
    chunker = PageChunker()
    embedder = OpenAIEmbedder()
    store = FaissVectorStore(dim=1536)
    retrieval = RetrievalPipeline(loader, chunker, embedder, store)
    pipeline = ExtractionPipeline(retrieval, model=args.model)

    import yaml

    schema_dict = yaml.safe_load(args.schema.read_text())

    result = pipeline.run(args.pdf, schema_dict)
    args.output.write_text(result.json())


if __name__ == "__main__":
    main()
