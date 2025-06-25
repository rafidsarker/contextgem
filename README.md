# Identifi Refactor

This example project demonstrates how to refactor a PDF extraction tool using patterns from [ContextGem](https://github.com/shcherbak-ai/contextgem).

## Installation

```bash
pip install contextgem google-generativeai faiss-cpu pdfminer.six openai
```

## Running Extraction

```bash
python run_extraction.py ./sample.pdf config/extractionSchema.yaml output.json
```

The resulting ``output.json`` contains an array of extraction results with
confidence scores and page locations.

## Running Classification

```bash
python run_classification.py output.json config/classificationOutputSchema.yaml classified.json --pdf ./sample.pdf
```

Both commands accept `--model` to specify the Gemini model ID.
