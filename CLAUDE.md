# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Patent Pipeline is a modular system for automated patent data extraction from historical patent archives (Swiss, French, German). The flow is: **PDF/image -> OCR -> LLM -> validated JSON (Pydantic)**.

## Build & Install

```bash
poetry install                    # core deps
poetry install --with dev         # + notebooks, matplotlib, streamlit
poetry install --with doctr       # + DocTR OCR backend
poetry install --with lightonocr  # + LightOnOCR backend
poetry install --with surya       # + Surya OCR backend
```

`torch` is intentionally excluded from main deps (installed explicitly in Docker with CUDA). `numpy` must stay <2.0 for PaddlePaddle/OpenCV compatibility.

## Running

```bash
# Full pipeline (OCR + LLM extraction)
poetry run python -m patent_pipeline --raw_dir data/raw_pdf --backend doctr --workers 4

# OCR only (skip LLM)
poetry run python -m patent_pipeline --raw_dir data/raw_pdf --skip_extraction

# Standalone vLLM extraction (separate package)
poetry run python scripts/run_patent_extraction.py --profile de_legacy_v4 --texts-dir data/ocr_text
```

## Testing

```bash
poetry run pytest                           # all tests
poetry run pytest tests/test_guardrails.py  # single file
poetry run pytest -k "test_name"            # single test by name
```

Tests use `sys.path` manipulation in `tests/__init__.py` to import from `src/`.

## Architecture

There are two separate packages under `src/`:

### `patent_pipeline` — the original OCR + HF/MLX extraction pipeline
- **`__main__.py`** — CLI entry point, parses args, calls `run_pipeline()`
- **`pipeline.py`** — orchestrates OCR -> LLM -> validation; loads backend by key via dynamic import (`_import_symbol`)
- **`patent_ocr/pipeline_modular.py`** — `Pipeline_OCR` class: load image -> deskew -> segment -> OCR -> write text. Core protocols: `OcrBackend` (must implement `run_blocks_ocr(block_imgs, ocr_config) -> list[str]`) and `Segmenter`
- **`patent_ocr/backends/`** — pluggable OCR backends: `tesseract`, `tesserocr`, `doctr`, `gotocr`, `paddle`, `lightonocr`, `glmocr_vllm`. Each backend is resolved by key in `pipeline.py:_default_backend_spec()`
- **`patent_ocr/custom_segmentation.py`** — layout segmentation for block-level OCR
- **`pydantic_extraction/patent_extractor.py`** — `PatentExtractor` wraps HF/MLX model loading, prompt rendering, JSON extraction, and Pydantic validation
- **`pydantic_extraction/models.py`** — `PatentMetadata`, `PatentExtraction`, `Inventor`, `Assignee` Pydantic models

### `patent_extraction` — standalone vLLM-only extraction package (newer)
- **`extractor.py`** — `PatentExtractor` and `PatentExtractionRunner` classes for vLLM-based batch extraction
- **`config.py`** — frozen dataclasses: `ExtractionConfig`, `VLLMConfig`, `StrategyConfig`, `ProfileConfig`
- **`profiles.py`** — profile system: loads JSON profile definitions from `profile_defs/` with prompt templates from `prompts/`
- **`strategies.py`** — extraction strategies: `baseline`, `chunked`, `header_first`, `two_pass_targeted`, `self_consistency`. Each strategy controls how OCR text is split, how many LLM passes run, and how results are merged
- **`cli/run_extract.py`** — CLI for standalone extraction runs

### Key design patterns

- **Two segmentation modes**: `custom` (pipeline segments into blocks, backend OCRs each block) vs `backend` (backend receives full page, handles its own layout detection)
- **GPU backends**: when `is_gpu=True`, the pipeline forces `workers=1` and `parallel=none`; throughput scales via `ocr_config.batch_size` instead
- **Extraction profiles**: JSON files in `patent_extraction/profile_defs/` bundle model, prompt, vLLM params, and strategy config into reusable presets

## Docker Images

- `Dockerfile` / `Dockerfile.lightonocr` — OCR-capable images
- `Dockerfile.slm-mistral31` — runtime-only SLM image (no OCR stack, no compiler)
- `Dockerfile.slm-mistral31-static` — adds compiler toolchain for `cache_implementation=static`
- `Dockerfile.slm-mistral31-vllm` — vLLM-based image for throughput benchmarks
- `Dockerfile.patent-extraction-vllm` — standalone extraction image

## Environment Variables

- `HF_MODEL` — HuggingFace model name (default: `mlx-community/Mistral-7B-Instruct-v0.3`)
- `USE_MLX` — force Apple MLX backend (`True` if available)
- `TESSERACT_LANGS` — OCR languages (default: `frk+deu+eng+fra+ita`)
- `VLLM_ENFORCE_EAGER=1` — workaround for CUDA graph capture errors in vLLM
