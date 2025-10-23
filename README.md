# 🧠 Patent Pipeline

A modular **OCR → LLM extraction pipeline** for historical patent documents.
It turns raw scanned PDFs into structured JSONL metadata using Tesseract-based OCR and a local (or cloud) LLM.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Folders](#data-folders)
- [Configuration](#configuration)
- [Run the Pipeline](#run-the-pipeline)
- [Limiting to a Few Documents](#limiting-to-a-few-documents)
- [Models & Backends](#models--backends)
- [Output Format](#output-format)
- [Prompting Notes](#prompting-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project automates the extraction of bibliographic fields from patent PDFs:

- **identifier** (e.g., `CH-117732`)
- **title** (keep original language)
- **inventors** (string; multiple separated by `; `)
- **assignee**
- **pub_date_application**, **pub_date_publication**, **pub_date_foreign** (ISO `YYYY-MM-DD` or `null`)
- **address** (English, e.g., `Kiruna (Sweden)`)
- **industrial_field** (short English category)

**Pipeline steps:**

1. **OCR**: Convert PDFs to text via `pdf2image` + `pytesseract` (+ OpenCV preprocessing).
2. **Extraction**: Feed OCR text to an LLM (HF/Transformers or MLX on Apple Silicon) with a strict JSON prompt.
3. **Validation**: Parse and validate with Pydantic models.
4. **Output**: Save one JSON object per document (JSON Lines).

---

## Project Structure

```
patent_pipeline/
├── data/
│   ├── raw_pdf/           # input PDFs
│   ├── ocr_text/          # OCR’d .txt
│   ├── predictions/       # JSONL predictions
│   └── ocr_report.csv     # OCR run report
├── src/patent_pipeline/
│   ├── patent_ocr/
│   │   └── ocr_utils.py               # OCR batch + preprocessing
│   ├── pydantic/
│   │   ├── hf_agent.py                # model loader (HF/MLX) + JSON extraction
│   │   ├── models.py                  # PatentMetadata / PatentExtraction
│   │   └── prompt_templates.py        # extraction prompt(s)
│   ├── utils/
│   │   └── device_utils.py            # device detection/logging
│   ├── pipeline.py                    # end-to-end runner (OCR → LLM → JSONL)
│   └── __main__.py                    # allows `python -m patent_pipeline`
├── scripts/
│   ├── run_local.sh                   # local run (Mac / MPS / MLX)
│   └── run_azure.sh                   # GPU VM run (CUDA)
├── pyproject.toml                     # Poetry deps
└── README.md
```

---

## Requirements

- **Python** ≥ 3.12 (Poetry-managed)
- **macOS (Apple Silicon)** or Linux
- OCR utilities:
  - **Tesseract**
  - **Poppler**
- Optional (Apple Silicon):
  - **MLX** backend (`mlx-lm`)

---

## Installation

```bash
git clone https://github.com/Just-PH/patent_pipeline.git
cd patent_pipeline

poetry install

# OCR deps
brew install tesseract poppler
# or
sudo apt install tesseract-ocr poppler-utils

poetry add sentencepiece
poetry add mlx-lm  # if using MLX backend
```

---

## Run the Pipeline

```bash
export HF_MODEL="mlx-community/Mistral-7B-Instruct-v0.3-8bit" # You can change the mode
export DEVICE="mps" # You can chance the device
bash scripts/run_local.sh
```

---

## License

MIT © 2025 — Pierre-Henri Delville
