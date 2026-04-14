# 🧠 Patent Pipeline

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/packaging-poetry-60A5FA.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Doctr](https://img.shields.io/badge/OCR-Doctr%20%2B%20Tesseract-orange.svg)](https://mindee.github.io/doctr)
[![LLM](https://img.shields.io/badge/LLM-Mistral%2FMLX-9cf.svg)](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3)

---

## 📘 Overview

**Patent Pipeline** is a modular and production-ready system for **automated patent data extraction**.
It combines **OCR (Tesseract + Doctr)**, **LLM inference (Hugging Face / MLX)**, and **Pydantic validation** to transform raw patent PDFs into structured and reliable metadata.

The goal is to make historical patent archives (Swiss, French, German, etc.) searchable and analyzable through a fully automated pipeline: **PDF → OCR → LLM → validated JSON.**

---

## 🚀 Key Features

* 🔍 **Hybrid OCR engine**:

  * `Tesseract` for robust multilingual extraction
  * `Doctr` for high-precision text recognition on complex scans
* ⚙️ **Image preprocessing**: deskewing, adaptive thresholding, denoising
* 🧩 **LLM extraction**: Mistral / Mixtral models (Hugging Face or MLX for Apple Silicon)
* 🧠 **Typed validation** with Pydantic (`PatentMetadata`, `PatentExtraction`)
* ⚡️ **Parallel processing** via `concurrent.futures`
* 📦 **Structured outputs**: `.jsonl` and `.csv`
* ☁️ **Local or cloud execution** (Azure-ready scripts)

---

## ⚙️ Installation

### Prerequisites

* **Python** ≥ 3.12 (tested up to 3.14)
* **[Poetry](https://python-poetry.org/)** for dependency management
* **Tesseract OCR** and **Doctr** installed on your system

#### macOS example

```bash
brew install tesseract
```

#### Ubuntu / Debian

```bash
sudo apt update && sudo apt install tesseract-ocr libtesseract-dev poppler-utils -y
```

### Clone and install dependencies

```bash
git clone https://github.com/Just-PH/patent_pipeline.git
cd patent_pipeline
poetry install
```

This installs all runtime dependencies defined in the `pyproject.toml`, including:

* **Core libraries**: `transformers`, `pydantic`, `torch`, `tqdm`, `opencv-python`
* **OCR stack**: `pytesseract`, `python-doctr[torch]`, `pdf2image`, `opencv-python-headless`
* **LLM backends**: `mlx-lm`, `accelerate`, `sentencepiece`
* **Utility libs**: `loguru`, `langdetect`, `regexp`

> 💡 You can check your Poetry environment with:
>
> ```bash
> poetry env info
> ```

### Optional: development setup

If you want to use Jupyter notebooks or local debugging:

```bash
poetry install --with dev
```

---

## 🧩 Usage

### Run the pipeline locally

```bash
poetry run python -m patent_pipeline \
  --csv data/ocr_report.csv \
  --out data/predictions \
  --tar data/raw_pdf/french-patents.tar \
  --workers 4
```

> ⚡️ Make sure your OCR text files and PDFs are located in `data/ocr_text/` and `data/raw_pdf/`.

### Example output

```json
{
  "publication_number": "CH-117732-A",
  "patentee": [
    {"name": "Emil Fredrik Schedwin", "location": "Kiruna"}
  ],
  "title": "Schneidevorrichtung für Käse, Butter usw.",
  "year": 1926,
  "language": "de",
  "class": "15b"
}
```

---

## 🧠 Internal Architecture

### 1. `patent_ocr/ocr_utils.py`

* Converts PDF → images
* Deskewing (`deskew_image`), binarization, and multi-language OCR
* Doctr-based OCR (`ocr_doctr`) for layout-sensitive text recognition

### 2. `pydantic/hf_agent.py`

* Interface to Hugging Face / MLX models (Mistral, Mixtral, etc.)
* Converts OCR text → structured JSON via prompts
* Automatic device detection and fallback using `utils/device_utils.py`

### 3. `pydantic/models.py`

* Defines strict Pydantic schemas for extracted fields
* Ensures output consistency and typing safety

### 4. `pipeline.py`

* Orchestrates the full OCR → LLM → validation workflow
* Handles multiprocessing and progress tracking (`tqdm`, `ProcessPoolExecutor`)

---

## 🔧 Environment Variables

| Variable          | Description             | Default                                  |
| ----------------- | ----------------------- | ---------------------------------------- |
| `HF_MODEL`        | Hugging Face model name | `mlx-community/Mistral-7B-Instruct-v0.3` |
| `USE_MLX`         | Force Apple MLX backend | `True` if available                      |
| `TESSERACT_LANGS` | Languages used for OCR  | `frk+deu+eng+fra+ita`                    |

---

## ☁️ Cloud Execution

```bash
bash scripts/run_azure.sh
```

> Configures remote storage, logs, and parallel workers for large-scale runs.

---

## 🐳 Docker Image: SLM Mistral 3.1 24B

The repo now includes three dedicated images for the text-only SLM path:

```bash
bash scripts/build_docker_slm_mistral31.sh
bash scripts/build_docker_slm_mistral31_static.sh
bash scripts/build_docker_slm_mistral31_vllm.sh
```

Runtime image defaults:
- image tag: `patent-pipeline:slm-mistral31`
- platform: `linux/amd64`
- legacy alias: `patent-pipeline:slm-ministral3`

Runtime image (`patent-pipeline:slm-mistral31`) is intentionally slim:
- no OCR stack
- pinned CUDA Torch + Transformers runtime
- Hugging Face and Torch caches under `/data/cache`
- `bitsandbytes` included for optional `--quantization bnb_8bit|bnb_4bit`

Static image (`patent-pipeline:slm-mistral31-static`) adds the compiler toolchain
required for `cache_implementation=static`:
- `build-essential`
- `python3-dev`
- `CC=/usr/bin/gcc`
- `CXX=/usr/bin/g++`

vLLM image defaults:
- image tag: `patent-pipeline:slm-mistral31-vllm`
- platform: `linux/amd64`
- based on the official `vllm/vllm-openai` image
- intended for `--backend vllm` benchmarks and smoke tests
- runs with `--ipc=host` on the VM to match vLLM Docker guidance

The VM extraction wrappers now default to an A100-friendly setup:
- `--torch-dtype bf16`
- `--attn-implementation sdpa`
- `--cache-implementation dynamic`

Example benchmark run inside the container:

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" \
  -v hf_cache:/data/cache/huggingface \
  -v torch_cache:/data/cache/torch \
  patent-pipeline:slm-mistral31 \
  python /app/scripts/bench_run_extract.py \
    --texts-dir /work/data/ocr_text \
    --out-root /work/output/slm \
    --run-name mistral31_smoke \
    --model-name mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --backend pytorch \
    --device cuda \
    --torch-dtype bf16 \
    --attn-implementation sdpa \
    --cache-implementation dynamic \
    --force
```

`flash_attention_2` is not installed by default in this image. The image is runtime-only and does not ship a C compiler, so prefer `cache_implementation=dynamic` or `auto` unless you build a compiler-enabled variant.

Example vLLM smoke inside the dedicated image:

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD:/work" \
  -v hf_cache:/data/cache/huggingface \
  -v torch_cache:/data/cache/torch \
  patent-pipeline:slm-mistral31-vllm \
  python /app/scripts/bench_run_extract.py \
    --texts-dir /work/data/ocr_text \
    --out-root /work/output/slm \
    --run-name mistral31_vllm_smoke \
    --model-name mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --backend vllm \
    --torch-dtype bf16 \
    --vllm-enable-prefix-caching \
    --vllm-tensor-parallel-size 1 \
    --vllm-gpu-memory-utilization 0.9 \
    --strategy baseline \
    --force
```

Known runtime note:
- if vLLM crashes during `Capturing CUDA graphs` with `CUDA error: operation not permitted`, see [docs/troubleshooting.md](./docs/troubleshooting.md) and retry with `VLLM_ENFORCE_EAGER=1`

Static smoke on the VM:

```bash
bash scripts/vm_smoke_mistral31.sh
bash scripts/vm_smoke_mistral31_static.sh
bash scripts/vm_smoke_mistral31_vllm.sh
```

Recommended split:
- `patent-pipeline:slm-mistral31`: production default for `dynamic|auto`
- `patent-pipeline:slm-mistral31-static`: compiler-enabled variant for `static`
- `patent-pipeline:slm-mistral31-vllm`: dedicated image for `backend=vllm` throughput and prefix-caching tests

---

## 📊 OCR Benchmark Example

| Engine            | Language mix | Accuracy (avg.) | Speed (pages/s) |
| ----------------- | ------------ | --------------- | --------------- |
| Tesseract         | fr+de+en     | 87%             | 2.1             |
| Doctr (ResNet-31) | fr+de+en     | **94%**         | 1.8             |
| Doctr (ViT-B)     | fr+de+en     | **96%**         | 1.4             |

> Benchmarks are indicative, measured on historical Swiss patent scans (1900–1960).

---

## ⚡ LightOnOCR GPU Batching

For `lightonocr` backend, pipeline GPU mode keeps `workers=1` by design.  
To scale throughput, increase `ocr_config.batch_size` (backend-side batching).

Example with `bench_run_ocr.py`:

```bash
python -u scripts/bench_run_ocr.py \
  --raw-dir /work/input/gold_standard_DE/PNGs_extracted \
  --out-root /work/patent-ocr-bench/output_vm/ocr \
  --run-name lightonocr_backend_b4 \
  --backend lightonocr \
  --segmentation backend \
  --workers 1 \
  --parallel none \
  --timings detailed \
  --limit 100 \
  --ocr-config-json '{"batch_size":4,"max_new_tokens":4096,"resize_longest_edge":1540,"temperature":0.0,"do_sample":false}'
```

---

## 🧾 License

Distributed under the **MIT License**.
© 2025 Pierre-Henri Delville.

---
