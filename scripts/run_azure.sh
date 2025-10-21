#!/usr/bin/env bash
set -euo pipefail
export DEVICE=cuda
export HF_MODEL=${HF_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
# Sur A100 tu peux forcer fp16 côté code si tu veux aller plus loin
poetry run python -m patent_pipeline
