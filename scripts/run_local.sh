#!/usr/bin/env bash
set -euo pipefail
export DEVICE=${DEVICE:-mps}     # mps par défaut sur Mac
export HF_MODEL=${HF_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
poetry run python -m patent_pipeline
