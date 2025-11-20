#!/usr/bin/env bash
set -euo pipefail

export DEVICE=${DEVICE:-mps}    # mps par défaut sur Mac
export HF_MODEL="mlx-community/Mistral-7B-Instruct-v0.3"

poetry run python -m patent_pipeline --threads 3 --backend "tesseract" --preproc_method "otsu" --country_hint "de"
