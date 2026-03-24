#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OCR_RUN_DIR="${OCR_RUN_DIR:-output/ch_ocr/tesserocr_ch500_backend_workers6}"
OUT_ROOT="${OUT_ROOT:-output/slm_extract}"
RUN_NAME="${RUN_NAME:-mistral7b_mlx_ch_biblio_v1}"
MODEL_NAME="${MODEL_NAME:-mlx-community/Mistral-7B-Instruct-v0.3}"
PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-data/prompts/swiss_patent_biblio_multilingual_v1.txt}"
MAX_OCR_CHARS="${MAX_OCR_CHARS:-12000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
STRATEGY="${STRATEGY:-baseline}"
TIMINGS="${TIMINGS:-basic}"
LIMIT="${LIMIT:-}"

if [[ ! -d "$OCR_RUN_DIR/texts" ]]; then
  echo "Missing OCR texts dir: $OCR_RUN_DIR/texts" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_TEMPLATE_PATH" ]]; then
  echo "Missing prompt template: $PROMPT_TEMPLATE_PATH" >&2
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

export PYTHONPATH="${PYTHONPATH:-src}"

python -u scripts/bench_run_extract.py \
  --ocr-run-dir "$OCR_RUN_DIR" \
  --out-root "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --model-name "$MODEL_NAME" \
  --backend mlx \
  --prompt-template-path "$PROMPT_TEMPLATE_PATH" \
  --strategy "$STRATEGY" \
  --max-ocr-chars "$MAX_OCR_CHARS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0.0 \
  --timings "$TIMINGS" \
  --force \
  "${LIMIT_ARGS[@]}"
