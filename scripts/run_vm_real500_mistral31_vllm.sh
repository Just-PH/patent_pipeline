#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"
PARENT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-$PARENT_ROOT/patent-ocr-bench}"
HELPER="${HELPER:-$BENCH_ROOT/scripts/vm_python.sh}"

IMAGE="${IMAGE:-patent-pipeline:slm-mistral31-vllm}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
TEXTS_DIR="${TEXTS_DIR:-$BENCH_ROOT/data/real/real500_tesserocr_texts}"
OUT_ROOT="${OUT_ROOT:-$BENCH_ROOT/output_vm/slm_500_real}"
PROMPT_ID="${PROMPT_ID:-}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-$BENCH_ROOT/prompts/extraction/exchaustive_prompt_V2-fixing-inventors-recalls.txt}"
GUARDRAIL_PROFILE="${GUARDRAIL_PROFILE:-auto}"
RUN_NAME="${RUN_NAME:-real500_mistral31_24b__baseline__v2inv_vllm}"

TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
STRATEGY="${STRATEGY:-baseline}"
TIMINGS="${TIMINGS:-detailed}"
LIMIT="${LIMIT:-}"
SAVE_RAW_OUTPUT="${SAVE_RAW_OUTPUT:-1}"
SAVE_STRATEGY_META="${SAVE_STRATEGY_META:-1}"

VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-4.0}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DOC_BATCH_SIZE="${VLLM_DOC_BATCH_SIZE:-32}"
VLLM_SORT_BY_PROMPT_LENGTH="${VLLM_SORT_BY_PROMPT_LENGTH:-1}"
VLLM_TOKENIZER_MODE="${VLLM_TOKENIZER_MODE:-auto}"

if [[ ! -f "$HELPER" ]]; then
  echo "Missing helper: $HELPER" >&2
  exit 1
fi

if [[ "$TEXTS_DIR" != /data/work/* && ! -d "$TEXTS_DIR" ]]; then
  echo "Missing texts dir: $TEXTS_DIR" >&2
  exit 1
fi

if [[ -z "$PROMPT_ID" && "$PROMPT_TEMPLATE" != /data/work/* && ! -f "$PROMPT_TEMPLATE" ]]; then
  echo "Missing prompt template: $PROMPT_TEMPLATE" >&2
  exit 1
fi

ARGS=(
  /app/scripts/bench_run_extract.py
  --texts-dir "$TEXTS_DIR"
  --out-root "$OUT_ROOT"
  --run-name "$RUN_NAME"
  --model-name "$MODEL_NAME"
  --backend vllm
  --torch-dtype "$TORCH_DTYPE"
  --strategy "$STRATEGY"
  --merge-policy prefer_non_null
  --header-lines 30
  --targeted-rerun-threshold 0.6
  --self-consistency-n 2
  --self-consistency-temp 0.2
  --timings "$TIMINGS"
  --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm-swap-space "$VLLM_SWAP_SPACE"
  --vllm-doc-batch-size "$VLLM_DOC_BATCH_SIZE"
  --vllm-tokenizer-mode "$VLLM_TOKENIZER_MODE"
  --guardrail-profile "$GUARDRAIL_PROFILE"
  --force
)

if [[ -n "$PROMPT_ID" ]]; then
  ARGS+=(--prompt-id "$PROMPT_ID")
else
  ARGS+=(--prompt-template-path "$PROMPT_TEMPLATE")
fi

if [[ "$VLLM_ENABLE_PREFIX_CACHING" == "1" ]]; then
  ARGS+=(--vllm-enable-prefix-caching)
fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  ARGS+=(--vllm-max-model-len "$VLLM_MAX_MODEL_LEN")
fi
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  ARGS+=(--vllm-enforce-eager)
fi
if [[ "$VLLM_SORT_BY_PROMPT_LENGTH" != "1" ]]; then
  ARGS+=(--no-vllm-sort-by-prompt-length)
fi
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "$SAVE_RAW_OUTPUT" == "1" ]]; then
  ARGS+=(--save-raw-output)
fi
if [[ "$SAVE_STRATEGY_META" == "1" ]]; then
  ARGS+=(--save-strategy-meta)
fi

exec bash "$HELPER" --image "$IMAGE" "${ARGS[@]}"
