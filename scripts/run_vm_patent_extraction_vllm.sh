#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"
PARENT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-$PARENT_ROOT/patent-ocr-bench}"
HELPER="${HELPER:-$BENCH_ROOT/scripts/vm_python.sh}"

IMAGE="${IMAGE:-patent-extraction:vllm}"
PROFILE="${PROFILE:-de_legacy_v4}"
PROFILE_PATH="${PROFILE_PATH:-}"
PROMPT_PATH="${PROMPT_PATH:-}"
TEXTS_DIR="${TEXTS_DIR:-$BENCH_ROOT/data/real/real500_tesserocr_texts}"
OUT_ROOT="${OUT_ROOT:-$BENCH_ROOT/output_vm/slm_500_real}"
RUN_NAME="${RUN_NAME:-smoke_patent_extraction_vllm}"
LIMIT="${LIMIT:-}"

MODEL_NAME="${MODEL_NAME:-}"
BACKEND="${BACKEND:-}"
DEVICE="${DEVICE:-}"
DEVICE_MAP="${DEVICE_MAP:-}"
PROMPT_ID="${PROMPT_ID:-}"
GUARDRAIL_PROFILE="${GUARDRAIL_PROFILE:-}"
STRATEGY="${STRATEGY:-}"
TIMINGS="${TIMINGS:-}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
QUANTIZATION="${QUANTIZATION:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
CACHE_IMPLEMENTATION="${CACHE_IMPLEMENTATION:-}"
MAX_OCR_CHARS="${MAX_OCR_CHARS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}"
VLLM_DOC_BATCH_SIZE="${VLLM_DOC_BATCH_SIZE:-}"
VLLM_SORT_BY_PROMPT_LENGTH="${VLLM_SORT_BY_PROMPT_LENGTH:-}"
VLLM_TOKENIZER_MODE="${VLLM_TOKENIZER_MODE:-}"
SAVE_RAW_OUTPUT="${SAVE_RAW_OUTPUT:-}"
SAVE_STRATEGY_META="${SAVE_STRATEGY_META:-}"
DO_SAMPLE="${DO_SAMPLE:-}"
FORCE="${FORCE:-1}"

if [[ ! -f "$HELPER" ]]; then
  echo "Missing helper: $HELPER" >&2
  exit 1
fi

if [[ "$TEXTS_DIR" != /data/work/* && "$TEXTS_DIR" != /work/* && ! -d "$TEXTS_DIR" ]]; then
  echo "Missing texts dir: $TEXTS_DIR" >&2
  exit 1
fi

if [[ -n "$PROFILE_PATH" && "$PROFILE_PATH" != /data/work/* && "$PROFILE_PATH" != /work/* && ! -f "$PROFILE_PATH" ]]; then
  echo "Missing profile path: $PROFILE_PATH" >&2
  exit 1
fi

if [[ -n "$PROMPT_PATH" && "$PROMPT_PATH" != /data/work/* && "$PROMPT_PATH" != /work/* && ! -f "$PROMPT_PATH" ]]; then
  echo "Missing prompt path: $PROMPT_PATH" >&2
  exit 1
fi

ARGS=(
  /app/scripts/run_patent_extraction.py
  --texts-dir "$TEXTS_DIR"
  --out-root "$OUT_ROOT"
  --run-name "$RUN_NAME"
  --profile "$PROFILE"
)

if [[ -n "$PROFILE_PATH" ]]; then
  ARGS+=(--profile-path "$PROFILE_PATH")
fi
if [[ -n "$PROMPT_PATH" ]]; then
  ARGS+=(--prompt-path "$PROMPT_PATH")
fi
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ -n "$MODEL_NAME" ]]; then
  ARGS+=(--model-name "$MODEL_NAME")
fi
if [[ -n "$BACKEND" ]]; then
  ARGS+=(--backend "$BACKEND")
fi
if [[ -n "$DEVICE" ]]; then
  ARGS+=(--device "$DEVICE")
fi
if [[ -n "$DEVICE_MAP" ]]; then
  ARGS+=(--device-map "$DEVICE_MAP")
fi
if [[ -n "$PROMPT_ID" ]]; then
  ARGS+=(--prompt-id "$PROMPT_ID")
fi
if [[ -n "$GUARDRAIL_PROFILE" ]]; then
  ARGS+=(--guardrail-profile "$GUARDRAIL_PROFILE")
fi
if [[ -n "$STRATEGY" ]]; then
  ARGS+=(--strategy "$STRATEGY")
fi
if [[ -n "$TIMINGS" ]]; then
  ARGS+=(--timings "$TIMINGS")
fi
if [[ -n "$TORCH_DTYPE" ]]; then
  ARGS+=(--torch-dtype "$TORCH_DTYPE")
fi
if [[ -n "$QUANTIZATION" ]]; then
  ARGS+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  ARGS+=(--attn-implementation "$ATTN_IMPLEMENTATION")
fi
if [[ -n "$CACHE_IMPLEMENTATION" ]]; then
  ARGS+=(--cache-implementation "$CACHE_IMPLEMENTATION")
fi
if [[ -n "$MAX_OCR_CHARS" ]]; then
  ARGS+=(--max-ocr-chars "$MAX_OCR_CHARS")
fi
if [[ -n "$MAX_NEW_TOKENS" ]]; then
  ARGS+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi
if [[ -n "$TEMPERATURE" ]]; then
  ARGS+=(--temperature "$TEMPERATURE")
fi
if [[ "$DO_SAMPLE" == "1" ]]; then
  ARGS+=(--do-sample)
fi
if [[ "$SAVE_RAW_OUTPUT" == "1" ]]; then
  ARGS+=(--save-raw-output)
elif [[ "$SAVE_RAW_OUTPUT" == "0" ]]; then
  ARGS+=(--no-save-raw-output)
fi
if [[ "$SAVE_STRATEGY_META" == "1" ]]; then
  ARGS+=(--save-strategy-meta)
elif [[ "$SAVE_STRATEGY_META" == "0" ]]; then
  ARGS+=(--no-save-strategy-meta)
fi
if [[ "$VLLM_ENABLE_PREFIX_CACHING" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
elif [[ "$VLLM_ENABLE_PREFIX_CACHING" == "0" ]]; then
  ARGS+=(--no-enable-prefix-caching)
fi
if [[ -n "$VLLM_TENSOR_PARALLEL_SIZE" ]]; then
  ARGS+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")
fi
if [[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]]; then
  ARGS+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  ARGS+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
fi
if [[ -n "$VLLM_SWAP_SPACE" ]]; then
  ARGS+=(--swap-space "$VLLM_SWAP_SPACE")
fi
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  ARGS+=(--enforce-eager)
elif [[ "$VLLM_ENFORCE_EAGER" == "0" ]]; then
  ARGS+=(--no-enforce-eager)
fi
if [[ -n "$VLLM_DOC_BATCH_SIZE" ]]; then
  ARGS+=(--doc-batch-size "$VLLM_DOC_BATCH_SIZE")
fi
if [[ "$VLLM_SORT_BY_PROMPT_LENGTH" == "1" ]]; then
  ARGS+=(--sort-by-prompt-length)
elif [[ "$VLLM_SORT_BY_PROMPT_LENGTH" == "0" ]]; then
  ARGS+=(--no-sort-by-prompt-length)
fi
if [[ -n "$VLLM_TOKENIZER_MODE" ]]; then
  ARGS+=(--tokenizer-mode "$VLLM_TOKENIZER_MODE")
fi
if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

exec bash "$HELPER" --image "$IMAGE" "${ARGS[@]}"
