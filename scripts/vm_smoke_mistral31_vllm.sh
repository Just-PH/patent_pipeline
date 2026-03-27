#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-patent-pipeline:slm-mistral31-vllm}"
TEXTS_DIR="${TEXTS_DIR:-/data/work/patent_pipeline/output/ch_ocr/tesserocr_ch500_backend_workers6/texts}"
OUT_ROOT="${OUT_ROOT:-/data/work/patent_pipeline/output_vm/slm_smoke}"
RUN_NAME="${RUN_NAME:-smoke_mistral31_vllm_prefix_3docs}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"

TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
STRATEGY="${STRATEGY:-baseline}"
TIMINGS="${TIMINGS:-detailed}"
LIMIT="${LIMIT:-3}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-4.0}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DOC_BATCH_SIZE="${VLLM_DOC_BATCH_SIZE:-32}"
VLLM_SORT_BY_PROMPT_LENGTH="${VLLM_SORT_BY_PROMPT_LENGTH:-1}"
VLLM_TOKENIZER_MODE="${VLLM_TOKENIZER_MODE:-auto}"

WORK_ROOT="/data/work"
CACHE_ROOT="${WORK_ROOT}/.cache"
HF_HOME="${CACHE_ROOT}/huggingface"
TORCH_HOME="${CACHE_ROOT}/torch"
XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
PIP_CACHE_DIR="${CACHE_ROOT}/pip"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

EXTRA_ARGS=()
if [[ "$VLLM_ENABLE_PREFIX_CACHING" == "1" ]]; then
  EXTRA_ARGS+=(--vllm-enable-prefix-caching)
fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  EXTRA_ARGS+=(--vllm-max-model-len "$VLLM_MAX_MODEL_LEN")
fi
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  EXTRA_ARGS+=(--vllm-enforce-eager)
fi
if [[ "$VLLM_SORT_BY_PROMPT_LENGTH" != "1" ]]; then
  EXTRA_ARGS+=(--no-vllm-sort-by-prompt-length)
fi

docker run --rm --gpus all --ipc=host -v "${WORK_ROOT}:/work" "$IMAGE" bash -lc "
  export HF_HOME=/work/.cache/huggingface \
         HF_HUB_CACHE=/work/.cache/huggingface \
         TORCH_HOME=/work/.cache/torch \
         XDG_CACHE_HOME=/work/.cache/xdg \
         PIP_CACHE_DIR=/work/.cache/pip && \
  python -u /app/scripts/bench_run_extract.py \
    --texts-dir /work${TEXTS_DIR#${WORK_ROOT}} \
    --out-root /work${OUT_ROOT#${WORK_ROOT}} \
    --run-name ${RUN_NAME} \
    --model-name ${MODEL_NAME} \
    --backend vllm \
    --torch-dtype ${TORCH_DTYPE} \
    --vllm-tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE} \
    --vllm-gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
    --vllm-swap-space ${VLLM_SWAP_SPACE} \
    --vllm-doc-batch-size ${VLLM_DOC_BATCH_SIZE} \
    --vllm-tokenizer-mode ${VLLM_TOKENIZER_MODE} \
    --strategy ${STRATEGY} \
    --timings ${TIMINGS} \
    --save-raw-output \
    --save-strategy-meta \
    --limit ${LIMIT} \
    --force \
    ${EXTRA_ARGS[*]}
"

RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
PREDS_JSONL="${RUN_DIR}/preds.jsonl"
RUN_JSON="${RUN_DIR}/run.json"

python3 - "$PREDS_JSONL" "$RUN_JSON" "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

preds_path = Path(sys.argv[1])
run_json_path = Path(sys.argv[2])
limit = int(sys.argv[3])

if not preds_path.exists():
    raise SystemExit(f"missing preds.jsonl: {preds_path}")
if not run_json_path.exists():
    raise SystemExit(f"missing run.json: {run_json_path}")

rows = [json.loads(line) for line in preds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
errors = [row.get("error") for row in rows if row.get("error")]
run_meta = json.loads(run_json_path.read_text(encoding="utf-8"))
config = run_meta.get("config") or {}
strategy_stats = run_meta.get("strategy_stats") or {}

summary = {
    "rows": len(rows),
    "errors": len(errors),
    "backend": config.get("backend"),
    "vllm_enable_prefix_caching": config.get("vllm_enable_prefix_caching"),
    "mean_confidence": strategy_stats.get("mean_confidence"),
    "first_error": errors[0] if errors else None,
}
print(summary)

if config.get("backend") != "vllm":
    raise SystemExit(f"unexpected backend in run.json: {config.get('backend')!r}")
if len(rows) != limit:
    raise SystemExit(f"expected {limit} rows, found {len(rows)}")
if errors:
    raise SystemExit(f"expected zero errors, found {len(errors)}")
PY
