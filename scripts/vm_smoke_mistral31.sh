#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-patent-pipeline:slm-mistral31}"
TEXTS_DIR="${TEXTS_DIR:-/data/work/patent_pipeline/output/ch_ocr/tesserocr_ch500_backend_workers6/texts}"
OUT_ROOT="${OUT_ROOT:-/data/work/patent_pipeline/output_vm/slm_smoke}"
RUN_NAME="${RUN_NAME:-smoke_mistral31_dynamic_3docs}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"

TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
QUANTIZATION="${QUANTIZATION:-none}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
CACHE_IMPLEMENTATION="${CACHE_IMPLEMENTATION:-dynamic}"
STRATEGY="${STRATEGY:-baseline}"
TIMINGS="${TIMINGS:-detailed}"
LIMIT="${LIMIT:-3}"

EXPECT_ERRORS="${EXPECT_ERRORS:-0}"
EXPECT_ERROR_SUBSTRING="${EXPECT_ERROR_SUBSTRING:-}"

WORK_ROOT="/data/work"
CACHE_ROOT="${WORK_ROOT}/.cache"
HF_HOME="${CACHE_ROOT}/huggingface"
HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
TRANSFORMERS_CACHE="${HF_HOME}/transformers"
TORCH_HOME="${CACHE_ROOT}/torch"
XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
PIP_CACHE_DIR="${CACHE_ROOT}/pip"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

docker run --rm --gpus all -v "${WORK_ROOT}:/work" "$IMAGE" bash -lc "
  export HF_HOME=/work/.cache/huggingface \
         HUGGINGFACE_HUB_CACHE=/work/.cache/huggingface/hub \
         TRANSFORMERS_CACHE=/work/.cache/huggingface/transformers \
         TORCH_HOME=/work/.cache/torch \
         XDG_CACHE_HOME=/work/.cache/xdg \
         PIP_CACHE_DIR=/work/.cache/pip && \
  python -u /app/scripts/bench_run_extract.py \
    --texts-dir /work${TEXTS_DIR#${WORK_ROOT}} \
    --out-root /work${OUT_ROOT#${WORK_ROOT}} \
    --run-name ${RUN_NAME} \
    --model-name ${MODEL_NAME} \
    --backend pytorch \
    --device cuda \
    --torch-dtype ${TORCH_DTYPE} \
    --quantization ${QUANTIZATION} \
    --attn-implementation ${ATTN_IMPLEMENTATION} \
    --cache-implementation ${CACHE_IMPLEMENTATION} \
    --strategy ${STRATEGY} \
    --timings ${TIMINGS} \
    --save-raw-output \
    --save-strategy-meta \
    --limit ${LIMIT} \
    --force
"

RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
PREDS_JSONL="${RUN_DIR}/preds.jsonl"
RUN_JSON="${RUN_DIR}/run.json"

python3 - "$PREDS_JSONL" "$RUN_JSON" "$LIMIT" "$CACHE_IMPLEMENTATION" "$EXPECT_ERRORS" "$EXPECT_ERROR_SUBSTRING" <<'PY'
import json
import sys
from pathlib import Path

preds_path = Path(sys.argv[1])
run_json_path = Path(sys.argv[2])
limit = int(sys.argv[3])
cache_implementation = sys.argv[4]
expect_errors = sys.argv[5] == "1"
expect_error_substring = sys.argv[6]

if not preds_path.exists():
    raise SystemExit(f"missing preds.jsonl: {preds_path}")
if not run_json_path.exists():
    raise SystemExit(f"missing run.json: {run_json_path}")

rows = [json.loads(line) for line in preds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
errors = [row.get("error") for row in rows if row.get("error")]
error_details = [row.get("error_detail") for row in rows if row.get("error_detail")]
run_meta = json.loads(run_json_path.read_text(encoding="utf-8"))

config = run_meta.get("config") or {}
strategy_stats = run_meta.get("strategy_stats") or {}

actual_cache_implementation = config.get("cache_implementation", run_meta.get("cache_implementation"))
summary = {
    "rows": len(rows),
    "errors": len(errors),
    "cache_implementation": actual_cache_implementation,
    "mean_confidence": strategy_stats.get("mean_confidence", run_meta.get("mean_confidence")),
    "first_error": errors[0] if errors else None,
    "first_error_detail": error_details[0] if error_details else None,
}
print(summary)

expected_cache_values = {cache_implementation}
if cache_implementation in {"dynamic", "auto"}:
    expected_cache_values.add(None)

if actual_cache_implementation not in expected_cache_values:
    raise SystemExit(
        f"unexpected cache_implementation in run.json: "
        f"{actual_cache_implementation} not in {sorted(expected_cache_values, key=str)}"
    )

if expect_errors:
    if not errors:
        raise SystemExit("expected at least one error, found zero")
    error_text = "\n".join([errors[0] if errors else "", error_details[0] if error_details else ""]).strip()
    if expect_error_substring and expect_error_substring not in error_text:
        raise SystemExit(
            f"expected substring {expect_error_substring!r} in first error, "
            f"got {error_text!r}"
        )
else:
    if len(rows) != limit:
        raise SystemExit(f"expected {limit} rows, found {len(rows)}")
    if errors:
        raise SystemExit(f"expected zero errors, found {len(errors)}")
PY
