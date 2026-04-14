#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
BENCH_ROOT="${BENCH_ROOT:-$PARENT_ROOT/patent-ocr-bench}"
VM_PYTHON_SH="${VM_PYTHON_SH:-$BENCH_ROOT/scripts/vm_python.sh}"

cd "$ROOT_DIR"

VM_HOST="${VM_HOST:-vm-ab02.francecentral.cloudapp.azure.com}"
VM_USER="${VM_USER:-user-vm-ab02}"
VM_WORK="${VM_WORK:-/data/work}"
IMAGE="${IMAGE:-patent-pipeline:lightonocr}"

RAW_DIR="${RAW_DIR:-$ROOT_DIR/data/gold_standard_CH/PNGs_extracted}"
GOLD_JSONL="${GOLD_JSONL:-$ROOT_DIR/data/gold_standard_CH/ch500_swiss_gold_manual.jsonl}"
LOCAL_OUT_ROOT="${LOCAL_OUT_ROOT:-$ROOT_DIR/output/vm_ocr}"
RUN_NAME="${RUN_NAME:-lightonocr_ch500_backend_b8}"

LIMIT="${LIMIT:-}"
FORCE="${FORCE:-1}"
DESKEW="${DESKEW:-0}"
TIMINGS="${TIMINGS:-detailed}"
RESIZE_LONGEST_EDGE="${RESIZE_LONGEST_EDGE:-1280}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DO_SAMPLE="${DO_SAMPLE:-false}"
REMOTE_SCORE="${REMOTE_SCORE:-0}"

RAW_DIR="$(cd "$(dirname "$RAW_DIR")" && pwd)/$(basename "$RAW_DIR")"
GOLD_JSONL="$(cd "$(dirname "$GOLD_JSONL")" && pwd)/$(basename "$GOLD_JSONL")"
LOCAL_OUT_ROOT="$(cd "$(dirname "$LOCAL_OUT_ROOT")" && pwd)/$(basename "$LOCAL_OUT_ROOT")"

if [[ ! -x "$VM_PYTHON_SH" ]]; then
  echo "Missing vm helper: $VM_PYTHON_SH" >&2
  exit 1
fi

if [[ ! -d "$RAW_DIR" ]]; then
  echo "Missing raw dir: $RAW_DIR" >&2
  exit 1
fi

if [[ ! -f "$GOLD_JSONL" ]]; then
  echo "Missing gold jsonl: $GOLD_JSONL" >&2
  exit 1
fi

mkdir -p "$LOCAL_OUT_ROOT"

RAW_DIR_REMOTE="${RAW_DIR/$PARENT_ROOT/$VM_WORK}"
GOLD_JSONL_REMOTE="${GOLD_JSONL/$PARENT_ROOT/$VM_WORK}"
OUT_ROOT_REMOTE="${LOCAL_OUT_ROOT/$PARENT_ROOT/$VM_WORK}"

quote() { printf "%q" "$1"; }

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

FORCE_ARGS=()
if [[ "$FORCE" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

DESKEW_ARGS=()
if [[ "$DESKEW" == "1" ]]; then
  DESKEW_ARGS+=(--deskew)
else
  DESKEW_ARGS+=(--no-deskew)
fi

OCR_CONFIG_JSON=$(
  BATCH_SIZE="$BATCH_SIZE" \
  RESIZE_LONGEST_EDGE="$RESIZE_LONGEST_EDGE" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  TEMPERATURE="$TEMPERATURE" \
  DO_SAMPLE="$DO_SAMPLE" \
  python3 - <<'PY'
import json
import os

raw_do_sample = os.environ["DO_SAMPLE"].strip().lower()
if raw_do_sample not in {"true", "false"}:
    raise SystemExit(f"DO_SAMPLE must be true|false, got: {raw_do_sample!r}")

print(json.dumps({
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "resize_longest_edge": int(os.environ["RESIZE_LONGEST_EDGE"]),
    "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "do_sample": raw_do_sample == "true",
}, ensure_ascii=False))
PY
)

echo "[1/5] Upload raw images"
ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "mkdir -p $(quote "$RAW_DIR_REMOTE")"

TAR_META_ARGS=()
if tar --help 2>&1 | grep -qi 'bsdtar'; then
  TAR_META_ARGS+=(--no-mac-metadata --no-xattrs)
fi

COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
  tar "${TAR_META_ARGS[@]}" -czf - -C "$RAW_DIR" . | \
  ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "tar -xzf - -C $(quote "$RAW_DIR_REMOTE")"

echo "[2/5] Upload gold jsonl"
ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "mkdir -p $(quote "$(dirname "$GOLD_JSONL_REMOTE")")"
cat "$GOLD_JSONL" | ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "cat > $(quote "$GOLD_JSONL_REMOTE")"

echo "[3/5] Run OCR on VM"
bash "$VM_PYTHON_SH" --image "$IMAGE" \
  /app/scripts/bench_run_ocr.py \
  --raw-dir "$RAW_DIR" \
  --out-root "$LOCAL_OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --backend lightonocr \
  --segmentation backend \
  --workers 1 \
  --parallel none \
  --timings "$TIMINGS" \
  --backend-kwargs-json '{"device":"cuda"}' \
  --ocr-config-json "$OCR_CONFIG_JSON" \
  "${FORCE_ARGS[@]}" \
  "${DESKEW_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

if [[ "$REMOTE_SCORE" == "1" ]]; then
  echo "[4/5] Score OCR on VM"
  bash "$VM_PYTHON_SH" --image "$IMAGE" \
    /app/scripts/bench_score_ocr.py \
    --gold-jsonl "$GOLD_JSONL" \
    --run-dir "$LOCAL_OUT_ROOT/$RUN_NAME"
else
  echo "[4/5] Skip remote scoring"
fi

echo "[5/5] Fetch run dir"
ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" \
  "tar -czf - -C $(quote "$OUT_ROOT_REMOTE") $(quote "$RUN_NAME")" | \
  tar -xzf - -C "$LOCAL_OUT_ROOT"

echo "Local results: $LOCAL_OUT_ROOT/$RUN_NAME"
