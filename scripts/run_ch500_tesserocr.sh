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
IMAGE="${IMAGE:-patent-pipeline:slm-ministral3}"

OCR_RUN_DIR="${OCR_RUN_DIR:-output/ch_ocr/tesserocr_ch500_backend_workers6}"
PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-$ROOT_DIR/data/prompts/swiss_patent_biblio_multilingual_v1.txt}"
LOCAL_OUT_ROOT="${LOCAL_OUT_ROOT:-$ROOT_DIR/output/vm_extract}"
RUN_NAME="${RUN_NAME:-mistral31_24b_ch_prenotation}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
MAX_OCR_CHARS="${MAX_OCR_CHARS:-12000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
STRATEGY="${STRATEGY:-baseline}"
TIMINGS="${TIMINGS:-basic}"
LIMIT="${LIMIT:-}"
SAVE_RAW_OUTPUT="${SAVE_RAW_OUTPUT:-0}"
SAVE_STRATEGY_META="${SAVE_STRATEGY_META:-1}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
GPU_BUSY_MIB_THRESHOLD="${GPU_BUSY_MIB_THRESHOLD:-2048}"

TEXTS_DIR_LOCAL="$ROOT_DIR/$OCR_RUN_DIR/texts"
TEXTS_DIR_LOCAL="$(cd "$(dirname "$TEXTS_DIR_LOCAL")" && pwd)/$(basename "$TEXTS_DIR_LOCAL")"
PROMPT_TEMPLATE_PATH="$(cd "$(dirname "$PROMPT_TEMPLATE_PATH")" && pwd)/$(basename "$PROMPT_TEMPLATE_PATH")"
LOCAL_OUT_ROOT="$(cd "$(dirname "$LOCAL_OUT_ROOT")" && pwd)/$(basename "$LOCAL_OUT_ROOT")"

if [[ ! -x "$VM_PYTHON_SH" ]]; then
  echo "Missing vm helper: $VM_PYTHON_SH" >&2
  exit 1
fi

if [[ ! -d "$TEXTS_DIR_LOCAL" ]]; then
  echo "Missing OCR texts dir: $TEXTS_DIR_LOCAL" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_TEMPLATE_PATH" ]]; then
  echo "Missing prompt template: $PROMPT_TEMPLATE_PATH" >&2
  exit 1
fi

if [[ "$TEXTS_DIR_LOCAL" != "$PARENT_ROOT"* ]]; then
  echo "Texts dir must be under $PARENT_ROOT for VM path remapping" >&2
  exit 1
fi

mkdir -p "$LOCAL_OUT_ROOT"

TEXTS_DIR_REMOTE="${TEXTS_DIR_LOCAL/$PARENT_ROOT/$VM_WORK}"
OUT_ROOT_REMOTE="${LOCAL_OUT_ROOT/$PARENT_ROOT/$VM_WORK}"

quote() { printf "%q" "$1"; }

check_remote_gpu_idle() {
  if [[ "$SKIP_GPU_CHECK" == "1" ]]; then
    echo "[preflight] SKIP_GPU_CHECK=1 -> GPU check skipped"
    return 0
  fi

  local used_mib
  used_mib="$(ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk 'NF{sum+=\$1} END{print sum+0}'")"
  used_mib="${used_mib//[[:space:]]/}"

  if [[ -z "$used_mib" ]]; then
    echo "[preflight] Could not read remote GPU usage; continuing without guard."
    return 0
  fi

  if (( used_mib > GPU_BUSY_MIB_THRESHOLD )); then
    echo "Remote GPU appears busy (${used_mib} MiB in use by compute apps)." >&2
    echo "Refusing to launch a second extraction to avoid CUDA OOM." >&2
    echo "Set SKIP_GPU_CHECK=1 only if you intentionally want concurrent runs." >&2
    ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "nvidia-smi" >&2 || true
    exit 1
  fi
}

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

EXTRA_ARGS=()
if [[ "$SAVE_RAW_OUTPUT" == "1" ]]; then
  EXTRA_ARGS+=(--save-raw-output)
fi
if [[ "$SAVE_STRATEGY_META" == "1" ]]; then
  EXTRA_ARGS+=(--save-strategy-meta)
fi

check_remote_gpu_idle

echo "[1/4] Upload texts"
ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "mkdir -p $(quote "$TEXTS_DIR_REMOTE")"

# Avoid macOS xattrs/provenance headers (removes LIBARCHIVE.xattr warnings remotely).
TAR_META_ARGS=()
if tar --help 2>&1 | grep -qi 'bsdtar'; then
  TAR_META_ARGS+=(--no-mac-metadata --no-xattrs)
fi

COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
  tar "${TAR_META_ARGS[@]}" -czf - -C "$TEXTS_DIR_LOCAL" . | \
  ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "tar -xzf - -C $(quote "$TEXTS_DIR_REMOTE")"

echo "[2/4] Run extraction on VM"
bash "$VM_PYTHON_SH" --image "$IMAGE" \
  /app/scripts/bench_run_extract.py \
  --texts-dir "$TEXTS_DIR_LOCAL" \
  --out-root "$LOCAL_OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --model-name "$MODEL_NAME" \
  --backend pytorch \
  --device cuda \
  --prompt-template-path "$PROMPT_TEMPLATE_PATH" \
  --strategy "$STRATEGY" \
  --max-ocr-chars "$MAX_OCR_CHARS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0.0 \
  --timings "$TIMINGS" \
  --force \
  "${LIMIT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

echo "[3/4] Fetch results"
mkdir -p "$LOCAL_OUT_ROOT"
ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" \
  "tar -czf - -C $(quote "$OUT_ROOT_REMOTE") $(quote "$RUN_NAME")" | \
  tar -xzf - -C "$LOCAL_OUT_ROOT"

echo "[4/4] Done"
echo "Local results: $LOCAL_OUT_ROOT/$RUN_NAME"
