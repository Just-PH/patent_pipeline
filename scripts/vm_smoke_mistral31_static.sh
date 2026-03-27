#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export IMAGE="${IMAGE:-patent-pipeline:slm-mistral31-static}"
export RUN_NAME="${RUN_NAME:-smoke_mistral31_static_3docs}"
export CACHE_IMPLEMENTATION="${CACHE_IMPLEMENTATION:-static}"
export EXPECT_ERRORS="${EXPECT_ERRORS:-0}"

exec "$ROOT_DIR/scripts/vm_smoke_mistral31.sh"
