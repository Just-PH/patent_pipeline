#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DOCKERFILE="${DOCKERFILE:-$ROOT_DIR/Dockerfile.slm-mistral31-vllm}"
export IMAGE_TAG="${IMAGE_TAG:-patent-pipeline:slm-mistral31-vllm}"
export LEGACY_IMAGE_TAG="${LEGACY_IMAGE_TAG:-$IMAGE_TAG}"
export OUT_TAR="${OUT_TAR:-$ROOT_DIR/patent-pipeline_slm-mistral31-vllm_amd64.tar.gz}"
export TAG_LEGACY_ALIAS="${TAG_LEGACY_ALIAS:-0}"

exec "$ROOT_DIR/scripts/build_docker_slm_mistral31.sh"
