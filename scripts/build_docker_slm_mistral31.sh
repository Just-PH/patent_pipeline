#!/usr/bin/env bash
set -euo pipefail

DEBUG="${DEBUG:-0}"
if [[ "$DEBUG" == "1" ]]; then
  set -x
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKERFILE="${DOCKERFILE:-$ROOT_DIR/Dockerfile.slm-mistral31}"
IMAGE_TAG="${IMAGE_TAG:-patent-pipeline:slm-mistral31}"
LEGACY_IMAGE_TAG="${LEGACY_IMAGE_TAG:-patent-pipeline:slm-ministral3}"
PLATFORM="${PLATFORM:-linux/amd64}"
BUILD_PROGRESS="${BUILD_PROGRESS:-plain}"
NO_CACHE="${NO_CACHE:-0}"
PULL="${PULL:-0}"
TAG_LEGACY_ALIAS="${TAG_LEGACY_ALIAS:-1}"
EXPORT_TAR="${EXPORT_TAR:-0}"
OUT_TAR="${OUT_TAR:-$ROOT_DIR/patent-pipeline_slm-mistral31_amd64.tar.gz}"

BUILD_ARGS=(
  --platform "$PLATFORM"
  --progress "$BUILD_PROGRESS"
  -f "$DOCKERFILE"
  -t "$IMAGE_TAG"
)

if [[ "$NO_CACHE" == "1" ]]; then
  BUILD_ARGS+=(--no-cache)
fi

if [[ "$PULL" == "1" ]]; then
  BUILD_ARGS+=(--pull)
fi

BUILD_CMD=("$DOCKER_BIN" build "${BUILD_ARGS[@]}" "$ROOT_DIR")

if ! "$DOCKER_BIN" info >/dev/null 2>&1; then
  echo "Docker daemon unavailable. Start Docker Desktop or verify docker permissions." >&2
  echo "Expected command:" >&2
  printf '  %q' "${BUILD_CMD[@]}" >&2
  printf '\n' >&2
  exit 1
fi

echo "Starting Docker build for $IMAGE_TAG"
printf 'Running:'
printf ' %q' "${BUILD_CMD[@]}"
printf '\n'

"${BUILD_CMD[@]}"

if [[ "$TAG_LEGACY_ALIAS" == "1" && "$LEGACY_IMAGE_TAG" != "$IMAGE_TAG" ]]; then
  "$DOCKER_BIN" tag "$IMAGE_TAG" "$LEGACY_IMAGE_TAG"
fi

if [[ "$EXPORT_TAR" == "1" ]]; then
  "$DOCKER_BIN" save "$IMAGE_TAG" | gzip -1 > "$OUT_TAR"
  echo "Exported tarball: $OUT_TAR"
fi

echo "Built image: $IMAGE_TAG"
if [[ "$TAG_LEGACY_ALIAS" == "1" && "$LEGACY_IMAGE_TAG" != "$IMAGE_TAG" ]]; then
  echo "Legacy alias: $LEGACY_IMAGE_TAG"
fi
