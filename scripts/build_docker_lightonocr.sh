#!/usr/bin/env bash
set -euo pipefail

DEBUG="${DEBUG:-0}"
if [[ "$DEBUG" == "1" ]]; then
  set -x
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKERFILE="${DOCKERFILE:-$ROOT_DIR/Dockerfile.lightonocr}"
IMAGE_TAG="${IMAGE_TAG:-patent-pipeline:lightonocr}"
PLATFORM="${PLATFORM:-linux/amd64}"
BUILD_PROGRESS="${BUILD_PROGRESS:-plain}"
USE_BUILDX="${USE_BUILDX:-1}"
NO_CACHE="${NO_CACHE:-0}"
PULL="${PULL:-0}"
EXPORT_TAR="${EXPORT_TAR:-0}"
OUT_TAR="${OUT_TAR:-$ROOT_DIR/patent-pipeline_lightonocr_amd64.tar.gz}"

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

if [[ "$USE_BUILDX" == "1" ]]; then
  if [[ "$PLATFORM" == *,* ]]; then
    echo "USE_BUILDX=1 with --load supports only a single platform. Got: $PLATFORM" >&2
    exit 1
  fi
  BUILD_CMD=("$DOCKER_BIN" buildx build --load "${BUILD_ARGS[@]}" "$ROOT_DIR")
else
  BUILD_CMD=("$DOCKER_BIN" build "${BUILD_ARGS[@]}" "$ROOT_DIR")
fi

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

if ! "$DOCKER_BIN" image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Build completed but image is not available in the local Docker image store: $IMAGE_TAG" >&2
  echo "If you changed the builder, retry with USE_BUILDX=1 so the image is loaded locally." >&2
  exit 1
fi

if [[ "$EXPORT_TAR" == "1" ]]; then
  "$DOCKER_BIN" save "$IMAGE_TAG" | gzip -1 > "$OUT_TAR"
  echo "Exported tarball: $OUT_TAR"
fi

echo "Built image: $IMAGE_TAG"
