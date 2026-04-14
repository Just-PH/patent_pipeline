#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-patent-pipeline:lightonocr}"
VM_HOST="${VM_HOST:-vm-ab02.francecentral.cloudapp.azure.com}"
VM_USER="${VM_USER:-user-vm-ab02}"
REMOTE_DOCKER_BIN="${REMOTE_DOCKER_BIN:-docker}"
LOCAL_DOCKER_BIN="${LOCAL_DOCKER_BIN:-docker}"
GZIP_LEVEL="${GZIP_LEVEL:-1}"

if ! "$LOCAL_DOCKER_BIN" image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Local image not found: $IMAGE_TAG" >&2
  echo "Build it first, for example:" >&2
  echo "  IMAGE_TAG=$IMAGE_TAG bash scripts/build_docker_lightonocr.sh" >&2
  exit 1
fi

echo "[1/3] Check SSH access"
ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "echo ok" >/dev/null

echo "[2/3] Stream image to VM and load it"
"$LOCAL_DOCKER_BIN" save "$IMAGE_TAG" | gzip "-${GZIP_LEVEL}" | \
  ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" "gunzip | ${REMOTE_DOCKER_BIN} load"

echo "[3/3] Verify image on VM"
ssh -o BatchMode=yes "${VM_USER}@${VM_HOST}" \
  "${REMOTE_DOCKER_BIN} image inspect $(printf '%q' "$IMAGE_TAG") --format '{{.Id}}'"

echo "Image available on VM: $IMAGE_TAG"
