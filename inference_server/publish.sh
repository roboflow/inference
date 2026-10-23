#!/usr/bin/env bash
#
# Build + push the inference_server GPU image to GCP Artifact Registry
# (staging internal — same project as other staging-internal tooling).
#
# Build context is the inference repo root, so the editable sibling packages
# (inference_models, inference_model_manager) are available to COPY.
#
# Prereq (one-time): gcloud auth configure-docker us-central1-docker.pkg.dev
#
# Usage:
#   ./publish.sh              # tag = current git short SHA + :latest
#   ./publish.sh phase0       # tag = phase0 + :latest
set -euo pipefail

REGISTRY="us-central1-docker.pkg.dev/roboflow-staging/inference-internal"
IMAGE="inference-server-next"
DOCKERFILE="inference_server/docker/Dockerfile.gpu"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"
FULL="${REGISTRY}/${IMAGE}"

echo "Building ${FULL}:${TAG}"
echo "  context:    ${REPO_ROOT}"
echo "  dockerfile: ${DOCKERFILE}"

# GPU nodes are amd64; build cross-platform from arm64 dev machines.
docker buildx build \
  --platform linux/amd64 \
  --pull \
  -f "${REPO_ROOT}/${DOCKERFILE}" \
  -t "${FULL}:${TAG}" \
  -t "${FULL}:latest" \
  "${REPO_ROOT}" \
  --push

echo "Pushed ${FULL}:${TAG}"
echo "Pushed ${FULL}:latest"
