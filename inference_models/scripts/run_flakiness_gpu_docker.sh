#!/usr/bin/env bash
# Run check_prediction_flakiness.py inside the Roboflow GPU inference image (or compatible),
# overriding the default uvicorn entrypoint. Mounts this repo's inference_models tree for
# the script + package, and a host directory for reports, caches, and logs.
#
# Prerequisites: Docker with NVIDIA Container Toolkit (--gpus all).
#
# Examples:
#   export ROBOFLOW_API_KEY=...
#   ./run_flakiness_gpu_docker.sh --num-test-images 25 --iterations 3
#
#   FLAKINESS_GPU_IMAGE=myregistry/inference-gpu:dev RESULTS_DIR=~/flakiness-out \
#     ./run_flakiness_gpu_docker.sh --num-test-images 10 --iterations 2
#
# Extra docker args (e.g. --shm-size):
#   EXTRA_DOCKER_RUN_ARGS="--shm-size=8g" ./run_flakiness_gpu_docker.sh ...

set -euo pipefail

IMAGE="${FLAKINESS_GPU_IMAGE:-roboflow/roboflow-inference-server-gpu:1.1.2-otel2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent of scripts/ is the inference_models package root (contains inference_models/, scripts/, pyproject.toml).
INFERENCE_MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INFERENCE_MODELS_ROOT}/.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-${PWD}/flakiness_docker_results}"
mkdir -p "${RESULTS_DIR}"

# Paths inside the container (must match -v mounts below).
CONTAINER_WORKSPACE="/workspace/inference_models"
CONTAINER_RESULTS="/results"

# Default models JSON: the file under the mounted inference_models tree.
CONTAINER_MODELS_CONFIG="${CONTAINER_MODELS_CONFIG:-${CONTAINER_WORKSPACE}/scripts/model_config.json}"

# Optional: env file on the host (repo .env is a common choice).
ENV_FILE_ARGS=()
if [[ -n "${FLAKINESS_ENV_FILE:-}" ]]; then
  ENV_FILE_ARGS=(--env-file "${FLAKINESS_ENV_FILE}")
elif [[ -f "${REPO_ROOT}/.env" ]]; then
  ENV_FILE_ARGS=(--env-file "${REPO_ROOT}/.env")
fi

# shellcheck disable=SC2206
EXTRA_ARR=(${EXTRA_DOCKER_RUN_ARGS-})

exec docker run --rm -it \
  --gpus all \
  --entrypoint python3 \
  -v "${INFERENCE_MODELS_ROOT}:${CONTAINER_WORKSPACE}:ro" \
  -v "${RESULTS_DIR}:${CONTAINER_RESULTS}" \
  -w "${CONTAINER_WORKSPACE}" \
  -e "PYTHONPATH=${CONTAINER_WORKSPACE}" \
  -e "INFERENCE_HOME=${CONTAINER_RESULTS}/inference_home" \
  "${ENV_FILE_ARGS[@]}" \
  "${EXTRA_ARR[@]}" \
  "${IMAGE}" \
  "${CONTAINER_WORKSPACE}/scripts/check_prediction_flakiness.py" \
  --models-config "${CONTAINER_MODELS_CONFIG}" \
  --inference-home "${CONTAINER_RESULTS}/inference_home" \
  --report-json "${CONTAINER_RESULTS}/flakiness_report.json" \
  --streams-output-dir "${CONTAINER_RESULTS}/flakiness_load_streams" \
  --roboflow-images-cache-dir "${CONTAINER_RESULTS}/flakiness_roboflow_images_cache" \
  "$@"
