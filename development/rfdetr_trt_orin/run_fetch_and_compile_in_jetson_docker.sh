#!/usr/bin/env bash
# Run fetch_and_compile_rfdetr_trt_orin.py inside the Jetson 6.2.0 inference image.
#
# Intended to run ON a Jetson Orin host with nvidia-container-runtime configured.
# The image entrypoint starts the inference server; this script overrides it to
# run the compile workflow instead.
#
# Mounts:
#   - development/rfdetr_trt_orin  (live-editable script tree)
#   - inference_models             (required by the compile script; not mounted as
#                                   editable source in the runtime image)
#   - host output directory        (compiled TRT package + logs)
#
# Usage:
#   ./run_fetch_and_compile_in_jetson_docker.sh
#   ./run_fetch_and_compile_in_jetson_docker.sh -- --precision fp16 --verify
#   OUTPUT_DIR=/data/rfdetr-trt ./run_fetch_and_compile_in_jetson_docker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${IMAGE:-roboflow/roboflow-inference-server-jetson-6.2.0:latest}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../rfdetr-seg-nano-orin-trt-build}"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace/inference}"
CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR:-/workspace/output}"
CONTAINER_SCRIPT="${CONTAINER_WORKSPACE}/development/rfdetr_trt_orin/fetch_and_compile_rfdetr_trt_orin.py"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/fetch_and_compile.log}"

MODEL_ID="${MODEL_ID:-rfdetr-seg-nano}"
PROD_API_HOST="${PROD_API_HOST:-https://api.roboflow.com}"
PULL_IMAGE="${PULL_IMAGE:-false}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${PULL_IMAGE}" == "true" ]]; then
    echo "Pulling ${IMAGE} ..."
    docker pull "${IMAGE}"
fi

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "--" ]]; then
        shift
    fi
    EXTRA_ARGS=("$@")
fi

echo "Repo root        : ${REPO_ROOT}"
echo "Script dir mount : ${SCRIPT_DIR}"
echo "Output dir mount : ${OUTPUT_DIR} -> ${CONTAINER_OUTPUT_DIR}"
echo "Docker image     : ${IMAGE}"
echo "Log file         : ${LOG_FILE}"
echo ""

DOCKER_ENV=(
    -e "ROBOFLOW_ENVIRONMENT=prod"
    -e "ROBOFLOW_API_HOST=${PROD_API_HOST}"
    -e "INFERENCE_HOME=${CONTAINER_OUTPUT_DIR}/.cache"
    -e "RUNNING_ON_JETSON=True"
)

if [[ -n "${ROBOFLOW_API_KEY:-}" ]]; then
    DOCKER_ENV+=(-e "ROBOFLOW_API_KEY=${ROBOFLOW_API_KEY}")
fi

PYTHON_ARGS=(
    "${CONTAINER_SCRIPT}"
    --model-id "${MODEL_ID}"
    --output-dir "${CONTAINER_OUTPUT_DIR}"
    --prod-api-host "${PROD_API_HOST}"
)
if ((${#EXTRA_ARGS[@]} > 0)); then
    PYTHON_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Starting container ..."
set +e
docker run --rm \
    --name rfdetr-trt-orin-compile \
    --runtime nvidia \
    --privileged \
    --network bridge \
    -v "${SCRIPT_DIR}:${CONTAINER_WORKSPACE}/development/rfdetr_trt_orin" \
    -v "${REPO_ROOT}/inference_models:${CONTAINER_WORKSPACE}/inference_models" \
    -v "${OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
    -v /tmp:/tmp \
    "${DOCKER_ENV[@]}" \
    --entrypoint python \
    "${IMAGE}" \
    "${PYTHON_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

echo ""
if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo "Compile finished successfully."
    echo "TRT package (host): ${OUTPUT_DIR}/trt_package"
    echo "Full log          : ${LOG_FILE}"
else
    echo "Compile failed with exit code ${EXIT_CODE}." >&2
    echo "Full log: ${LOG_FILE}" >&2
    echo "" >&2
    echo "Last 80 log lines:" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    exit "${EXIT_CODE}"
fi
