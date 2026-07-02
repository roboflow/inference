#!/usr/bin/env bash
set -euo pipefail

# Start the experimental inference_server image and run the RF-DETR TRT benchmark
# against it. Required inputs:
#
#   SOURCE=/absolute/path/to/video-or-images
#   MODEL_PACKAGE_DIR=/absolute/path/to/rfdetr-trt-package
#
# Optional inputs:
#
#   OUTPUT_DIR=/tmp/rfdetr-trt-results
#   INFERENCE_IMAGE=roboflow/inference-server-experimental:gpu-0.1.3-rc2
#   CONTAINER_NAME=rfdetr-new-inference-server
#   HOST_PORT=8000
#   CONTAINER_PORT=8000
#   SHM_SIZE=1g
#   CACHE_DIR=/home/damiankosowski/inference/inference_cache
#   ROBOFLOW_API_KEY=...
#   NEW_INFERENCE_ENV_FILE=/absolute/path/to/env-file
#   MODEL_ID=/models/rfdetr-trt-package
#   ENDPOINT_MODE=raw  # or v2-json for registry-resolvable model IDs
#   FRAME_LIMIT=...
#   WARMUP=10

if [[ -z "${SOURCE:-}" ]]; then
  echo "SOURCE must be set" >&2
  exit 1
fi

if [[ -z "${MODEL_PACKAGE_DIR:-}" ]]; then
  echo "MODEL_PACKAGE_DIR must be set" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/rfdetr-trt-results}"
INFERENCE_IMAGE="${INFERENCE_IMAGE:-roboflow/inference-server-experimental:gpu-0.1.3-rc2}"
CONTAINER_NAME="${CONTAINER_NAME:-rfdetr-new-inference-server}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
SHM_SIZE="${SHM_SIZE:-1g}"
CACHE_DIR="${CACHE_DIR:-/home/damiankosowski/inference/inference_cache}"
MODEL_ID="${MODEL_ID:-/models/rfdetr-trt-package}"
ENDPOINT_MODE="${ENDPOINT_MODE:-raw}"
WARMUP="${WARMUP:-10}"
RESULT_OUT="${RESULT_OUT:-$OUTPUT_DIR/new-inference-server.json}"
DEBUG_PASSTHROUGH_MODEL="${DEBUG_PASSTHROUGH_MODEL:-1}"
INFERENCE_DECODER="${INFERENCE_DECODER:-nvjpeg}"
ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND="${ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND:-True}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

docker_env_args=(
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}"
  -e "PORT=$CONTAINER_PORT"
  -e "NUM_WORKERS=${NUM_WORKERS:-1}"
  -e "INFERENCE_DECODER=${INFERENCE_DECODER:-nvjpeg}"
  -e "INFERENCE_BATCH_MAX_SIZE=${INFERENCE_BATCH_MAX_SIZE:-1}"
  -e "INFERENCE_BATCH_MAX_WAIT_MS=${INFERENCE_BATCH_MAX_WAIT_MS:-0}"
  -e "INFERENCE_N_SLOTS=${INFERENCE_N_SLOTS:-1}"
  -e "INFERENCE_INPUT_MB=${INFERENCE_INPUT_MB:-25}"
  -e "INFERENCE_LIMIT_CONCURRENCY=${INFERENCE_LIMIT_CONCURRENCY:-2}"
  -e "INFERENCE_LOAD_WAIT_S=${INFERENCE_LOAD_WAIT_S:-120}"
  -e "INFERENCE_MODEL_IDLE_TIMEOUT_S=${INFERENCE_MODEL_IDLE_TIMEOUT_S:-31536000}"
  -e "INFERENCE_VRAM_IDLE_CUTOFF_S=${INFERENCE_VRAM_IDLE_CUTOFF_S:-31536000}"
  -e "INFERENCE_VRAM_ADMISSION_CONTROL=${INFERENCE_VRAM_ADMISSION_CONTROL:-false}"
  -e "INFERENCE_VRAM_RECENT_WINDOW_S=${INFERENCE_VRAM_RECENT_WINDOW_S:-0}"
  -e "INFERENCE_ALLOC_TIMEOUT_S=${INFERENCE_ALLOC_TIMEOUT_S:-1}"
  -e "INFERENCE_WORKER_EMPTY_CACHE_EVERY_N_BATCHES=${INFERENCE_WORKER_EMPTY_CACHE_EVERY_N_BATCHES:-0}"
  -e "INFERENCE_WORKER_EMPTY_CACHE_CHECK_INTERVAL_S=${INFERENCE_WORKER_EMPTY_CACHE_CHECK_INTERVAL_S:-0}"
  -e "INFERENCE_HOME=/cache"
  -e "INFERENCE_DEPLOYMENT_MODE=${INFERENCE_DEPLOYMENT_MODE:-mmp}"
  -e "LOG_LEVEL=${LOG_LEVEL:-WARNING}"
  -e "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=${ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND:-True}"
)

if [[ -n "${ROBOFLOW_API_KEY:-}" ]]; then
  docker_env_args+=(-e "ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY")
fi

if [[ -n "${NEW_INFERENCE_ENV_FILE:-}" ]]; then
  docker_env_args+=(--env-file "$NEW_INFERENCE_ENV_FILE")
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --rm \
  --gpus all \
  --name "$CONTAINER_NAME" \
  --shm-size "$SHM_SIZE" \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  "${docker_env_args[@]}" \
  -v "$CACHE_DIR:/cache" \
  -v "$MODEL_PACKAGE_DIR:/models/rfdetr-trt-package:ro" \
  "$INFERENCE_IMAGE"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[benchmark] waiting for server health on port $HOST_PORT"
for _ in $(seq 1 120); do
  if curl -fsS "http://localhost:$HOST_PORT/v2/server/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "http://localhost:$HOST_PORT/v2/server/health" >/dev/null

benchmark_args=(
  development/benchmark_scripts/triton_comparison/rfdetr-trt-new-inference-server-percentiles.py
  --source "$SOURCE"
  --resize-width 512
  --resize-height 512
  --model-id "$MODEL_ID"
  --server-url "http://localhost:$HOST_PORT"
  --endpoint-mode "$ENDPOINT_MODE"
  --warmup "$WARMUP"
  --result-out "$RESULT_OUT"
)

if [[ -n "${ROBOFLOW_API_KEY:-}" ]]; then
  benchmark_args+=(--api-key "$ROBOFLOW_API_KEY")
fi

if [[ -n "${FRAME_LIMIT:-}" ]]; then
  benchmark_args+=(--frame-limit "$FRAME_LIMIT")
fi

python3 "${benchmark_args[@]}"
