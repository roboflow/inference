#!/usr/bin/env bash
# Launch the Roboflow Inference HTTP server (CPU) for local development.
#
# Runs the same ASGI app the CPU Docker image serves
# (docker/config/cpu_http.py:app) via uvicorn. Static landing-page assets are
# resolved relative to the repository root, so this script always runs from
# there.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/docker/config"
export PROJECT="${PROJECT:-roboflow-platform}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-9001}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export WORKFLOWS_STEP_EXECUTION_MODE="${WORKFLOWS_STEP_EXECUTION_MODE:-local}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/tmp/cache}"
export ENABLE_STREAM_API="${ENABLE_STREAM_API:-False}"
export API_LOGGING_ENABLED="${API_LOGGING_ENABLED:-True}"

exec python3 -m uvicorn cpu_http:app --host "$HOST" --port "$PORT" --workers "$NUM_WORKERS"
