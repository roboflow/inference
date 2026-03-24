#!/usr/bin/env bash
#
# Start a local OpenTelemetry development stack (Grafana + Tempo + Prometheus + OTel Collector)
# with a pre-configured Inference Server dashboard.
#
# Usage:
#   ./development/otel/start-otel-dev.sh              # start (preserves data across restarts)
#   ./development/otel/start-otel-dev.sh --clean       # start fresh (wipes previous data)
#   ./development/otel/start-otel-dev.sh stop          # stop and remove the container
#
# Then run the inference server with:
#   OTEL_TRACING_ENABLED=True OTEL_EXPORTER_PROTOCOL=http OTEL_EXPORTER_ENDPOINT=localhost:4318 python debugrun.py
#
# Open http://localhost:3000 (admin/admin) to view traces and metrics.

set -euo pipefail

CONTAINER_NAME="inference-otel-dev"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_JSON="${SCRIPT_DIR}/dashboard.json"

if [ "${1:-}" = "stop" ]; then
    echo "Stopping ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    echo "Done."
    exit 0
fi

CLEAN=false
if [ "${1:-}" = "--clean" ]; then
    CLEAN=true
fi

# If --clean, remove the existing container to wipe all data
if [ "$CLEAN" = true ]; then
    echo "Removing existing container and data..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
fi

# Check if already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "${CONTAINER_NAME} is already running."
    echo "  Grafana:    http://localhost:3000 (admin/admin)"
    echo "  OTLP gRPC:  localhost:4317"
    echo "  OTLP HTTP:  localhost:4318"
    echo ""
    echo "Run '$0 stop' to stop it."
    echo "Run '$0 --clean' to restart with fresh data."
    exit 0
fi

# Try to restart a stopped container (preserves data)
if [ "$CLEAN" = false ] && docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Restarting existing ${CONTAINER_NAME} (data preserved)..."
    docker start "${CONTAINER_NAME}"
else
    # Remove stale container if exists
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    echo "Starting OTel dev stack (Grafana + Tempo + Prometheus + Collector)..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p 3000:3000 \
        -p 4317:4317 \
        -p 4318:4318 \
        grafana/otel-lgtm:latest
fi

echo "Waiting for Grafana to be ready..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/api/health | grep -q '200'; then
        break
    fi
    sleep 1
done

# Import the Inference Server dashboard
echo "Importing Inference Server dashboard..."
RESPONSE=$(curl -s -u admin:admin -X POST 'http://localhost:3000/api/dashboards/db' \
    -H 'Content-Type: application/json' \
    -d @"${DASHBOARD_JSON}")

DASH_URL=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('url',''))" 2>/dev/null || echo "")

echo ""
echo "=== OTel Dev Stack Ready ==="
echo ""
echo "  Grafana:    http://localhost:3000 (admin/admin)"
if [ -n "${DASH_URL}" ]; then
echo "  Dashboard:  http://localhost:3000${DASH_URL}"
fi
echo "  OTLP gRPC:  localhost:4317"
echo "  OTLP HTTP:  localhost:4318"
echo ""
echo "Run the inference server with:"
echo "  OTEL_TRACING_ENABLED=True OTEL_EXPORTER_PROTOCOL=http OTEL_EXPORTER_ENDPOINT=localhost:4318 python debugrun.py"
echo ""
echo "Stop with: $0 stop"
