#!/bin/bash

# Development watch script with SSL enabled
# Note: For development with watchmedo, we disable ENABLE_STREAM_API to avoid port conflicts
# when the server auto-restarts. You can enable it in production deployments.
#
# SSL_CERTIFICATE=INDIRECT will try to download roboflow.host wildcard certificate,
# but will fall back to self-signed certificate if not available.

PROJECT=roboflow-platform \
ENABLE_BUILDER=True \
ENABLE_STREAM_API=True \
ENABLE_SSL=True \
SSL_CERTIFICATE=INDIRECT \
E2B_API_KEY=e2b_4b8b2e92f2fe1c6e0033cde5f1e528f617a5adbd \
E2B_TEMPLATE_ID="qfupheopqmf6w7b36h6o" \
E2B_SANDBOX_TIMEOUT=120 \
E2B_SANDBOX_IDLE_TIMEOUT=120 \
WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=remote \
ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=False \
watchmedo auto-restart --pattern="*.py" --recursive -- uvicorn cpu_http:app --port 9001

# This will start:
# - HTTP server on port 9001
# - HTTPS server on port 9002
#
# Access the server at:
# - http://localhost:9001/
# - https://localhost:9002/ (with self-signed cert warning)
# - https://127-0-0-1.roboflow.host:9002/ (if roboflow.host cert is available)

# Alternative configurations:
# Use self-signed certificate explicitly:
# PROJECT=roboflow-platform ENABLE_BUILDER=True ENABLE_STREAM_API=False ENABLE_SSL=True SSL_CERTIFICATE=GENERATE watchmedo auto-restart --pattern="*.py" --recursive -- python3 start_server.py

# Without SSL (original behavior):
# PROJECT=roboflow-platform ENABLE_BUILDER=True ENABLE_STREAM_API=True watchmedo auto-restart --pattern="*.py" --recursive -- uvicorn cpu_http:app --port 9001
