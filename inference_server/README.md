# inference-server

HTTP server for model inference. Wraps `inference-model-manager` with FastAPI endpoints.

## Install

Requires Python 3.10–3.12. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install uv

# For development: install inference-models and inference-model-manager editable first
# uv pip install -e "../inference_models"
# uv pip install -e "../inference_model_manager"

# CPU (torch + ONNX)
uv pip install -e ".[torch-cpu,onnx-cpu]"

# CUDA 12.4
uv pip install -e ".[torch-cu124,onnx-cu12]"
```

Extras cascade through `inference-model-manager` to `inference-models`.

## Quick start

```bash
python -m inference_server.app
```

Models load on first request via a direct in-process `ModelManager`.

## Run in Docker

Build from the **repo root** (the Dockerfile COPYs `inference_models`, `inference_model_manager`, `inference_server`):

```bash
docker build -f inference_server/docker/Dockerfile.cpu -t inference-server:cpu .
```

Run:

```bash
docker run --rm -it \
  -p 8000:8000 \
  inference-server:cpu
```

```bash
curl -X POST "http://localhost:8000/v2/models/infer?model_id=yolov8n-640" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg
```

## Extension points

`inference-server` and `inference-model-manager` resolve additional
implementations via entry points, so a separate package can extend either
without a code change here:

| Entry-point group | Resolves | Selected via |
|---|---|---|
| `inference_server.gateway` | Alternative `resolve_gateway()` targets | `INFERENCE_GATEWAY` env var |
| `inference_model_manager.backends` | Alternative `ModelManager.load(backend=...)` implementations | `backend=` kwarg |
| `inference_model_manager.decoders` | Alternative image decoders | decoder name |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | HTTP port (`__main__` dev runner) |
| `NUM_WORKERS` | `1` | uvicorn worker processes (`__main__` dev runner) |
| `INFERENCE_GATEWAY` | `direct` | Gateway resolved by `gateway_resolver.resolve_gateway()` |
| `INFERENCE_PRELOAD_MODELS` | | Comma-separated model IDs loaded at server startup; `/v2/server/ready` reports not-ready until each finishes loading |
| `INFERENCE_LOAD_WAIT_S` | `10.0` | Seconds `ensure_loaded()` waits before reporting a load timeout |
| `INFERENCE_INFER_TIMEOUT_S` | `30.0` | Per-request inference timeout |
| `INFERENCE_MAX_BODY_BYTES` | `100MB` | Max request body / aggregate URL-image size |
| `INFERENCE_MAX_IMAGES_PER_REQUEST` | `32` | Max images per request (body, multipart, or `?image=<url>` params) |
| `API_BASE_URL` | `https://api.roboflow.com` | Roboflow API for auth |
| `ENABLE_CONTROL_PLANE_ROUTES` | `false` | Enables model list/load/unload and server info/metrics routes; they accept any valid key without workspace scoping |
| `INFERENCE_MAX_ACTIVE_MODELS` | `8` | Max concurrently loaded models (direct gateway); LRU drain-unload on overflow, `<=0` unbounded |
| `INFERENCE_MEMORY_FREE_THRESHOLD` | `0` | Free-VRAM fraction below which loads evict LRU models first; `0` off |
| `PRELOAD_API_KEY` | | API key for `INFERENCE_PRELOAD_MODELS` startup loads |
| `MAX_INFERENCE_MODELS_CACHE_SIZE_MB` | `-1` | Disk cache watchdog size cap; `-1` off |
| `INFERENCE_MODELS_CACHE_WATCHDOG_INTERVAL_MINUTES` | `60` | Disk cache watchdog interval |
| `ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG` | `false` | Periodic CUDA cache reclamation daemon |
| `CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS` | `300` | Reclamation interval |
