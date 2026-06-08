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
python -m inference_server.server
```

This starts the Model Manager Process (MMP) and uvicorn workers. Models load on first request.

## Run in Docker

Build from the **repo root** (the Dockerfile COPYs `inference_models`, `inference_model_manager`, `inference_server`):

```bash
docker build -f inference_server/docker/Dockerfile.gpu -t inference-server:gpu .
```

Run (GPU):

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=2g \
  -p 8443:8443 \
  -e LOG_LEVEL=INFO \
  inference-server:gpu
```

`--shm-size` is **required**. The MMP allocates a shared-memory pool of
`INFERENCE_N_SLOTS × INFERENCE_INPUT_MB` (defaults: `32 × 25MB ≈ 800MB`) on
`/dev/shm`. Docker's default `/dev/shm` is 64MB, so the pool fails to allocate
and the server exits during startup (locally `OSError: No space left on
device`; in k8s a `SIGBUS`). Set `--shm-size` above the pool size, or shrink
the pool via `INFERENCE_N_SLOTS` / `INFERENCE_INPUT_MB`.

The server listens on plain **HTTP** by default. Set `ENABLE_HTTPS=true` to
serve self-signed TLS (a cert is generated at startup; clients must then skip
verification with `curl --insecure` / `requests verify=False`).

```bash
curl -X POST "http://localhost:8443/infer?model_id=yolov8n-640&format=json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg
```

CPU image: build with `inference_server/docker/Dockerfile.cpu` and drop `--gpus all`.

## Fast infer

```bash
curl -X POST "http://localhost:8443/infer?model_id=yolov8n-640&format=json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8443` | HTTPS port |
| `NUM_WORKERS` | `4` | uvicorn worker processes |
| `LOG_LEVEL` | `INFO` | Log level |
| `ENABLE_HTTPS` | | Set truthy to serve self-signed TLS (default: plain HTTP) |
| `INFERENCE_PRELOAD_MODELS` | | Comma-separated model IDs to load at startup |
| `INFERENCE_N_SLOTS` | `32` | SHM pool slot count |
| `INFERENCE_INPUT_MB` | `25.0` | SHM slot size in MB; pool = `N_SLOTS × INPUT_MB` |
| `INFERENCE_DECODER` | `imagecodecs` | Image decoder (`imagecodecs` or `nvjpeg`) |
| `INFERENCE_BATCH_MAX_SIZE` | `0` | Max batch size (`0` = model default) |
| `INFERENCE_BATCH_MAX_WAIT_MS` | `5.0` | Batch window in ms |
| `API_BASE_URL` | `https://api.roboflow.com` | Roboflow API for auth |
| `DEBUG_BENCHMARK_MODE` | | Set to `1` to skip auth. **Testing only — never use in production.** |
