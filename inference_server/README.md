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

## Fast infer

```bash
curl -X POST "http://localhost:8000/infer?model_id=yolov8n-640&format=json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | HTTP port |
| `NUM_WORKERS` | `4` | uvicorn worker processes |
| `INFERENCE_PRELOAD_MODELS` | | Comma-separated model IDs to load at startup |
| `INFERENCE_N_SLOTS` | `256` | SHM pool slot count |
| `INFERENCE_DECODER` | `imagecodecs` | Image decoder (`imagecodecs` or `nvjpeg`) |
| `INFERENCE_BATCH_MAX_SIZE` | `0` | Max batch size (`0` = model default) |
| `INFERENCE_BATCH_MAX_WAIT_MS` | `5.0` | Batch window in ms |
| `API_BASE_URL` | `https://api.roboflow.com` | Roboflow API for auth |
| `DEBUG_BENCHMARK_MODE` | | Set to `1` to skip auth. **Testing only — never use in production.** |
