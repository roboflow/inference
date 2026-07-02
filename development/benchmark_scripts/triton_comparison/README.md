# RF-DETR TRT Triton Comparison

This directory contains two single-server benchmark clients and one Triton
serving helper. Run the server containers separately, collect one JSON result
per server, then compare the JSON outputs.

## Files

- `rfdetr-trt-inference-server-percentiles.py` benchmarks Roboflow Inference
  Server over HTTP.
- `rfdetr-trt-triton-percentiles.py` benchmarks Triton over HTTP. It replays
  RF-DETR preprocessing from the same TRT model package on the client side.
- `rfdetr_triton_server.py` prepares a Triton model repository from a Roboflow
  TRT package and starts `tritonserver`.
- `docker/dockerfiles/Dockerfile.rfdetr.triton` builds the Triton serving image.

## Inputs

Set these paths before running the examples:

```bash
export MODEL_PACKAGE_DIR=/absolute/path/to/rfdetr-small-trt-package
export SOURCE=/absolute/path/to/512x512/images-or-video
export OUTPUT_DIR=/tmp/rfdetr-trt-results
mkdir -p "$OUTPUT_DIR"
```

`MODEL_PACKAGE_DIR` must contain `engine.plan`, `trt_config.json`, and
`inference_config.json`.

## Inference Server Run

Use a stable GPU Inference image and mount this checkout's server code:

```bash
export INFERENCE_IMAGE=roboflow/roboflow-inference-server-gpu:1.3.3
```

Start the server with the local `inference` package and TRT package mounted:

```bash
docker run --rm --gpus all \
  --name rfdetr-inference-server \
  -p 9001:9001 \
  -e ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True \
  -v "$PWD/inference:/app/inference:ro" \
  -v "$MODEL_PACKAGE_DIR:/models/rfdetr-trt-package:ro" \
  "$INFERENCE_IMAGE"
```

The image provides the dependency stack, while the mounted `/app/inference`
directory lets the server run against local code from this checkout.

Alternatively, build a benchmark image from this checkout:

```bash
docker build \
  -f docker/dockerfiles/Dockerfile.onnx.gpu \
  -t roboflow-inference-gpu:rfdetr-benchmark \
  .
```

Then start that image with the TRT package mounted:

```bash
docker run --rm --gpus all \
  --name rfdetr-inference-server \
  -p 9001:9001 \
  -e ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True \
  -v "$MODEL_PACKAGE_DIR:/models/rfdetr-trt-package:ro" \
  roboflow-inference-gpu:rfdetr-benchmark
```

Run the Inference Server benchmark from a Python environment that can import this
repo and has the script dependencies installed:

```bash
python development/benchmark_scripts/triton_comparison/rfdetr-trt-inference-server-percentiles.py \
  --source "$SOURCE" \
  --resize-width 512 \
  --resize-height 512 \
  --model-id /models/rfdetr-trt-package \
  --task object_detection \
  --inference-server-url http://localhost:9001 \
  --result-out "$OUTPUT_DIR/inference-server-detailed.json"
```

If you are benchmarking instance segmentation, use
`--task instance_segmentation`.

## Experimental New Model Manager Server Run

Use the experimental GPU image that contains the new `inference_server` package:

```bash
export NEW_INFERENCE_IMAGE=roboflow/inference-server-experimental:gpu-0.1.3-rc2
```

The helper below starts the container with the MMP-oriented tuning used for this
benchmark, waits for `/v2/server/health`, and then runs the client against the
raw `/infer` endpoint. This is the correct path for local package IDs like
`/models/rfdetr-trt-package`; the structured `/v2/models/infer` path first
resolves model metadata through the Roboflow registry.

```bash
INFERENCE_IMAGE="$NEW_INFERENCE_IMAGE" \
SOURCE="$SOURCE" \
MODEL_PACKAGE_DIR="$MODEL_PACKAGE_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
development/benchmark_scripts/triton_comparison/run-rfdetr-new-inference-server-benchmark.sh
```

By default, the wrapper uses:

```bash
docker run --rm --gpus all --shm-size=1g -p 8000:8000 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PORT=8000 \
  -e NUM_WORKERS=1 \
  -e INFERENCE_DECODER=nvjpeg \
  -e INFERENCE_BATCH_MAX_SIZE=1 \
  -e INFERENCE_BATCH_MAX_WAIT_MS=0 \
  -e INFERENCE_N_SLOTS=8 \
  -e INFERENCE_INPUT_MB=25 \
  -e INFERENCE_LIMIT_CONCURRENCY=8 \
  -e INFERENCE_LOAD_WAIT_S=120 \
  -e INFERENCE_MODEL_IDLE_TIMEOUT_S=31536000 \
  -e INFERENCE_VRAM_IDLE_CUTOFF_S=31536000 \
  -e INFERENCE_VRAM_ADMISSION_CONTROL=false \
  -e INFERENCE_VRAM_RECENT_WINDOW_S=0 \
  -e INFERENCE_ALLOC_TIMEOUT_S=1 \
  -e INFERENCE_WORKER_EMPTY_CACHE_EVERY_N_BATCHES=0 \
  -e INFERENCE_WORKER_EMPTY_CACHE_CHECK_INTERVAL_S=0 \
  -e INFERENCE_HOME=/cache \
  -e INFERENCE_DEPLOYMENT_MODE=mmp \
  -e LOG_LEVEL=WARNING \
  -e ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=True \
  -v /home/damiankosowski/inference/inference_cache:/cache \
  -v "$MODEL_PACKAGE_DIR:/models/rfdetr-trt-package:ro" \
  "$NEW_INFERENCE_IMAGE"
```

The result is written to `$OUTPUT_DIR/new-inference-server.json` unless
`RESULT_OUT` is set. The mounted model package is addressed as
`/models/rfdetr-trt-package`.

For a direct client run without starting Docker, first start the server and then
run:

```bash
python development/benchmark_scripts/triton_comparison/rfdetr-trt-new-inference-server-percentiles.py \
  --source "$SOURCE" \
  --resize-width 512 \
  --resize-height 512 \
  --model-id /models/rfdetr-trt-package \
  --server-url http://localhost:8000 \
  --endpoint-mode raw \
  --result-out "$OUTPUT_DIR/new-inference-server.json"
```

`--endpoint-mode raw` sends `POST /infer?model_id=...&format=json` with raw image
bytes. Use `--endpoint-mode v2-json` only for registry-resolvable model IDs; it
sends `POST /v2/models/infer` with `model_id` in the query string and a JSON body
shaped as `{"inputs": {"image": {"type": "base64", "value": "..."}, "confidence":
0.4}}`.

## Triton Server Run

Build the Triton serving image:

```bash
docker build \
  -f docker/dockerfiles/Dockerfile.rfdetr.triton \
  -t rfdetr-triton-server:benchmark \
  .
```

Start Triton with the same TRT package mounted:

```bash
docker run --rm --gpus all \
  --name rfdetr-triton-server \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -e TRT_ENGINE_HOST_CODE_ALLOWED=True \
  -v "$MODEL_PACKAGE_DIR:/models/rfdetr-trt-package:ro" \
  rfdetr-triton-server:benchmark \
  --model-name rfdetr_small
```

The entrypoint creates `/triton-model-repository/rfdetr_small/1/model.plan`,
writes `config.pbtxt`, then starts `tritonserver`.
For forward-compatible TensorRT engines that include the lean runtime,
`TRT_ENGINE_HOST_CODE_ALLOWED=True` allows local engine inspection and enables
Triton's TensorRT `version-compatible` backend mode. Only enable it for trusted
engine packages.

Run the Triton benchmark from a Python environment that can import this repo's
`inference_models` package and has `tritonclient[http]` installed:

```bash
PYTHONPATH="$PWD:$PWD/inference_models" \
python development/benchmark_scripts/triton_comparison/rfdetr-trt-triton-percentiles.py \
  --source "$SOURCE" \
  --resize-width 512 \
  --resize-height 512 \
  --model-package-dir "$MODEL_PACKAGE_DIR" \
  --triton-url localhost:8000 \
  --triton-model-name rfdetr_small \
  --result-out "$OUTPUT_DIR/triton.json"
```

If Triton reports ambiguous tensor names, pass `--triton-input-name` and one or
more `--triton-output-name` flags.

## Result JSON

Both benchmark scripts write the same top-level shape:

```json
{
  "mode": "run",
  "server": "inference_server",
  "source": "...",
  "resize": {"width": 512, "height": 512},
  "result": {
    "aggregate_fps": 0.0,
    "total_latency_ms": {"p_50": 0.0, "p_95": 0.0, "p_99": 0.0},
    "request_latency_ms": {"p_50": 0.0, "p_95": 0.0, "p_99": 0.0},
    "client_prepare_ms": {"p_50": 0.0, "p_95": 0.0, "p_99": 0.0}
  }
}
```

`total_latency_ms` is the number to compare first. For Triton,
`request_latency_ms` isolates the Triton HTTP call while `client_prepare_ms`
captures RF-DETR preprocessing done by the benchmark client.

## Quick Compare

```bash
python - <<'PY'
import json
from pathlib import Path

output_dir = Path("/tmp/rfdetr-trt-results")
inference = json.loads((output_dir / "inference-server.json").read_text())
triton = json.loads((output_dir / "triton.json").read_text())

inference_p50 = inference["result"]["total_latency_ms"]["p_50"]
triton_p50 = triton["result"]["total_latency_ms"]["p_50"]
ratio = inference_p50 / triton_p50

print(f"inference_server p50 total latency: {inference_p50:.2f} ms")
print(f"triton p50 total latency:           {triton_p50:.2f} ms")
print(f"inference_server / triton:          {ratio:.2f}x")
PY
```

