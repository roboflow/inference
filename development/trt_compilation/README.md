# Local TensorRT model compilation

Tools to fetch an ONNX model package from the Roboflow registry, compile a TensorRT engine on **local hardware**, and optionally register the sealed package on **Roboflow staging**.

Works for any model with an ONNX backend and TRT runtime support — not tied to a specific architecture or device class. Compile on the same class of hardware you deploy on (Jetson Orin, GPU server, etc.).

The workflow is split into three phases:

| Phase | What | Where to run |
|-------|------|--------------|
| 1 | Fetch ONNX from **production** | Compile host (Jetson or GPU server) |
| 2 | Compile TRT engine | Same machine as deployment target |
| 3 | Register sealed package | Dev machine with `MODELS_SERVICE_INTERNAL_SECRET` |

Public models do not require an API key to **download** ONNX. Staging **registration** uses the internal models-service secret (Roboflow contributors only).

## Files

| File | Purpose |
|------|---------|
| `fetch_and_compile_trt.py` | Phase 1 + 2: download prod ONNX, compile TRT, write registry artefacts |
| `run_in_docker.sh` | Docker wrapper for phases 1 + 2 using an inference image |
| `register_trt_staging.py` | Phase 3: upload and seal the package on staging |
| `_common.py` | Shared constants and manifest helpers |

## Prerequisites

### Compile host

- NVIDIA hardware with TensorRT (Jetson or GPU server)
- Inference image or bare-metal environment aligned with your deployment target
- `nvidia-container-runtime` if using the Docker wrapper
- Outbound network access to `https://api.roboflow.com` (ONNX download)

### Staging registration

- Repo installed with `inference-cli` (for the models-service client)
- `MODELS_SERVICE_INTERNAL_SECRET` for the staging GCP project
- See `.claude/skills/add-inference-model/SKILL.md` for the full internal registration process

## Quick start (Docker on Jetson)

From the repo on a Jetson Orin:

```bash
cd development/trt_compilation
chmod +x run_in_docker.sh

MODEL_ID=rfdetr-seg-nano ./run_in_docker.sh
```

AGX has more memory than NX/Nano — you can use higher compile settings:

```bash
MODEL_ID=rfdetr-seg-nano ./run_in_docker.sh -- \
  --workspace-size-gb 8 \
  --opt-batch-size 8 \
  --max-batch-size 16 \
  --verify
```

Custom output directory:

```bash
MODEL_ID=rfdetr-seg-nano OUTPUT_DIR=/data/trt-build ./run_in_docker.sh
```

Use a locally built Jetson image:

```bash
MODEL_ID=rfdetr-seg-nano \
  IMAGE=roboflow/roboflow-inference-server-jetson-6.2.0:latest \
  ./run_in_docker.sh
```

## Quick start (Docker on GPU server)

```bash
MODEL_ID=yolov8n-640 \
  IMAGE=roboflow/roboflow-inference-server-gpu:latest \
  ./run_in_docker.sh -- --verify
```

Pick an inference image whose CUDA/TRT versions match your deployment environment.

## Quick start (bare metal)

```bash
cd development/trt_compilation

python fetch_and_compile_trt.py \
  --model-id rfdetr-seg-nano \
  --verify
```

When multiple ONNX packages match, list available ids from the error message and pass `--package-id`:

```bash
python fetch_and_compile_trt.py \
  --model-id my-model \
  --package-id abc123-onnx-fp32
```

## Output layout

After a successful compile:

```
{model-id}-trt-build/
├── fetch_and_compile.log          # when using the Docker wrapper
├── source_metadata.json           # model metadata from fetch (used by --skip-fetch)
├── source_onnx/
│   └── <package-id>/              # downloaded prod ONNX package
│       ├── weights.onnx
│       ├── inference_config.json
│       └── class_names.txt
└── trt_package/                   # upload this directory to staging
    ├── engine.plan
    ├── trt_config.json
    ├── inference_config.json      # adjusted for TRT (static spatial size)
    ├── class_names.txt
    ├── env-x-ray.json             # compile-time environment snapshot
    ├── registration_manifest.json # metadata for register_trt_staging.py
    └── model_config.json          # optional (--write-model-config); local use only
```

`model_config.json` is **not** part of the sealed registry package. When loading via `AutoModel.from_pretrained("my-model")`, the library generates it in the cache from registry metadata. Pass `--write-model-config` only if you want to load the folder directly:

```python
AutoModel.from_pretrained("/path/to/trt_package", backend="trt")
```

## Phase 3: Register on staging

On a dev machine (not the compile host):

```bash
export MODELS_SERVICE_INTERNAL_SECRET=$(gcloud secrets versions access latest \
  --secret=MODELS_SERVICE_INTERNAL_SECRET --project=878913763597)

cd development/trt_compilation

python register_trt_staging.py \
  --trt-package-dir ../rfdetr-seg-nano-trt-build/trt_package
```

Dry run (validate manifest, no upload):

```bash
python register_trt_staging.py \
  --trt-package-dir ../rfdetr-seg-nano-trt-build/trt_package \
  --dry-run
```

Verify on staging:

```bash
export ROBOFLOW_ENVIRONMENT=staging
export ROBOFLOW_API_HOST=https://api.roboflow.one

python -c "from inference_models import AutoModel; AutoModel.describe_model('rfdetr-seg-nano')"
```

Production registration is a separate step after staging E2E passes (see the add-inference-model skill).

## `fetch_and_compile_trt.py` options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | *(required)* | Prod model to fetch |
| `--output-dir` | `./{model-id}-trt-build` | Build output root |
| `--package-id` | — | Select one ONNX package when multiple match |
| `--prod-api-host` | `https://api.roboflow.com` | Prod API for ONNX fetch |
| `--precision` | `fp16` | TRT precision |
| `--workspace-size-gb` | `4` | TRT builder workspace (raise on high-memory hosts) |
| `--min/opt/max-batch-size` | `1` / `4` / `8` | Dynamic batch profile |
| `--static-batch` | off | Static batch instead of dynamic |
| `--skip-fetch` | off | Reuse existing `source_onnx/` and `source_metadata.json` |
| `--skip-compile` | off | Download only |
| `--verify` | off | Smoke test with `AutoModel.from_pretrained(..., backend="trt")` |
| `--write-model-config` | off | Write local-only `model_config.json` (not registered on staging) |
| `--staging-model-id` | same as `--model-id` | Model id written into the manifest |

## `register_trt_staging.py` options

| Option | Default | Description |
|--------|---------|-------------|
| `--trt-package-dir` | *(required)* | Compiled package directory |
| `--staging-api-host` | `https://api.roboflow.one` | Staging API host |
| `--service-secret` | env var | Override `MODELS_SERVICE_INTERNAL_SECRET` |
| `--model-id` | from manifest | Override staging model id |
| `--dry-run` | off | Print payload without uploading |

## Troubleshooting

**TensorRT not detected**

Run on hardware with TensorRT available. Inside Docker, use `--runtime nvidia --privileged` (the wrapper sets both).

**No matching ONNX packages**

Check that the model has an ONNX package on production. For dynamic-batch ONNX, do not pass `--static-batch` unless the source package is static.

**Multiple matching ONNX packages**

Pass `--package-id` with one of the ids listed in the error message.

**Import errors for `compilation` or `inference_models`**

The compile helpers live in `inference_models/development/` (excluded from the published wheel). The fetch script adds that directory to `sys.path` and imports `compilation.core` directly.

When using Docker, ensure both `inference_models/` and `inference_models/development/` are on `PYTHONPATH` (the wrapper sets this automatically).

**Registration 409**

A package with the same manifest may already exist on staging. Check existing packages with `AutoModel.describe_model` on staging.

**Engine not portable**

TRT engines are tied to the GPU architecture, driver/L4T version, and TRT version recorded in `registration_manifest.json`. Compile on the same class of device you deploy on.

## Related docs

- [`inference_models/docs/getting-started/trt-compilation.md`](../../inference_models/docs/getting-started/trt-compilation.md) — TRT compilation overview
- [`docker/dockerfiles/`](../../docker/dockerfiles/) — inference images for Jetson and GPU
- [`.claude/skills/add-inference-model/SKILL.md`](../../.claude/skills/add-inference-model/SKILL.md) — internal model registration

## Migration from `rfdetr_trt_orin/`

The previous RF-DETR Orin-specific directory has been replaced by this generic tool:

| Old | New |
|-----|-----|
| `development/rfdetr_trt_orin/` | `development/trt_compilation/` |
| `fetch_and_compile_rfdetr_trt_orin.py` | `fetch_and_compile_trt.py` |
| `register_rfdetr_trt_orin_staging.py` | `register_trt_staging.py` |
| `run_fetch_and_compile_in_jetson_docker.sh` | `run_in_docker.sh` |
