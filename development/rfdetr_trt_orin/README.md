# RF-DETR Jetson Orin TRT package workflow

Tools to build a registry-ready TensorRT package for **`rfdetr-seg-nano`** (or another RF-DETR instance-segmentation model) on **Jetson Orin**, then register it on **Roboflow staging**.

The workflow is split into three phases:

| Phase | What | Where to run |
|-------|------|--------------|
| 1 | Fetch ONNX from **production** | Jetson (or inside Jetson Docker) |
| 2 | Compile TRT engine | Jetson Orin (same GPU you deploy on) |
| 3 | Register sealed package | Dev machine with `MODELS_SERVICE_INTERNAL_SECRET` |

Public models like `rfdetr-seg-nano` do not require an API key to **download** ONNX. Staging **registration** uses the internal models-service secret (Roboflow contributors only).

## Files

| File | Purpose |
|------|---------|
| `fetch_and_compile_rfdetr_trt_orin.py` | Phase 1 + 2: download prod ONNX, compile TRT, write registry artefacts |
| `run_fetch_and_compile_in_jetson_docker.sh` | Docker wrapper for phases 1 + 2 using the Jetson 6.2.0 inference image |
| `register_rfdetr_trt_orin_staging.py` | Phase 3: upload and seal the package on staging |
| `_common.py` | Shared constants and manifest helpers |

## Prerequisites

### Jetson compile host

- Jetson Orin (AGX, NX, or Nano) with CUDA CC **8.7**
- JetPack **6.2.x** (L4T r36.4.x) aligned with `docker/dockerfiles/Dockerfile.onnx.jetson.6.2.0`
- `nvidia-container-runtime` if using the Docker wrapper
- Outbound network access to `https://api.roboflow.com` (ONNX download)

### Staging registration

- Repo installed with `inference-cli` (for the models-service client)
- `MODELS_SERVICE_INTERNAL_SECRET` for the staging GCP project
- See `.claude/skills/add-inference-model/SKILL.md` for the full internal registration process

## Quick start (recommended: Docker on Jetson)

From the repo on the Jetson AGX:

```bash
cd development/rfdetr_trt_orin
chmod +x run_fetch_and_compile_in_jetson_docker.sh

# Default output: ../rfdetr-seg-nano-orin-trt-build/
./run_fetch_and_compile_in_jetson_docker.sh
```

AGX has more memory than NX/Nano — you can use higher compile settings:

```bash
./run_fetch_and_compile_in_jetson_docker.sh -- \
  --workspace-size-gb 8 \
  --opt-batch-size 8 \
  --max-batch-size 16 \
  --verify
```

Custom output directory:

```bash
OUTPUT_DIR=/data/rfdetr-trt ./run_fetch_and_compile_in_jetson_docker.sh
```

Use a locally built image:

```bash
IMAGE=roboflow/roboflow-inference-server-jetson-6.2.0:latest \
  ./run_fetch_and_compile_in_jetson_docker.sh
```

The wrapper:

- Overrides the image entrypoint (normally starts the inference server)
- Mounts this directory and `inference_models/` (required — not editable in the runtime image)
- Mounts `OUTPUT_DIR` for artefacts and logs
- Writes a full log to `${OUTPUT_DIR}/fetch_and_compile.log` and prints the last 80 lines on failure

## Quick start (bare metal on Jetson)

```bash
cd development/rfdetr_trt_orin

python fetch_and_compile_rfdetr_trt_orin.py \
  --model-id rfdetr-seg-nano \
  --output-dir ./rfdetr-seg-nano-orin-trt-build \
  --verify
```

## Output layout

After a successful compile:

```
rfdetr-seg-nano-orin-trt-build/
├── fetch_and_compile.log          # when using the Docker wrapper
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
    ├── env-x-ray.json             # compile-time Jetson environment snapshot
    └── registration_manifest.json # metadata for register_rfdetr_trt_orin_staging.py
```

## Phase 3: Register on staging

On a dev machine (not the Jetson):

```bash
export MODELS_SERVICE_INTERNAL_SECRET=$(gcloud secrets versions access latest \
  --secret=MODELS_SERVICE_INTERNAL_SECRET --project=878913763597)

cd development/rfdetr_trt_orin

python register_rfdetr_trt_orin_staging.py \
  --trt-package-dir ../rfdetr-seg-nano-orin-trt-build/trt_package
```

Dry run (validate manifest, no upload):

```bash
python register_rfdetr_trt_orin_staging.py \
  --trt-package-dir ../rfdetr-seg-nano-orin-trt-build/trt_package \
  --dry-run
```

Verify on staging:

```bash
export ROBOFLOW_ENVIRONMENT=staging
export ROBOFLOW_API_HOST=https://api.roboflow.one

python -c "from inference_models import AutoModel; AutoModel.describe_model('rfdetr-seg-nano')"
```

Production registration is a separate step after staging E2E passes (see the add-inference-model skill).

## `fetch_and_compile_rfdetr_trt_orin.py` options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | `rfdetr-seg-nano` | Prod model to fetch |
| `--output-dir` | `./rfdetr-seg-nano-orin-trt-build` | Build output root |
| `--prod-api-host` | `https://api.roboflow.com` | Prod API for ONNX fetch |
| `--precision` | `fp16` | TRT precision |
| `--workspace-size-gb` | `4` | TRT builder workspace (raise on AGX) |
| `--min/opt/max-batch-size` | `1` / `4` / `8` | Dynamic batch profile |
| `--static-batch` | off | Static batch instead of dynamic |
| `--skip-fetch` | off | Reuse existing `source_onnx/` |
| `--skip-compile` | off | Download only |
| `--verify` | off | Smoke test with `RFDetrForInstanceSegmentationTRT.from_pretrained` |
| `--staging-model-id` | `rfdetr-seg-nano` | Model id written into the manifest |

## `register_rfdetr_trt_orin_staging.py` options

| Option | Default | Description |
|--------|---------|-------------|
| `--trt-package-dir` | `./rfdetr-seg-nano-orin-trt-build/trt_package` | Compiled package directory |
| `--staging-api-host` | `https://api.roboflow.one` | Staging API host |
| `--service-secret` | env var | Override `MODELS_SERVICE_INTERNAL_SECRET` |
| `--model-id` | from manifest | Override staging model id |
| `--dry-run` | off | Print payload without uploading |

## Troubleshooting

**TensorRT not detected**

Run on Jetson hardware with JetPack TRT available. Inside Docker, use `--runtime nvidia --privileged` (the wrapper sets both).

**No matching ONNX packages**

RF-DETR seg ONNX uses dynamic batch by default. Do not pass `--static-batch` unless the source package is static.

**Import errors for `compilation` or `inference_models`**

The compile helpers live in `inference_models/development/` (excluded from the published wheel). The fetch script adds that directory to `sys.path` and imports `compilation.core` directly — the same pattern as `inference_models/development/compile_rfdetr.py`.

When using Docker, ensure both `inference_models/` and `inference_models/development/` are on `PYTHONPATH` (the wrapper sets this automatically).

**Registration 409**

A package with the same manifest may already exist on staging. Check existing packages with `AutoModel.describe_model` on staging.

**Engine not portable**

TRT engines are tied to the GPU architecture, L4T version, and TRT version recorded in `registration_manifest.json`. Compile on the same class of device you deploy on.

## Related docs

- [`inference_models/docs/getting-started/trt-compilation.md`](../../inference_models/docs/getting-started/trt-compilation.md) — TRT compilation overview
- [`docker/dockerfiles/Dockerfile.onnx.jetson.6.2.0`](../../docker/dockerfiles/Dockerfile.onnx.jetson.6.2.0) — Jetson 6.2.0 inference image
- [`.claude/skills/add-inference-model/SKILL.md`](../../.claude/skills/add-inference-model/SKILL.md) — internal model registration
