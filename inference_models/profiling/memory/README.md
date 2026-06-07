# GPU memory profiling (`profiling.memory`)

Profiles `inference_models` registry classes on CUDA and emits normalized
`MemoryProfileRecord` JSON for future model-manager GPU admission. Each run
measures idle memory after load and peak memory during a synthetic inference
loop, together with structured metadata about the model, package, inputs, and
environment.

## Install

From the `inference_models` package root:

```bash
uv sync --extra profiling-memory
# plus a CUDA torch/onnxruntime/tensorrt stack as needed for the backend you profile
```

Set `ROBOFLOW_API_KEY` when resolving packages from Roboflow.

## CLI entry points

All three console scripts invoke the same CLI; pass `--backend` to select the
harness:

| Script | Typical `--backend` |
|--------|---------------------|
| `memory-profile-pytorch` | `torch` (also torch-script and Hugging Face registry rows) |
| `memory-profile-onnx` | `onnx` |
| `memory-profile-trt` | `trt` |

Equivalent module invocation:

```bash
uv run python -m profiling.memory.cli profile --backend onnx ...
```

## Quick start

List registry rows for a harness:

```bash
uv run memory-profile-onnx --list-onnx-models
```

Profile a registered model (package resolved under `--packages-target-dir`):

```bash
uv run memory-profile-onnx profile \
  --model-id workspace/yolov8n-640 \
  --architecture yolov8 \
  --task-type object-detection \
  --backend onnx \
  --quantization fp32 \
  --profile-tier customer
```

Inspect resolved inputs without loading the GPU (`--dry-run`):

```bash
uv run memory-profile-pytorch profile \
  --model-id workspace/my-vlm \
  --architecture gemma-4 \
  --task-type vlm \
  --backend torch \
  --quantization fp32 \
  --dry-run
```

## What the CLI resolves automatically

1. **Registry class** — from `--architecture`, `--task-type`, and `--backend`
   ([`backend_registry.py`](backend_registry.py)).
2. **Package directory** — from `--model-id` via the weights provider into
   `{packages-target-dir}/{model_id}/{package_id}/` ([`package_resolve.py`](package_resolve.py)).
   Local cache is reused unless the directory is missing or `--force-download` is set.
3. **Image shapes** — batch / height / width from package artifacts
   (`inference_config.json`, `trt_config.json`), falling back to 1×640×640
   ([`package_input_profile.py`](package_input_profile.py)).
4. **Method and non-image inputs** — from the registry task profile in
   [`registry_input_profiles.json`](registry_input_profiles.json) via
   [`profiling_inputs.py`](profiling_inputs.py) (e.g. VLM `prompt`,
   `max_new_tokens`; open-vocabulary `classes`). Override with
   `--infer-kwargs-json` or `--method`.

## Useful flags

| Flag | Purpose |
|------|---------|
| `--dry-run` | Print the worker payload JSON and exit |
| `--profile-tier` | Label the record: `customer`, `registry_template`, or `validation` |
| `--force-download` | Re-fetch package artifacts even when cached locally |
| `--package-id` | Pin a specific Roboflow package version |
| `--output-json` | Write the profile record to a file |
| `--infer-kwargs-json` | Override registry-driven infer kwargs |

## Helper scripts

```bash
# Regenerate registry_input_profiles.json after REGISTERED_MODELS changes
uv run python profiling/memory/scripts/generate_registry_input_profiles.py

# Inspect shape constraints and resolved inputs for a package
uv run python profiling/memory/scripts/inspect_package_input_profile.py \
  --model-id … --architecture … --task-type … --backend onnx

# Download a package without profiling
uv run python profiling/scripts/fetch_model_package.py --model-id … ...
```

## Package layout

| Path | Role |
|------|------|
| `cli.py` | Orchestration, payload build, subprocess workers |
| `workers/{torch,onnx,trt}.py` | Per-backend measurement harnesses |
| `metadata.py` | `MemoryProfileRecord` schema and assembly |
| `registry_input_profiles.json` | Input contracts linked to registry entries |
| `package_input_profile.py` | Shape resolution and validation |
| `profiling_inputs.py` | Task-profile method and infer-kwargs defaults |
| `docs/` | Design notes and admission context |

## Output

Profiling prints a `MemoryProfileRecord` (schema `1.1`) to stdout: backend-specific
**metrics** plus **model_metadata**, **runtime_metadata**, **backend_metadata**,
**input_metadata**, **environment_metadata**, and **profiling_run**.

See [`docs/description.md`](docs/description.md) for admission metrics and runtime
workflows, and [`docs/input_profiles_and_package_tiers.md`](docs/input_profiles_and_package_tiers.md)
for how input profiles and profile tiers relate to packages and harness behavior.
