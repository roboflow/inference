# RF-DETR Nano Segmentation TRT Workflow Benchmark

Minimal throughput benchmark for `rfdetr-seg-nano` instance segmentation via
`InferencePipeline` and a single-workflow-block setup. The script counts frames
and reports FPS — no visualization, no prediction export.

Script: [`rfdetr_nano_seg_trt_workflow.py`](rfdetr_nano_seg_trt_workflow.py)

## Single run

Runs one benchmark in the current process. Respects optimization-related
environment variables you set in the shell (see [Manual flag overrides](#manual-flag-overrides)).

```bash
PYTHONPATH=<repo>:<repo>/inference_models \
  python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
    --video_reference <video> \
    --backend trt
```

With a pinned registry package:

```bash
PYTHONPATH=<repo>:<repo>/inference_models \
  python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
    --video_reference <video> \
    --backend trt \
    --model_package_id <your-package-id>
```

With a local on-disk package (no registry fetch):

```bash
PYTHONPATH=<repo>:<repo>/inference_models \
  python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
    --video_reference <video> \
    --backend trt \
    --local_package /path/to/trt-package
```

Relative paths are resolved against the repo root when present there.

## Compare baseline vs optimized

Use `--mode compare` to run **baseline first**, then **optimized**, each in a
fresh child process. The parent never loads inference or touches the GPU, so the
two runs do not interleave and are not warmed up by a parent process.

```bash
PYTHONPATH=<repo>:<repo>/inference_models \
  python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
    --mode compare \
    --video_reference <video> \
    --backend trt
```

With a pinned package:

```bash
python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
  --mode compare \
  --video_reference <video> \
  --backend trt \
  --model_package_id <your-package-id>
```

With a local package:

```bash
python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
  --mode compare \
  --video_reference <video> \
  --backend trt \
  --local_package /path/to/trt-package
```

### Profiles and flags

| Profile | Flags |
|---------|-------|
| **baseline** | `ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=false`, `INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=false`, `INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED=false`, `RFDETR_PIPELINE_DEPTH=1` |
| **optimized** | `ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=true`, `INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true`, `INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED=true`, `RFDETR_PIPELINE_DEPTH=2` |

### Manual flag overrides

For a single run (`--mode run`, the default), set flags yourself instead of using
compare mode:

**Baseline (optimizations off):**

```bash
ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=false \
  INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=false \
  INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED=false \
  RFDETR_PIPELINE_DEPTH=1 \
  PYTHONPATH=<repo>:<repo>/inference_models \
  python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
    --video_reference <video> \
    --backend trt
```

**Optimized (optimizations on):**

```bash
ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=true \
  INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true \
  INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED=true \
  RFDETR_PIPELINE_DEPTH=2 \
  PYTHONPATH=<repo>:<repo>/inference_models \
  python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
    --video_reference <video> \
    --backend trt
```

## Output

Each child run prints its active flags, a compute-environment summary from
`AutoModel.describe_compute_environment()`, progress every 50 frames, then a
final line:

```
[benchmark] profile=baseline
[benchmark] flags: ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=false, ...
[benchmark] compute environment:
                         Compute environment details
...
[benchmark] profile=baseline frames=1200 elapsed=27.85s fps=43.09
```

In compare mode, the driver prints a summary:

```
---- compare ----
  baseline   frames=1200 elapsed=27.85s fps=43.09
  optimized  frames=1200 elapsed=20.12s fps=59.64
  speedup    1.39x
```

## CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `run` | `run`: single benchmark. `compare`: baseline then optimized in separate processes. |
| `--video_reference` | *(required)* | Video file path, RTSP/HTTP URL, or camera device index. |
| `--model_id` | `rfdetr-seg-nano` | Model alias or full Roboflow model id. |
| `--confidence` | `0.4` | Detection confidence threshold. |
| `--backend` | `trt` | `trt`, `onnx`, or `torch` — pins the inference-models backend. |
| `--local_package` | — | Path to an on-disk package directory (`model_config.json` required). Skips registry fetch. Mutually exclusive with `--model_package_id`. |
| `--model_package_id` | — | Registry package id to download and pin. Overrides cwd TRT discovery and auto-negotiation. |
| `--benchmark-profile` | `run` | Label for a single run (compare mode sets `baseline` / `optimized` in children). |
| `--result-out` | — | Optional JSON path for the final benchmark result. |

## Model package resolution

Priority order:

1. **`--local_package`** — use the given directory as-is (must contain `model_config.json`). No registry access.
2. **`--model_package_id`** — download and pin that registry package (cached under `$INFERENCE_HOME/models-cache/`, default `/tmp/cache`), or reuse if already cached.
3. **Local TRT directory in cwd** — `rfdetr-seg-nano-orin-trt-package/`, or any single valid TRT package folder (TRT backend only, when neither flag above is set).
4. **Autoresolve** — registry download and negotiation on first inference.

To list available packages on the target device:

```python
from inference_models import AutoModel

AutoModel.describe_model("rfdetr-seg-nano")
```

## Setup notes

- Run from the inference repo with dependencies installed (CUDA/TRT as needed for `--backend trt`).
- Set `PYTHONPATH` to `<repo>:<repo>/inference_models`, or run from the repo root (the script also prepends those paths).
- TRT engines are hardware-specific — download and benchmark on the target device (e.g. Jetson Orin).
- Use a persistent `INFERENCE_HOME` if you do not want the default `/tmp/cache` cleared on reboot.
- `ROBOFLOW_API_KEY` is only required for private models; `rfdetr-seg-nano` is public.

## Troubleshooting

### `CorruptedModelPackageError: Could not find model config while attempting to load model from local directory`

This can happen in `--mode compare` when the **baseline** child succeeds and the **optimized**
child fails before inference starts.

**Cause:** the baseline run materializes `rfdetr-seg-nano/1` (a symlink to the downloaded TRT
package). That leaves a `rfdetr-seg-nano/` directory in the repo root. The optimized child then
calls `AutoModel.from_pretrained("rfdetr-seg-nano", ...)` to fetch the same
`--model_package_id`. Because `rfdetr-seg-nano/` already exists on disk, the loader treats it as
a local package directory, looks for `rfdetr-seg-nano/model_config.json` (which is only under
`rfdetr-seg-nano/1/`), and fails.

**Fix:** use a current version of the script (it reuses the materialized package / cache path
instead of re-fetching through the colliding alias). As a one-off workaround:

```bash
rm -rf rfdetr-seg-nano
```

Then re-run compare mode.

### `FileExistsError` when creating `rfdetr-seg-nano/1` symlink

**Cause:** `AutoModel.from_pretrained` may already create `rfdetr-seg-nano/1` in the cwd
(often as a symlink into the package cache). The script used `Path.exists()`, which returns
`False` for a **broken** symlink even though the path is present, so `symlink_to()` raises
`FileExistsError`.

**Fix:** use a current version of the script (it uses `os.path.lexists()` / `is_symlink()` and
replaces stale or broken links). As a one-off workaround:

```bash
rm -f rfdetr-seg-nano/1
# or
rm -rf rfdetr-seg-nano
```

### `Using an engine plan file across different models of devices is not supported`

The pinned TRT package was built for a specific GPU (e.g. `nvidia-l4` in the registry
metadata). Running it on a different device may work slowly or unreliably. Pick a
`--model_package_id` that matches your hardware, or use `AutoModel.describe_model("rfdetr-seg-nano")`
on the target machine to see compatible packages.
