# Pipeline Optimization with Triton Kernels for Preprocessing and Postprocessing

We have introduced high-performance [Triton](https://github.com/triton-lang/triton) kernels that accelerate preprocessing and postprocessing during model inference. We have also introduced pipelining, which interleaves the GPU model forward pass for frame n with preprocessing for frame n+1 and postprocessing for frame n-1. This can improve throughput by 2x-8x depending on the model and input image size.

## Instructions

1. Use this block in your workflow:

    ```python
    "type": "roboflow_core/roboflow_instance_segmentation_model@v3",  # v3 workflow only at the moment
    "name": "segmentation",
    "images": "$inputs.image",
    "model_id": model_id,  # Any RF-DETR instance segmentation model
    "confidence_mode": "custom",
    "custom_confidence": confidence,
    # Required: the optimization runs on the non-dense RLE path.
    "enforce_dense_masks_in_inference_models": False,
    ```


2. Currently, only the **TensorRT** package of **RF-DETR instance segmentation** models is supported.
3. Workflows with static **batch size** of **1** are supported. Blocks like Image Slicer increase the batch size, which disables the optimization.
4. Only `STRETCH_TO` resize mode for input pre-processing is supported at the moment.
5. Use these env vars while running your workflow script:

    ```shell
    ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=true \
    INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true \
    INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED=true \
    RFDETR_PIPELINE_DEPTH=2
    ```

### Script (script.py)

Save this script as `script.py` in the repository root.

```python
"""Minimal benchmark: RF-DETR instance segmentation through inference-models,
run via InferencePipeline on a single video source.

Workflow has exactly one block: the segmentation model. No annotators, no
buffer strategies, no rate limiting.

The example pins the TRT backend before importing `inference` by setting
`DISABLED_INFERENCE_MODELS_BACKENDS` to every backend except TRT, so the
benchmark numbers correspond unambiguously to a single execution path.

Defaults: rfdetr-seg-nano @ confidence 0.4 on the native TRT backend.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

_ALL_BACKENDS = {
    "torch",
    "torch-script",
    "onnx",
    "trt",
    "hugging-face",
    "ultralytics",
    "custom",
}
_DEFAULT_MODEL_ID = "rfdetr-seg-nano"
_PREFERRED_LOCAL_TRT_PACKAGE = "rfdetr-seg-nano-orin-trt-package"
_LOCAL_WORKFLOW_MODEL_ID = f"{_DEFAULT_MODEL_ID}/1"


def _looks_like_repo_root(path: Path) -> bool:
    return (
        (path / "inference" / "__init__.py").is_file()
        and (path / "inference_models").is_dir()
    )


def _find_repo_root() -> Path:
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for candidate in (start, *start.parents):
            if _looks_like_repo_root(candidate):
                return candidate
    raise RuntimeError(
        "Could not locate the inference repository root. Run this script from "
        "the repository root or place it somewhere inside the repository."
    )


_REPO_ROOT = _find_repo_root()
_INFERENCE_MODELS_ROOT = _REPO_ROOT / "inference_models"


def _resolve_local_package_path(path: str) -> Path:
    package_path = Path(path).expanduser()
    if package_path.is_absolute():
        return package_path
    return _REPO_ROOT / package_path


def _is_local_trt_package(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_files = ("engine.plan", "model_config.json", "inference_config.json")
    if not all((path / f).is_file() for f in required_files):
        return False
    try:
        model_config = json.loads((path / "model_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return model_config.get("backend_type") == "trt"


def _find_local_trt_package() -> str | None:
    preferred = _resolve_local_package_path(_PREFERRED_LOCAL_TRT_PACKAGE)
    if _is_local_trt_package(preferred):
        return str(preferred.resolve())

    candidates = sorted(
        path.resolve()
        for path in _REPO_ROOT.iterdir()
        if _is_local_trt_package(path)
    )
    if len(candidates) == 1:
        return str(candidates[0])
    return None


_BACKEND = "trt"
_LOCAL_TRT_PACKAGE = _find_local_trt_package() if _BACKEND == "trt" else None
if _LOCAL_TRT_PACKAGE is not None:
    os.environ.setdefault(
        "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "True"
    )
os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
)
os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
    sorted(_ALL_BACKENDS - {_BACKEND})
)
for path in (str(_INFERENCE_MODELS_ROOT), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
for module_name in list(sys.modules):
    if module_name == "inference" or module_name.startswith("inference."):
        del sys.modules[module_name]
    if (
        module_name == "inference_models"
        or module_name.startswith("inference_models.")
    ):
        del sys.modules[module_name]

from time import perf_counter

_LOCAL_INFERENCE_SPEC = importlib.util.spec_from_file_location(
    "inference",
    _REPO_ROOT / "inference" / "__init__.py",
    submodule_search_locations=[str(_REPO_ROOT / "inference")],
)
if _LOCAL_INFERENCE_SPEC is None or _LOCAL_INFERENCE_SPEC.loader is None:
    raise RuntimeError("Could not load local inference package")
_LOCAL_INFERENCE_MODULE = importlib.util.module_from_spec(_LOCAL_INFERENCE_SPEC)
sys.modules["inference"] = _LOCAL_INFERENCE_MODULE
_LOCAL_INFERENCE_SPEC.loader.exec_module(_LOCAL_INFERENCE_MODULE)
InferencePipeline = _LOCAL_INFERENCE_MODULE.InferencePipeline


def _resolve_model_id(model_id: str, backend: str) -> str:
    if backend == "trt" and model_id == _DEFAULT_MODEL_ID and _LOCAL_TRT_PACKAGE:
        return _LOCAL_WORKFLOW_MODEL_ID
    return model_id


def _prepare_local_workflow_model_bundle(model_id: str) -> None:
    if _LOCAL_TRT_PACKAGE is None or model_id != _LOCAL_WORKFLOW_MODEL_ID:
        return

    model_dir = _REPO_ROOT / model_id
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir = Path(_LOCAL_TRT_PACKAGE)
    if not model_dir.exists():
        model_dir.symlink_to(target_dir, target_is_directory=True)

    model_cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/cache")) / model_id
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    model_type_path = model_cache_dir / "model_type.json"
    model_metadata = {
        "project_task_type": "instance-segmentation",
        "model_type": _DEFAULT_MODEL_ID,
    }
    model_type_path.write_text(json.dumps(model_metadata, indent=4))


def build_workflow(model_id: str, confidence: float) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
                "name": "segmentation",
                "images": "$inputs.image",
                "model_id": model_id,
                "confidence_mode": "custom",
                "custom_confidence": confidence,
                "enforce_dense_masks_in_inference_models": False,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segmentation.predictions",
            },
        ],
    }


FRAME_COUNT = 0
START_TIME = None
PROGRESS_EVERY = 50


def sink(predictions, _video_frames) -> None:
    global FRAME_COUNT, START_TIME
    del _video_frames
    if not isinstance(predictions, list):
        predictions = [predictions]
    FRAME_COUNT += sum(p is not None for p in predictions)
    if START_TIME is None:
        START_TIME = perf_counter()
    if FRAME_COUNT % PROGRESS_EVERY == 0:
        fps = FRAME_COUNT / (perf_counter() - START_TIME)
        print(f"[progress] frames={FRAME_COUNT} fps={fps:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", "--video-reference", required=True)
    parser.add_argument("--model_id", default=_DEFAULT_MODEL_ID)
    parser.add_argument("--confidence", type=float, default=0.4)
    args = parser.parse_args()
    model_id = _resolve_model_id(args.model_id, _BACKEND)
    _prepare_local_workflow_model_bundle(model_id)
    if model_id != args.model_id:
        print(
            f"[model] using local TRT package via workflow model id: {model_id}",
            flush=True,
        )

    pipeline = InferencePipeline.init_with_workflow(
        video_reference=args.video_reference,
        workflow_specification=build_workflow(model_id, args.confidence),
        on_prediction=sink,
    )
    pipeline.start()
    pipeline.join()

    elapsed = perf_counter() - START_TIME if START_TIME else 0.0
    fps = FRAME_COUNT / elapsed if elapsed > 0 else 0.0
    print(f"frames={FRAME_COUNT} elapsed={elapsed:.2f}s fps={fps:.2f}")


if __name__ == "__main__":
    main()
```

Here's an example of a single-step workflow for the `rfdetr-seg-nano` model running with the `trt` backend. To run the script with the Triton kernels and pipelining optimization, run this from the repository root:

```shell
ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=true \
INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true \
INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED=true \
RFDETR_PIPELINE_DEPTH=2 \
python script.py --video-reference path_to_your_video
```

Optional Arguments:
* `--model_id`: set to `rfdetr-seg-nano` by default. Use `rfdetr-seg-small`, `rfdetr-seg-medium`, `rfdetr-seg-large`, `rfdetr-seg-xlarge`, or `rfdetr-seg-2xlarge`.
* `--confidence`: set to `0.4` by default.

Note: you may set `_PREFERRED_LOCAL_TRT_PACKAGE` to the path to your local model build for the `rfdetr-seg` variant you want to test. Relative paths are resolved from the repository root.
