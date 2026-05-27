"""Minimal benchmark: RF-DETR instance segmentation through inference-models,
run via InferencePipeline on a single video source.

Workflow has exactly one block — the segmentation model. No annotators, no
buffer strategies, no rate limiting.

The `--backend` flag (trt | onnx | torch) is parsed before importing
`inference` and pins the auto-loader by setting
`DISABLED_INFERENCE_MODELS_BACKENDS` to every backend except the chosen one,
so the benchmark numbers correspond unambiguously to a single execution path.

When `RFDETR_TRITON_POSTPROC=true`, the script also wires the local TRT package
layout used by the RF-DETR Triton post-processing integration path.

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
    "mediapipe",
    "custom",
}
_DEFAULT_MODEL_ID = "rfdetr-seg-nano"
_PREFERRED_LOCAL_TRT_PACKAGE = "rfdetr-seg-nano-orin-trt-package"
_LOCAL_WORKFLOW_MODEL_ID = f"{_DEFAULT_MODEL_ID}/1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFERENCE_MODELS_ROOT = _REPO_ROOT / "inference_models"


def _str2bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


_RFDETR_TRITON_POSTPROC = _str2bool(os.getenv("RFDETR_TRITON_POSTPROC"))


def _is_local_trt_package(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_files = ("engine.plan", "model_config.json", "inference_config.json")
    if not all((path / file_name).is_file() for file_name in required_files):
        return False
    try:
        model_config = json.loads((path / "model_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return model_config.get("backend_type") == "trt"


def _find_local_trt_package() -> str | None:
    preferred = Path.cwd() / _PREFERRED_LOCAL_TRT_PACKAGE
    if _is_local_trt_package(preferred):
        return str(preferred.resolve())

    candidates = sorted(
        path.resolve() for path in Path.cwd().iterdir() if _is_local_trt_package(path)
    )
    if len(candidates) == 1:
        return str(candidates[0])
    return None


def _select_backend_from_argv() -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=("trt", "onnx", "torch"), default="trt")
    args, _ = pre.parse_known_args()
    return args.backend


_BACKEND = _select_backend_from_argv()
_LOCAL_TRT_PACKAGE = None
if _BACKEND == "trt":
    _LOCAL_TRT_PACKAGE = _find_local_trt_package()
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

if _RFDETR_TRITON_POSTPROC:
    for path in (str(_INFERENCE_MODELS_ROOT), str(_REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    for module_name in list(sys.modules):
        if module_name == "inference" or module_name.startswith("inference."):
            del sys.modules[module_name]
        if module_name == "inference_models" or module_name.startswith(
            "inference_models."
        ):
            del sys.modules[module_name]

from time import perf_counter

if _RFDETR_TRITON_POSTPROC:
    local_inference_spec = importlib.util.spec_from_file_location(
        "inference",
        _REPO_ROOT / "inference" / "__init__.py",
        submodule_search_locations=[str(_REPO_ROOT / "inference")],
    )
    if local_inference_spec is None or local_inference_spec.loader is None:
        raise RuntimeError("Could not load local inference package")
    local_inference_module = importlib.util.module_from_spec(local_inference_spec)
    sys.modules["inference"] = local_inference_module
    local_inference_spec.loader.exec_module(local_inference_module)
    InferencePipeline = local_inference_module.InferencePipeline
else:
    from inference import InferencePipeline


def _resolve_model_id(model_id: str, backend: str) -> str:
    if (
        backend == "trt"
        and model_id == _DEFAULT_MODEL_ID
        and _LOCAL_TRT_PACKAGE
    ):
        return _LOCAL_WORKFLOW_MODEL_ID
    return model_id


def _prepare_local_workflow_model_bundle(model_id: str) -> None:
    if _LOCAL_TRT_PACKAGE is None or model_id != _LOCAL_WORKFLOW_MODEL_ID:
        return

    model_dir = Path(model_id)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir = Path(_LOCAL_TRT_PACKAGE)
    if not model_dir.exists():
        model_dir.symlink_to(target_dir, target_is_directory=True)

    model_cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/cache")) / model_id
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    model_type_path = model_cache_dir / "model_type.json"
    model_metadata = {
        "project_task_type": "instance-segmentation",
        "model_type": "rfdetr-seg-nano",
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
    parser.add_argument("--video_reference", required=True)
    parser.add_argument("--model_id", default=_DEFAULT_MODEL_ID)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--backend",
        choices=("trt", "onnx", "torch"),
        default="trt",
        help="inference-models backend (consumed pre-import via env var).",
    )
    args = parser.parse_args()
    model_id = _resolve_model_id(args.model_id, args.backend)
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
