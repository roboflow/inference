"""Minimal benchmark: RF-DETR instance segmentation through inference-models,
run via InferencePipeline on a single video source.

Workflow has exactly one block — the segmentation model. No annotators, no
buffer strategies, no rate limiting.

The `--backend` flag (trt | onnx | torch) pins the auto-loader by setting
`DISABLED_INFERENCE_MODELS_BACKENDS` to every backend except the chosen one,
so the benchmark numbers correspond unambiguously to a single execution path.

Pass `--local_package` to benchmark against an on-disk package directory (no
registry fetch). Pass `--model_package_id` to download a specific registry
package (cached under `$INFERENCE_HOME/models-cache/`) instead of auto-negotiation.
A TRT package directory in the cwd is still used when neither flag is set.

Use `--mode compare` to run baseline and optimized configurations sequentially
in separate child processes (no interleaving, no parent GPU warmup).

Defaults: rfdetr-seg-nano @ confidence 0.4 on the native TRT backend.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

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
_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFERENCE_MODELS_ROOT = _REPO_ROOT / "inference_models"
_SELF = Path(__file__).resolve()
_PY = sys.executable

_BASELINE_FLAGS = {
    "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND": "false",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED": "false",
    "INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED": "false",
    "RFDETR_PIPELINE_DEPTH": "1",
}
_OPTIMIZED_FLAGS = {
    "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND": "true",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED": "false",
    "INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED": "true",
    "RFDETR_PIPELINE_DEPTH": "2",
}
_OPTIMIZATION_FLAG_KEYS = sorted(
    {*_BASELINE_FLAGS.keys(), *_OPTIMIZED_FLAGS.keys()}
)

FRAME_COUNT = 0
START_TIME = None
PROGRESS_EVERY = 50


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
    preferred = Path.cwd() / _PREFERRED_LOCAL_TRT_PACKAGE
    if _is_local_trt_package(preferred):
        return str(preferred.resolve())

    candidates = sorted(
        path.resolve() for path in Path.cwd().iterdir() if _is_local_trt_package(path)
    )
    if len(candidates) == 1:
        return str(candidates[0])
    return None


def _configure_backend(backend: str) -> None:
    os.environ.setdefault(
        "ONNXRUNTIME_EXECUTION_PROVIDERS",
        "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
    )
    os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
        sorted(_ALL_BACKENDS - {backend})
    )


def _prioritize_repo_imports() -> None:
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


def _load_inference_pipeline(*, backend: str):
    _configure_backend(backend=backend)
    _prioritize_repo_imports()
    spec = importlib.util.spec_from_file_location(
        "inference",
        _REPO_ROOT / "inference" / "__init__.py",
        submodule_search_locations=[str(_REPO_ROOT / "inference")],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load local inference package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["inference"] = module
    spec.loader.exec_module(module)
    return module.InferencePipeline


def _is_valid_package_dir(package_dir: Path) -> bool:
    return package_dir.is_dir() and (package_dir / "model_config.json").is_file()


def _resolve_package_dir_candidate(candidate: Path) -> Path | None:
    if not os.path.lexists(candidate):
        return None
    if candidate.is_symlink():
        try:
            return candidate.resolve()
        except OSError:
            return None
    if candidate.is_dir():
        return candidate.resolve()
    return None


def _try_reuse_materialized_package(model_id: str, package_id: str) -> str | None:
    from inference_models.models.auto_loaders.core import (
        generate_model_package_cache_path,
    )

    candidates = (
        Path(model_id) / "1",
        Path(
            generate_model_package_cache_path(
                model_id=model_id,
                package_id=package_id,
            )
        ),
    )
    for candidate in candidates:
        package_dir = _resolve_package_dir_candidate(candidate)
        if package_dir is not None and _is_valid_package_dir(package_dir):
            return str(package_dir)
    return None


def _ensure_workflow_model_symlink(model_dir: Path, target_dir: Path) -> None:
    """Create or refresh ``model_dir`` as a symlink to ``target_dir`` when needed."""
    resolved_target = target_dir.resolve()
    if model_dir.is_symlink():
        try:
            if model_dir.resolve() == resolved_target:
                return
        except OSError:
            pass
        model_dir.unlink()
    elif model_dir.exists():
        if (
            model_dir.is_dir()
            and _is_valid_package_dir(model_dir)
            and model_dir.resolve() == resolved_target
        ):
            return
        raise RuntimeError(
            f"Workflow model path {model_dir} already exists and is not a symlink "
            f"to {resolved_target}"
        )
    model_dir.symlink_to(resolved_target, target_is_directory=True)


def _registry_model_id_for_fetch(model_id: str) -> str:
    model_dir = Path(model_id)
    if model_dir.is_dir() and not (model_dir / "model_config.json").is_file():
        from inference.models.aliases import resolve_roboflow_model_alias

        return resolve_roboflow_model_alias(model_id=model_id)
    return model_id


def _fetch_model_package(model_id: str, package_id: str, backend: str) -> str:
    from inference_models import AutoModel
    from inference_models.models.auto_loaders.core import (
        generate_model_package_cache_path,
    )

    fetch_model_id = _registry_model_id_for_fetch(model_id)
    package_dirs: list[str] = []

    def capture_package_dir(path: str) -> None:
        package_dirs.append(path)

    AutoModel.from_pretrained(
        model_id_or_path=fetch_model_id,
        backend=backend,
        model_package_id=package_id,
        verbose=True,
        point_model_directory=capture_package_dir,
    )
    if package_dirs:
        return package_dirs[0]

    cache_dir = Path(
        generate_model_package_cache_path(model_id=model_id, package_id=package_id)
    )
    if _is_valid_package_dir(cache_dir):
        return str(cache_dir.resolve())

    raise RuntimeError(
        f"Model package {package_id!r} for {model_id!r} did not report a cache path."
    )


def _resolve_local_package_path(local_package: str) -> str:
    package_path = Path(local_package)
    if not package_path.is_absolute():
        candidate = _REPO_ROOT / package_path
        if candidate.exists():
            package_path = candidate
    package_path = package_path.resolve()
    if not _is_valid_package_dir(package_path):
        raise FileNotFoundError(
            "Local package directory does not contain model_config.json: "
            f"{package_path}"
        )
    return str(package_path)


def _resolve_local_package(
    *,
    backend: str,
    model_id: str,
    model_package_id: str | None,
    local_package: str | None,
) -> str | None:
    if local_package is not None:
        package_dir = _resolve_local_package_path(local_package)
        print(f"[model] using local package from {package_dir}", flush=True)
        return package_dir

    if model_package_id is not None:
        reused_package_dir = _try_reuse_materialized_package(
            model_id=model_id,
            package_id=model_package_id,
        )
        if reused_package_dir is not None:
            print(
                f"[model] reusing package_id={model_package_id} from {reused_package_dir}",
                flush=True,
            )
            return reused_package_dir

        package_dir = _fetch_model_package(
            model_id=model_id,
            package_id=model_package_id,
            backend=backend,
        )
        print(
            f"[model] fetched package_id={model_package_id} from {package_dir}",
            flush=True,
        )
        return package_dir

    if backend == "trt":
        return _find_local_trt_package()

    return None


def _resolve_model_id(model_id: str, local_package: str | None) -> str:
    if local_package is not None:
        return f"{model_id}/1"
    return model_id


def _prepare_local_workflow_model_bundle(
    workflow_model_id: str,
    local_package: str,
) -> None:
    model_dir = Path(workflow_model_id)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    _ensure_workflow_model_symlink(model_dir=model_dir, target_dir=Path(local_package))

    model_cache_dir = (
        Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/cache")) / workflow_model_id
    )
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


def _format_optimization_flags() -> str:
    rendered_flags = [
        f"{key}={os.environ.get(key, '<unset>')}" for key in _OPTIMIZATION_FLAG_KEYS
    ]
    return ", ".join(rendered_flags)


def _log_compute_environment() -> None:
    _prioritize_repo_imports()
    from inference_models import AutoModel

    print("[benchmark] compute environment:", flush=True)
    AutoModel.describe_compute_environment()
    try:
        import triton

        print(f"triton {triton.__version__}", flush=True)
    except ImportError:
        print("triton unavailable", flush=True)


def _emit_benchmark_result(
    *,
    profile: str,
    frame_count: int,
    elapsed: float,
    fps: float,
    result_out: str | None,
) -> None:
    result = {
        "profile": profile,
        "frames": frame_count,
        "elapsed": elapsed,
        "fps": fps,
        "flags": {key: os.environ.get(key) for key in _OPTIMIZATION_FLAG_KEYS},
    }
    print(
        f"[benchmark] profile={profile} frames={frame_count} "
        f"elapsed={elapsed:.2f}s fps={fps:.2f}",
        flush=True,
    )
    if result_out is not None:
        Path(result_out).write_text(json.dumps(result, indent=2))


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


def _resolve_video_reference(video_reference: str) -> str:
    video_path = Path(video_reference)
    if video_path.is_absolute():
        return str(video_path)
    candidate = _REPO_ROOT / video_path
    if candidate.exists():
        return str(candidate.resolve())
    return video_reference


def _child_pythonpath(existing_pythonpath: str | None) -> str:
    entries = [
        str(path)
        for path in (_REPO_ROOT, _INFERENCE_MODELS_ROOT)
        if path.exists()
    ]
    if existing_pythonpath:
        entries.append(existing_pythonpath)
    return os.pathsep.join(entries)


def do_run(
    *,
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
    benchmark_profile: str,
    result_out: str | None,
) -> dict:
    global FRAME_COUNT, START_TIME
    FRAME_COUNT = 0
    START_TIME = None

    print(f"[benchmark] profile={benchmark_profile}", flush=True)
    print(f"[benchmark] flags: {_format_optimization_flags()}", flush=True)
    _log_compute_environment()

    resolved_local_package = _resolve_local_package(
        backend=backend,
        model_id=model_id,
        model_package_id=model_package_id,
        local_package=local_package,
    )
    if resolved_local_package is not None:
        os.environ.setdefault(
            "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES",
            "True",
        )

    workflow_model_id = _resolve_model_id(
        model_id=model_id,
        local_package=resolved_local_package,
    )
    if resolved_local_package is not None:
        _prepare_local_workflow_model_bundle(
            workflow_model_id=workflow_model_id,
            local_package=resolved_local_package,
        )
        print(
            f"[model] using package via workflow model id: {workflow_model_id}",
            flush=True,
        )

    inference_pipeline = _load_inference_pipeline(backend=backend)
    pipeline = inference_pipeline.init_with_workflow(
        video_reference=_resolve_video_reference(video_reference),
        workflow_specification=build_workflow(workflow_model_id, confidence),
        on_prediction=sink,
    )
    pipeline.start()
    pipeline.join()

    elapsed = perf_counter() - START_TIME if START_TIME else 0.0
    fps = FRAME_COUNT / elapsed if elapsed > 0 else 0.0
    _emit_benchmark_result(
        profile=benchmark_profile,
        frame_count=FRAME_COUNT,
        elapsed=elapsed,
        fps=fps,
        result_out=result_out,
    )
    return {
        "profile": benchmark_profile,
        "frames": FRAME_COUNT,
        "elapsed": elapsed,
        "fps": fps,
    }


def _build_child_command(
    *,
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
    benchmark_profile: str,
    result_out: str,
) -> list[str]:
    command = [
        _PY,
        str(_SELF),
        "--mode",
        "run",
        "--video_reference",
        video_reference,
        "--model_id",
        model_id,
        "--confidence",
        str(confidence),
        "--backend",
        backend,
        "--benchmark-profile",
        benchmark_profile,
        "--result-out",
        result_out,
    ]
    if model_package_id is not None:
        command.extend(["--model_package_id", model_package_id])
    if local_package is not None:
        command.extend(["--local_package", local_package])
    return command


def _run_child_benchmark(
    *,
    benchmark_profile: str,
    flags: dict[str, str],
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
    result_out: str,
) -> dict:
    env = os.environ.copy()
    env.update(flags)
    env["PYTHONPATH"] = _child_pythonpath(env.get("PYTHONPATH"))
    command = _build_child_command(
        video_reference=video_reference,
        model_id=model_id,
        confidence=confidence,
        backend=backend,
        model_package_id=model_package_id,
        local_package=local_package,
        benchmark_profile=benchmark_profile,
        result_out=result_out,
    )
    print(
        "\n---- child ----\n"
        f"  profile={benchmark_profile}\n"
        f"  flags={flags}\n"
        f"  result_out={result_out}",
        flush=True,
    )
    subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        env=env,
        check=True,
    )
    return json.loads(Path(result_out).read_text())


def do_compare(
    *,
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
) -> None:
    resolved_video_reference = _resolve_video_reference(video_reference)
    with tempfile.TemporaryDirectory(prefix="rfdetr-nano-seg-benchmark-") as tmp_dir:
        baseline_result_path = str(Path(tmp_dir) / "baseline.json")
        optimized_result_path = str(Path(tmp_dir) / "optimized.json")
        baseline = _run_child_benchmark(
            benchmark_profile="baseline",
            flags=_BASELINE_FLAGS,
            video_reference=resolved_video_reference,
            model_id=model_id,
            confidence=confidence,
            backend=backend,
            model_package_id=model_package_id,
            local_package=local_package,
            result_out=baseline_result_path,
        )
        optimized = _run_child_benchmark(
            benchmark_profile="optimized",
            flags=_OPTIMIZED_FLAGS,
            video_reference=resolved_video_reference,
            model_id=model_id,
            confidence=confidence,
            backend=backend,
            model_package_id=model_package_id,
            local_package=local_package,
            result_out=optimized_result_path,
        )

    baseline_fps = baseline["fps"]
    optimized_fps = optimized["fps"]
    speedup = optimized_fps / baseline_fps if baseline_fps > 0 else 0.0
    print("\n---- compare ----", flush=True)
    print(
        f"  baseline   frames={baseline['frames']} "
        f"elapsed={baseline['elapsed']:.2f}s fps={baseline_fps:.2f}",
        flush=True,
    )
    print(
        f"  optimized  frames={optimized['frames']} "
        f"elapsed={optimized['elapsed']:.2f}s fps={optimized_fps:.2f}",
        flush=True,
    )
    print(f"  speedup    {speedup:.2f}x", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("run", "compare"),
        default="run",
        help=(
            "run: single benchmark in this process. "
            "compare: baseline then optimized, each in a fresh child process."
        ),
    )
    parser.add_argument("--video_reference", required=True)
    parser.add_argument("--model_id", default=_DEFAULT_MODEL_ID)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--backend",
        choices=("trt", "onnx", "torch"),
        default="trt",
        help="inference-models backend.",
    )
    parser.add_argument(
        "--local_package",
        default=None,
        help=(
            "Path to an on-disk model package directory (must contain "
            "model_config.json). Skips registry fetch and cwd TRT discovery."
        ),
    )
    parser.add_argument(
        "--model_package_id",
        default=None,
        help=(
            "Registry package id to download and pin (via inference-models cache). "
            "Overrides auto-negotiation and any cwd TRT package discovery."
        ),
    )
    parser.add_argument(
        "--benchmark-profile",
        default="run",
        help="Label for a single run (compare mode sets baseline/optimized in children).",
    )
    parser.add_argument(
        "--result-out",
        default=None,
        help="Optional JSON path for the final benchmark result (used by compare mode).",
    )
    args = parser.parse_args()

    if args.local_package is not None and args.model_package_id is not None:
        parser.error("--local_package and --model_package_id are mutually exclusive")

    if args.mode == "compare":
        do_compare(
            video_reference=args.video_reference,
            model_id=args.model_id,
            confidence=args.confidence,
            backend=args.backend,
            model_package_id=args.model_package_id,
            local_package=args.local_package,
        )
        return

    do_run(
        video_reference=args.video_reference,
        model_id=args.model_id,
        confidence=args.confidence,
        backend=args.backend,
        model_package_id=args.model_package_id,
        local_package=args.local_package,
        benchmark_profile=args.benchmark_profile,
        result_out=args.result_out,
    )


if __name__ == "__main__":
    main()
