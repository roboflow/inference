"""Capture/replay benchmark for RF-DETR reference preprocessing.

Default usage captures 100 invocations of
``inference_models.models.rfdetr.pre_processing.pre_process_network_input`` from
the e2e workflow and immediately replays them:

    python development/stream_interface/rfdetr_preprocess_microbenchmark.py \
        --video_reference vehicles_1080p.mp4

Replay-only usage:

    python development/stream_interface/rfdetr_preprocess_microbenchmark.py \
        --mode replay --cases-dir temp/rfdetr_preprocess_cases

The TRT RF-DETR model has a Triton fast path that bypasses this function. Capture
mode forces ``INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=false`` before
loading the workflow so the reference preprocessing function is exercised.
"""

import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import threading
from time import perf_counter, time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFERENCE_MODELS_ROOT = _REPO_ROOT / "inference_models"
_WORKFLOW_PATH = (
    _REPO_ROOT / "development" / "stream_interface" / "rfdetr_nano_seg_trt_workflow.py"
)
_TARGET_FUNCTION = "pre_process_network_input"
_SCHEMA_VERSION = 1
_FORCED_PREPROC_ENV = "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED"
_TRITON_REPLAY_STATE: Dict[Tuple[str, int, int, int, int], Any] = {}


def _ensure_local_import_paths() -> None:
    for path in (str(_INFERENCE_MODELS_ROOT), str(_REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_workflow_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rfdetr_nano_seg_trt_workflow_for_preprocess_microbenchmark",
        _WORKFLOW_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load workflow module from {_WORKFLOW_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _snapshot_images(images: Any) -> Any:
    if isinstance(images, torch.Tensor):
        return {"kind": "tensor", "value": _tensor_to_cpu(images)}
    if isinstance(images, np.ndarray):
        return {"kind": "ndarray", "value": np.array(images, copy=True)}
    if isinstance(images, list):
        return {"kind": "list", "value": [_snapshot_images(image) for image in images]}
    return {"kind": "raw", "value": images}


def _materialize_images(payload: Any, device: torch.device) -> Any:
    kind = payload["kind"]
    value = payload["value"]
    if kind == "tensor":
        return value.to(device=device).clone()
    if kind == "ndarray":
        return np.array(value, copy=True)
    if kind == "list":
        return [_materialize_images(image, device=device) for image in value]
    if kind == "raw":
        return value
    raise RuntimeError(f"Unknown image payload kind: {kind}")


def _snapshot_inputs(
    *,
    images: Any,
    image_pre_processing: Any,
    network_input: Any,
    target_device: torch.device,
    input_color_format: Optional[Any],
    image_size_wh: Optional[Union[int, Tuple[int, int]]],
    pre_processing_overrides: Optional[Any],
) -> dict:
    return {
        "images": _snapshot_images(images),
        "image_pre_processing": image_pre_processing,
        "network_input": network_input,
        "target_device": str(target_device),
        "input_color_format": input_color_format,
        "image_size_wh": image_size_wh,
        "pre_processing_overrides": pre_processing_overrides,
    }


def _snapshot_output(output: Tuple[torch.Tensor, List[Any]]) -> dict:
    tensor, metadata = output
    return {
        "tensor": _tensor_to_cpu(tensor),
        "metadata": list(metadata),
    }


def _bind_target_arguments(args: tuple, kwargs: dict) -> dict:
    names = (
        "images",
        "image_pre_processing",
        "network_input",
        "target_device",
        "input_color_format",
        "image_size_wh",
        "pre_processing_overrides",
    )
    bound = {
        "input_color_format": None,
        "image_size_wh": None,
        "pre_processing_overrides": None,
    }
    bound.update(dict(zip(names, args)))
    bound.update(kwargs)
    missing = [
        name
        for name in ("images", "image_pre_processing", "network_input", "target_device")
        if name not in bound
    ]
    if missing:
        raise RuntimeError(f"Cannot capture target call; missing args: {missing}")
    return {name: bound[name] for name in names}


def _write_pickle(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


class _CaptureState:
    def __init__(self, cases_dir: Path, limit: int) -> None:
        self.cases_dir = cases_dir
        self.limit = limit
        self.count = 0
        self.lock = threading.Lock()

    def maybe_save(self, inputs: dict, output: Tuple[torch.Tensor, List[Any]]) -> None:
        with self.lock:
            if self.count >= self.limit:
                return
            case_index = self.count
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "case_index": case_index,
                "inputs": _snapshot_inputs(**inputs),
                "expected_output": _snapshot_output(output),
            }
            _write_pickle(self.cases_dir / f"case_{case_index:04d}.pkl", payload)
            self.count += 1
            if self.count == 1 or self.count % 10 == 0 or self.count == self.limit:
                print(
                    f"[capture] saved {self.count}/{self.limit} preprocess calls",
                    flush=True,
                )


def _install_capture_hook(state: _CaptureState) -> None:
    _ensure_local_import_paths()
    from inference_models.models.rfdetr import pre_processing as rfdetr_pre_processing

    original = getattr(rfdetr_pre_processing, _TARGET_FUNCTION)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, List[Any]]:
        result = original(*args, **kwargs)
        state.maybe_save(inputs=_bind_target_arguments(args, kwargs), output=result)
        return result

    setattr(rfdetr_pre_processing, _TARGET_FUNCTION, wrapper)
    for module_name in (
        "inference_models.models.rfdetr.rfdetr_instance_segmentation_trt",
        "inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx",
        "inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch",
        "inference_models.models.rfdetr.rfdetr_object_detection_trt",
        "inference_models.models.rfdetr.rfdetr_object_detection_onnx",
        "inference_models.models.rfdetr.rfdetr_object_detection_pytorch",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, _TARGET_FUNCTION):
            setattr(module, _TARGET_FUNCTION, wrapper)


def _prepare_cases_dir(cases_dir: Path, overwrite: bool) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    existing = list(cases_dir.glob("case_*.pkl"))
    manifest_path = cases_dir / "manifest.json"
    if not overwrite and (existing or manifest_path.exists()):
        raise RuntimeError(
            f"{cases_dir} already contains captured cases; pass --overwrite "
            "or choose a different --cases-dir."
        )
    if overwrite:
        for path in existing:
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()


def _write_manifest(cases_dir: Path, payload: dict) -> None:
    with (cases_dir / "manifest.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _run_capture(args: argparse.Namespace) -> int:
    if args.force_reference_preprocess:
        os.environ[_FORCED_PREPROC_ENV] = "false"

    cases_dir = args.cases_dir.resolve()
    _prepare_cases_dir(cases_dir=cases_dir, overwrite=args.overwrite)

    workflow = _load_workflow_module()
    model_id = workflow._resolve_model_id(args.model_id, args.backend)
    workflow._prepare_local_workflow_model_bundle(model_id)
    if model_id != args.model_id:
        print(
            f"[model] using local TRT package via workflow model id: {model_id}",
            flush=True,
        )

    state = _CaptureState(cases_dir=cases_dir, limit=args.capture_count)
    _install_capture_hook(state=state)

    frame_count = 0
    start_time: Optional[float] = None
    pipeline_ref: Dict[str, Any] = {}

    def sink(predictions: Any, video_frames: Any) -> None:
        nonlocal frame_count, start_time
        del video_frames
        if not isinstance(predictions, list):
            predictions = [predictions]
        frame_count += sum(p is not None for p in predictions)
        if start_time is None:
            start_time = perf_counter()
        if frame_count % args.progress_every == 0:
            elapsed = perf_counter() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] frames={frame_count} fps={fps:.2f} "
                f"captures={state.count}/{state.limit}",
                flush=True,
            )
        if state.count >= state.limit and "pipeline" in pipeline_ref:
            pipeline_ref["pipeline"].terminate()

    pipeline = workflow.InferencePipeline.init_with_workflow(
        video_reference=args.video_reference,
        workflow_specification=workflow.build_workflow(model_id, args.confidence),
        on_prediction=sink,
    )
    pipeline_ref["pipeline"] = pipeline
    pipeline.start()
    pipeline.join()

    if state.count < args.capture_count:
        raise RuntimeError(
            f"Captured only {state.count}/{args.capture_count} invocations. "
            "Use a longer video or lower --capture-count."
        )

    elapsed = perf_counter() - start_time if start_time else 0.0
    _write_manifest(
        cases_dir=cases_dir,
        payload={
            "schema_version": _SCHEMA_VERSION,
            "function": (
                "inference_models.models.rfdetr.pre_processing."
                "pre_process_network_input"
            ),
            "case_count": state.count,
            "video_reference": args.video_reference,
            "backend": args.backend,
            "model_id": model_id,
            "confidence": args.confidence,
            "frames_seen_by_sink": frame_count,
            "capture_elapsed_seconds": elapsed,
            "created_at_unix": time(),
            "forced_env": (
                f"{_FORCED_PREPROC_ENV}=false"
                if args.force_reference_preprocess
                else None
            ),
        },
    )
    print(f"[capture] wrote {state.count} cases to {cases_dir}", flush=True)
    return state.count


def _resolve_device(device: str, captured: str) -> torch.device:
    if device == "captured":
        return torch.device(captured)
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return resolved


def _materialize_inputs(case: dict, device_override: str) -> dict:
    inputs = case["inputs"]
    target_device = _resolve_device(
        device=device_override,
        captured=inputs["target_device"],
    )
    return {
        "images": _materialize_images(inputs["images"], device=target_device),
        "image_pre_processing": inputs["image_pre_processing"],
        "network_input": inputs["network_input"],
        "target_device": target_device,
        "input_color_format": inputs["input_color_format"],
        "image_size_wh": inputs["image_size_wh"],
        "pre_processing_overrides": inputs["pre_processing_overrides"],
    }


def _uses_enabled(config: Optional[Any]) -> bool:
    return bool(config is not None and config.enabled)


def _run_triton_fast_preprocess(inputs: dict) -> Tuple[torch.Tensor, List[Any]]:
    from inference_models.entities import ImageDimensions
    from inference_models.models.common.roboflow.model_packages import (
        ColorMode,
        PreProcessingMetadata,
        ResizeMode,
        StaticCropOffset,
    )
    from inference_models.models.rfdetr.triton_preprocess import (
        TRITON_AVAILABLE,
        build_resample_tables,
        triton_preprocess_rfdetr_stretch,
    )

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton RF-DETR preprocessing is not available")

    target_device = inputs["target_device"]
    if target_device.type != "cuda":
        raise RuntimeError(
            f"Triton replay requires CUDA target_device, got {target_device}"
        )

    images = inputs["images"]
    if isinstance(images, list):
        if len(images) != 1:
            raise RuntimeError("Triton replay only supports batch size 1")
        candidate = images[0]
    else:
        candidate = images
    if (
        not isinstance(candidate, np.ndarray)
        or candidate.dtype != np.uint8
        or candidate.ndim != 3
        or candidate.shape[2] != 3
    ):
        raise RuntimeError(
            "Triton replay only supports one uint8 HWC ndarray input; "
            f"got type={type(candidate)} shape={getattr(candidate, 'shape', None)}"
        )

    if inputs["image_size_wh"] is not None:
        raise RuntimeError("Triton replay does not support image_size_wh overrides")

    image_pre_processing = inputs["image_pre_processing"]
    if (
        _uses_enabled(image_pre_processing.static_crop)
        or _uses_enabled(image_pre_processing.contrast)
        or _uses_enabled(image_pre_processing.grayscale)
    ):
        raise RuntimeError(
            "Triton replay only supports cases without static crop, contrast, "
            "or grayscale preprocessing"
        )

    network_input = inputs["network_input"]
    if network_input.dataset_version_resize_dimensions is not None:
        raise RuntimeError("Triton replay does not support dataset-version resize")
    if network_input.input_channels != 3:
        raise RuntimeError("Triton replay only supports 3 input channels")
    if network_input.scaling_factor not in (None, 255):
        raise RuntimeError(
            "Triton replay only supports scaling_factor in (None, 255)"
        )
    if network_input.normalization is None:
        raise RuntimeError("Triton replay requires network_input.normalization")
    if network_input.resize_mode not in (
        ResizeMode.STRETCH_TO,
        ResizeMode.LETTERBOX,
        ResizeMode.CENTER_CROP,
        ResizeMode.LETTERBOX_REFLECT_EDGES,
    ):
        raise RuntimeError(
            f"Triton replay does not support resize_mode={network_input.resize_mode}"
        )

    caller_mode = (
        ColorMode(inputs["input_color_format"])
        if inputs["input_color_format"] is not None
        else ColorMode.BGR
    )
    swap_rb = caller_mode != network_input.color_mode

    means, stds = network_input.normalization
    means_t = (float(means[0]), float(means[1]), float(means[2]))
    stds_t = (float(stds[0]), float(stds[1]), float(stds[2]))
    target_h = network_input.training_input_size.height
    target_w = network_input.training_input_size.width
    orig_h, orig_w = int(candidate.shape[0]), int(candidate.shape[1])

    state_key = (str(target_device), orig_h, orig_w, target_h, target_w)
    state = _TRITON_REPLAY_STATE.get(state_key)
    if state is None:
        pinned_host = torch.empty(
            (orig_h, orig_w, 3), dtype=torch.uint8, pin_memory=True
        )
        src_gpu = torch.empty(
            (orig_h, orig_w, 3), dtype=torch.uint8, device=target_device
        )
        out_buffer = torch.empty(
            (1, 3, target_h, target_w), dtype=torch.float32, device=target_device
        )
        tables = build_resample_tables(
            src_h=orig_h,
            src_w=orig_w,
            target_h=target_h,
            target_w=target_w,
            device=target_device,
        )
        state = {
            "pinned_host": pinned_host,
            "src_gpu": src_gpu,
            "out_buffer": out_buffer,
            "tables": tables,
        }
        _TRITON_REPLAY_STATE[state_key] = state

    pinned_np = state["pinned_host"].numpy()
    np.copyto(pinned_np, candidate, casting="no")
    state["src_gpu"].copy_(state["pinned_host"], non_blocking=True)
    triton_preprocess_rfdetr_stretch(
        src=state["src_gpu"],
        tables=state["tables"],
        target_h=target_h,
        target_w=target_w,
        means=means_t,
        stds=stds_t,
        swap_rb=swap_rb,
        out=state["out_buffer"],
    )

    meta = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(width=orig_w, height=orig_h),
        size_after_pre_processing=ImageDimensions(width=orig_w, height=orig_h),
        inference_size=ImageDimensions(width=target_w, height=target_h),
        scale_width=target_w / orig_w,
        scale_height=target_h / orig_h,
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=orig_w,
            crop_height=orig_h,
        ),
    )
    return state["out_buffer"], [meta]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_case(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError(
            f"{path} has schema_version={payload.get('schema_version')}; "
            f"expected {_SCHEMA_VERSION}."
        )
    return payload


def _assert_tensor_equal(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    atol: float,
    rtol: float,
) -> None:
    actual_cpu = actual.detach().cpu()
    if torch.is_floating_point(actual_cpu) and (atol != 0.0 or rtol != 0.0):
        equal = torch.allclose(actual_cpu, expected, atol=atol, rtol=rtol)
    else:
        equal = torch.equal(actual_cpu, expected)
    if not equal:
        max_abs = (
            (actual_cpu - expected).abs().max().item()
            if actual_cpu.shape == expected.shape
            else None
        )
        raise AssertionError(
            f"{label} differs: actual shape={tuple(actual_cpu.shape)} "
            f"expected shape={tuple(expected.shape)} max_abs_diff={max_abs}"
        )


def _assert_outputs_equal(
    *,
    actual: Tuple[torch.Tensor, List[Any]],
    expected: dict,
    case_index: int,
    atol: float,
    rtol: float,
) -> None:
    actual_tensor, actual_metadata = actual
    _assert_tensor_equal(
        actual=actual_tensor,
        expected=expected["tensor"],
        label=f"case {case_index} tensor",
        atol=atol,
        rtol=rtol,
    )
    if list(actual_metadata) != list(expected["metadata"]):
        raise AssertionError(f"case {case_index} metadata differs")


def _run_one_replay_case(
    *,
    case_path: Path,
    device_override: str,
    implementation: str,
    atol: float,
    rtol: float,
) -> float:
    from inference_models.models.rfdetr.pre_processing import pre_process_network_input

    case = _load_case(case_path)
    inputs = _materialize_inputs(case=case, device_override=device_override)
    _synchronize(inputs["target_device"])
    start = perf_counter()
    if implementation == "reference":
        actual = pre_process_network_input(**inputs)
    elif implementation == "triton":
        actual = _run_triton_fast_preprocess(inputs)
    else:
        raise RuntimeError(f"Unknown replay implementation: {implementation}")
    _synchronize(inputs["target_device"])
    elapsed = perf_counter() - start
    _assert_outputs_equal(
        actual=actual,
        expected=case["expected_output"],
        case_index=case["case_index"],
        atol=atol,
        rtol=rtol,
    )
    return elapsed


def _summarize_timings(timings: List[float]) -> dict:
    sorted_timings = sorted(timings)
    total = sum(sorted_timings)
    count = len(sorted_timings)

    def percentile(p: float) -> float:
        if count == 0:
            return 0.0
        index = min(count - 1, int(round((count - 1) * p)))
        return sorted_timings[index]

    return {
        "count": count,
        "total_seconds": total,
        "mean_ms": (total / count) * 1000 if count else 0.0,
        "min_ms": sorted_timings[0] * 1000 if count else 0.0,
        "p50_ms": percentile(0.50) * 1000,
        "p90_ms": percentile(0.90) * 1000,
        "p99_ms": percentile(0.99) * 1000,
        "max_ms": sorted_timings[-1] * 1000 if count else 0.0,
    }


def _print_timing_summary(summary: dict) -> None:
    print(
        "[replay] "
        f"calls={summary['count']} "
        f"total={summary['total_seconds']:.3f}s "
        f"mean={summary['mean_ms']:.3f}ms "
        f"p50={summary['p50_ms']:.3f}ms "
        f"p90={summary['p90_ms']:.3f}ms "
        f"p99={summary['p99_ms']:.3f}ms "
        f"min={summary['min_ms']:.3f}ms "
        f"max={summary['max_ms']:.3f}ms",
        flush=True,
    )


def _run_replay(args: argparse.Namespace) -> dict:
    _ensure_local_import_paths()
    cases_dir = args.cases_dir.resolve()
    case_paths = sorted(cases_dir.glob("case_*.pkl"))
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]
    if not case_paths:
        raise RuntimeError(f"No case_*.pkl files found in {cases_dir}")

    print(
        f"[replay] cases={len(case_paths)} repeats={args.repeats} "
        f"warmup_repeats={args.warmup_repeats} device={args.device} "
        f"implementation={args.replay_implementation}",
        flush=True,
    )
    for _ in range(args.warmup_repeats):
        for case_path in case_paths:
            _run_one_replay_case(
                case_path=case_path,
                device_override=args.device,
                implementation=args.replay_implementation,
                atol=args.atol,
                rtol=args.rtol,
            )

    timings = []
    for repeat_index in range(args.repeats):
        for case_path in case_paths:
            timings.append(
                _run_one_replay_case(
                    case_path=case_path,
                    device_override=args.device,
                    implementation=args.replay_implementation,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
        print(
            f"[replay] completed repeat {repeat_index + 1}/{args.repeats}",
            flush=True,
        )

    summary = _summarize_timings(timings)
    _print_timing_summary(summary)
    print("[replay] all outputs matched captured e2e outputs", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("capture", "replay", "capture-and-replay"),
        default="capture-and-replay",
    )
    parser.add_argument("--video_reference", default="vehicles_1080p.mp4")
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--backend", choices=("trt", "onnx", "torch"), default="trt")
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("temp/rfdetr_preprocess_cases"),
    )
    parser.add_argument("--capture-count", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force-reference-preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Set INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=false before "
            "loading the workflow so pre_process_network_input is called."
        ),
    )
    parser.add_argument(
        "--device",
        default="captured",
        help="'captured', 'auto', 'cpu', or a torch device string used on replay.",
    )
    parser.add_argument(
        "--replay-implementation",
        choices=("reference", "triton"),
        default="reference",
        help="Implementation used by replay; capture always hooks pre_process_network_input.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    if args.capture_count <= 0:
        raise ValueError("--capture-count must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmup_repeats < 0:
        raise ValueError("--warmup-repeats must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.mode in {"capture", "capture-and-replay"}:
        _run_capture(args=args)
    if args.mode in {"replay", "capture-and-replay"}:
        _run_replay(args=args)


if __name__ == "__main__":
    main()
