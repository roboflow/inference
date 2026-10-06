"""Capture/replay benchmark for RF-DETR RLE instance-segmentation postprocess.

Default usage captures 100 invocations from the e2e workflow and immediately
replays them:

    python development/stream_interface/rfdetr_rle_postprocess_microbenchmark.py \
        --video_reference vehicles_1080p.mp4

Replay-only usage:

    python development/stream_interface/rfdetr_rle_postprocess_microbenchmark.py \
        --mode replay --cases-dir temp/rfdetr_rle_postprocess_cases
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

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFERENCE_MODELS_ROOT = _REPO_ROOT / "inference_models"
_WORKFLOW_PATH = (
    _REPO_ROOT / "development" / "stream_interface" / "rfdetr_nano_seg_trt_workflow.py"
)
_TARGET_FUNCTION = "post_process_instance_segmentation_results_to_rle_masks"
_SCHEMA_VERSION = 1


def _ensure_local_import_paths() -> None:
    for path in (str(_INFERENCE_MODELS_ROOT), str(_REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_workflow_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rfdetr_nano_seg_trt_workflow_for_microbenchmark",
        _WORKFLOW_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load workflow module from {_WORKFLOW_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _threshold_to_cpu(
    threshold: Union[float, torch.Tensor],
) -> Union[float, torch.Tensor]:
    if isinstance(threshold, torch.Tensor):
        return _tensor_to_cpu(threshold)
    return threshold


def _classes_re_mapping_to_cpu(classes_re_mapping: Optional[Any]) -> Optional[dict]:
    if classes_re_mapping is None:
        return None
    return {
        "remaining_class_ids": _tensor_to_cpu(classes_re_mapping.remaining_class_ids),
        "class_mapping": _tensor_to_cpu(classes_re_mapping.class_mapping),
    }


def _snapshot_inputs(
    *,
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[Any],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[Any],
) -> dict:
    return {
        "bboxes": _tensor_to_cpu(bboxes),
        "logits": _tensor_to_cpu(logits),
        "masks": _tensor_to_cpu(masks),
        "pre_processing_meta": pre_processing_meta,
        "threshold": _threshold_to_cpu(threshold),
        "num_classes": num_classes,
        "classes_re_mapping": _classes_re_mapping_to_cpu(classes_re_mapping),
    }


def _snapshot_mask(mask: Any) -> dict:
    if isinstance(mask, torch.Tensor):
        return {"kind": "dense", "tensor": _tensor_to_cpu(mask)}
    return {
        "kind": "rle",
        "image_size": tuple(mask.image_size),
        "masks": list(mask.masks),
    }


def _snapshot_output(output: List[Any]) -> List[dict]:
    return [
        {
            "xyxy": _tensor_to_cpu(detection.xyxy),
            "class_id": _tensor_to_cpu(detection.class_id),
            "confidence": _tensor_to_cpu(detection.confidence),
            "mask": _snapshot_mask(detection.mask),
            "image_metadata": detection.image_metadata,
            "bboxes_metadata": detection.bboxes_metadata,
        }
        for detection in output
    ]


def _bind_target_arguments(args: tuple, kwargs: dict) -> dict:
    names = (
        "bboxes",
        "logits",
        "masks",
        "pre_processing_meta",
        "threshold",
        "num_classes",
        "classes_re_mapping",
    )
    bound = dict(zip(names, args))
    bound.update(kwargs)
    missing = [name for name in names if name not in bound]
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

    def maybe_save(self, inputs: dict, output: List[Any]) -> None:
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
            _write_pickle(
                self.cases_dir / f"case_{case_index:04d}.pkl",
                payload,
            )
            self.count += 1
            if self.count == 1 or self.count % 10 == 0 or self.count == self.limit:
                print(
                    f"[capture] saved {self.count}/{self.limit} postprocess calls",
                    flush=True,
                )


def _install_capture_hook(state: _CaptureState) -> None:
    _ensure_local_import_paths()
    from inference_models.models.rfdetr import common as rfdetr_common

    original = getattr(rfdetr_common, _TARGET_FUNCTION)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> List[Any]:
        result = original(*args, **kwargs)
        state.maybe_save(inputs=_bind_target_arguments(args, kwargs), output=result)
        return result

    setattr(rfdetr_common, _TARGET_FUNCTION, wrapper)
    for module_name in (
        "inference_models.models.rfdetr.rfdetr_instance_segmentation_trt",
        "inference_models.models.rfdetr.rfdetr_instance_segmentation_onnx",
        "inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch",
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
                "inference_models.models.rfdetr.common."
                "post_process_instance_segmentation_results_to_rle_masks"
            ),
            "case_count": state.count,
            "video_reference": args.video_reference,
            "backend": args.backend,
            "model_id": model_id,
            "confidence": args.confidence,
            "frames_seen_by_sink": frame_count,
            "capture_elapsed_seconds": elapsed,
            "created_at_unix": time(),
        },
    )
    print(f"[capture] wrote {state.count} cases to {cases_dir}", flush=True)
    return state.count


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return resolved


def _to_device(value: Union[float, torch.Tensor], device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device).clone()
    return value


def _classes_re_mapping_to_device(
    payload: Optional[dict], device: torch.device
) -> Optional[Any]:
    if payload is None:
        return None
    from inference_models.models.rfdetr.class_remapping import ClassesReMapping

    return ClassesReMapping(
        remaining_class_ids=payload["remaining_class_ids"].to(device=device).clone(),
        class_mapping=payload["class_mapping"].to(device=device).clone(),
    )


def _materialize_inputs(case: dict, device: torch.device) -> dict:
    inputs = case["inputs"]
    return {
        "bboxes": inputs["bboxes"].to(device=device).clone(),
        "logits": inputs["logits"].to(device=device).clone(),
        "masks": inputs["masks"].to(device=device).clone(),
        "pre_processing_meta": inputs["pre_processing_meta"],
        "threshold": _to_device(inputs["threshold"], device=device),
        "num_classes": inputs["num_classes"],
        "classes_re_mapping": _classes_re_mapping_to_device(
            inputs["classes_re_mapping"],
            device=device,
        ),
    }


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
        raise AssertionError(
            f"{label} differs: actual shape={tuple(actual_cpu.shape)} "
            f"expected shape={tuple(expected.shape)}"
        )


def _assert_mask_equal(
    *,
    actual: Any,
    expected: dict,
    label: str,
    atol: float,
    rtol: float,
) -> None:
    if expected["kind"] == "dense":
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{label} expected dense tensor mask")
        _assert_tensor_equal(
            actual=actual,
            expected=expected["tensor"],
            label=f"{label}.tensor",
            atol=atol,
            rtol=rtol,
        )
        return
    if isinstance(actual, torch.Tensor):
        raise AssertionError(f"{label} expected RLE mask")
    if tuple(actual.image_size) != tuple(expected["image_size"]):
        raise AssertionError(
            f"{label}.image_size differs: {actual.image_size} != "
            f"{expected['image_size']}"
        )
    if list(actual.masks) != list(expected["masks"]):
        raise AssertionError(f"{label}.masks differ")


def _assert_outputs_equal(
    *,
    actual: List[Any],
    expected: List[dict],
    case_index: int,
    atol: float,
    rtol: float,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(
            f"case {case_index}: output length differs: "
            f"{len(actual)} != {len(expected)}"
        )
    for output_index, (actual_detection, expected_detection) in enumerate(
        zip(actual, expected)
    ):
        label = f"case {case_index} output {output_index}"
        _assert_tensor_equal(
            actual=actual_detection.xyxy,
            expected=expected_detection["xyxy"],
            label=f"{label}.xyxy",
            atol=atol,
            rtol=rtol,
        )
        _assert_tensor_equal(
            actual=actual_detection.class_id,
            expected=expected_detection["class_id"],
            label=f"{label}.class_id",
            atol=atol,
            rtol=rtol,
        )
        _assert_tensor_equal(
            actual=actual_detection.confidence,
            expected=expected_detection["confidence"],
            label=f"{label}.confidence",
            atol=atol,
            rtol=rtol,
        )
        _assert_mask_equal(
            actual=actual_detection.mask,
            expected=expected_detection["mask"],
            label=f"{label}.mask",
            atol=atol,
            rtol=rtol,
        )
        if actual_detection.image_metadata != expected_detection["image_metadata"]:
            raise AssertionError(f"{label}.image_metadata differs")
        if actual_detection.bboxes_metadata != expected_detection["bboxes_metadata"]:
            raise AssertionError(f"{label}.bboxes_metadata differs")


def _run_one_replay_case(
    *,
    case_path: Path,
    device: torch.device,
    atol: float,
    rtol: float,
) -> float:
    from inference_models.models.rfdetr.common import (
        post_process_instance_segmentation_results_to_rle_masks,
    )

    case = _load_case(case_path)
    inputs = _materialize_inputs(case=case, device=device)
    _synchronize(device)
    start = perf_counter()
    actual = post_process_instance_segmentation_results_to_rle_masks(**inputs)
    _synchronize(device)
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
    device = _resolve_device(args.device)
    cases_dir = args.cases_dir.resolve()
    case_paths = sorted(cases_dir.glob("case_*.pkl"))
    if args.max_cases is not None:
        case_paths = case_paths[: args.max_cases]
    if not case_paths:
        raise RuntimeError(f"No case_*.pkl files found in {cases_dir}")

    print(
        f"[replay] cases={len(case_paths)} repeats={args.repeats} "
        f"warmup_repeats={args.warmup_repeats} device={device}",
        flush=True,
    )
    for _ in range(args.warmup_repeats):
        for case_path in case_paths:
            _run_one_replay_case(
                case_path=case_path,
                device=device,
                atol=args.atol,
                rtol=args.rtol,
            )

    timings = []
    for repeat_index in range(args.repeats):
        for case_path in case_paths:
            timings.append(
                _run_one_replay_case(
                    case_path=case_path,
                    device=device,
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
        default=Path("temp/rfdetr_rle_postprocess_cases"),
    )
    parser.add_argument("--capture-count", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="auto")
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
