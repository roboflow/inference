"""Capture/replay benchmark for RF-DETR RLE-to-polygon conversion.

This targets the CPU path in
``inference.core.utils.rle_to_polygon.rle_masks_to_polygons``:

    COCO RLE counts -> sparse crop -> cv2.findContours -> polygon arrays

Default usage captures 100 invocations from the 1080p workflow and immediately
replays them with exact output checks:

    python development/stream_interface/rfdetr_rle_to_poly_microbenchmark.py \
        --video_reference vehicles_1080p.mp4

Replay-only usage:

    python development/stream_interface/rfdetr_rle_to_poly_microbenchmark.py \
        --mode replay --cases-dir temp/rfdetr_rle_to_poly_cases
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
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFERENCE_MODELS_ROOT = _REPO_ROOT / "inference_models"
_WORKFLOW_PATH = (
    _REPO_ROOT / "development" / "stream_interface" / "rfdetr_nano_seg_trt_workflow.py"
)
_TARGET_FUNCTION = "rle_masks_to_polygons"
_SCHEMA_VERSION = 1


def _ensure_local_import_paths() -> None:
    for path in (str(_INFERENCE_MODELS_ROOT), str(_REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_workflow_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rfdetr_nano_seg_trt_workflow_for_rle_to_poly_microbenchmark",
        _WORKFLOW_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load workflow module from {_WORKFLOW_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot_masks(masks: Any) -> dict:
    snapshot = {
        "image_size": tuple(masks.image_size),
        "masks": list(masks.masks),
        "mask_count": len(masks.masks),
    }
    counts = getattr(masks, "_rle_counts_cpu", None)
    lengths = getattr(masks, "_rle_lengths_cpu", None)
    if counts is not None and lengths is not None:
        snapshot["rle_counts_cpu"] = np.array(counts, copy=True)
        snapshot["rle_lengths_cpu"] = np.array(lengths, copy=True)
    return snapshot


def _snapshot_output(output: List[np.ndarray]) -> List[np.ndarray]:
    return [np.array(poly, copy=True) for poly in output]


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
        self.total_masks = 0
        self.lock = threading.Lock()

    def maybe_save(self, masks: Any, output: List[np.ndarray]) -> None:
        with self.lock:
            if self.count >= self.limit:
                return
            case_index = self.count
            mask_snapshot = _snapshot_masks(masks=masks)
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "case_index": case_index,
                "inputs": {"masks": mask_snapshot},
                "expected_output": _snapshot_output(output=output),
            }
            _write_pickle(
                self.cases_dir / f"case_{case_index:04d}.pkl",
                payload,
            )
            self.count += 1
            self.total_masks += mask_snapshot["mask_count"]
            if self.count == 1 or self.count % 10 == 0 or self.count == self.limit:
                print(
                    f"[capture] saved {self.count}/{self.limit} "
                    f"rle-to-poly calls masks={self.total_masks}",
                    flush=True,
                )


def _install_capture_hook(state: _CaptureState) -> None:
    _ensure_local_import_paths()
    from inference.core.models import inference_models_adapters as adapters
    from inference.core.utils import rle_to_polygon

    original = getattr(rle_to_polygon, _TARGET_FUNCTION)

    @functools.wraps(original)
    def wrapper(masks: Any) -> List[np.ndarray]:
        result = original(masks)
        state.maybe_save(masks=masks, output=result)
        return result

    setattr(rle_to_polygon, _TARGET_FUNCTION, wrapper)
    # The adapter imports the function directly at module load time.
    setattr(adapters, _TARGET_FUNCTION, wrapper)


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
            "function": "inference.core.utils.rle_to_polygon.rle_masks_to_polygons",
            "case_count": state.count,
            "total_masks": state.total_masks,
            "video_reference": args.video_reference,
            "backend": args.backend,
            "model_id": model_id,
            "confidence": args.confidence,
            "frames_seen_by_sink": frame_count,
            "capture_elapsed_seconds": elapsed,
            "created_at_unix": time(),
        },
    )
    print(
        f"[capture] wrote {state.count} cases to {cases_dir} "
        f"total_masks={state.total_masks}",
        flush=True,
    )
    return state.count


def _load_case(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError(
            f"{path} has schema_version={payload.get('schema_version')}; "
            f"expected {_SCHEMA_VERSION}."
        )
    return payload


def _materialize_masks(case: dict, use_lazy_counts: bool) -> Any:
    _ensure_local_import_paths()
    from inference_models.models.base.types import InstancesRLEMasks

    payload = case["inputs"]["masks"]
    masks = InstancesRLEMasks(
        image_size=tuple(payload["image_size"]),
        masks=list(payload["masks"]),
    )
    if use_lazy_counts and "rle_counts_cpu" in payload and "rle_lengths_cpu" in payload:
        masks._rle_counts_cpu = np.array(payload["rle_counts_cpu"], copy=True)
        masks._rle_lengths_cpu = np.array(payload["rle_lengths_cpu"], copy=True)
    return masks


def _assert_outputs_equal(
    *,
    actual: List[np.ndarray],
    expected: List[np.ndarray],
    case_index: int,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(
            f"case {case_index}: output length differs: "
            f"{len(actual)} != {len(expected)}"
        )
    for poly_index, (actual_poly, expected_poly) in enumerate(zip(actual, expected)):
        if actual_poly.shape != expected_poly.shape:
            raise AssertionError(
                f"case {case_index} polygon {poly_index}: shape differs "
                f"{actual_poly.shape} != {expected_poly.shape}"
            )
        if actual_poly.dtype != expected_poly.dtype:
            raise AssertionError(
                f"case {case_index} polygon {poly_index}: dtype differs "
                f"{actual_poly.dtype} != {expected_poly.dtype}"
            )
        if not np.array_equal(actual_poly, expected_poly):
            raise AssertionError(
                f"case {case_index} polygon {poly_index}: values differ"
            )


def _run_one_replay_case(*, case_path: Path, use_lazy_counts: bool) -> float:
    from inference.core.utils.rle_to_polygon import rle_masks_to_polygons

    case = _load_case(case_path)
    masks = _materialize_masks(case=case, use_lazy_counts=use_lazy_counts)
    start = perf_counter()
    actual = rle_masks_to_polygons(masks)
    elapsed = perf_counter() - start
    _assert_outputs_equal(
        actual=actual,
        expected=case["expected_output"],
        case_index=case["case_index"],
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
        f"warmup_repeats={args.warmup_repeats} "
        f"use_lazy_counts={args.use_lazy_counts}",
        flush=True,
    )
    for _ in range(args.warmup_repeats):
        for case_path in case_paths:
            _run_one_replay_case(
                case_path=case_path,
                use_lazy_counts=args.use_lazy_counts,
            )

    timings = []
    for repeat_index in range(args.repeats):
        for case_path in case_paths:
            timings.append(
                _run_one_replay_case(
                    case_path=case_path,
                    use_lazy_counts=args.use_lazy_counts,
                )
            )
        print(
            f"[replay] completed repeat {repeat_index + 1}/{args.repeats}",
            flush=True,
        )

    summary = _summarize_timings(timings)
    _print_timing_summary(summary)
    print("[replay] all polygons matched captured e2e outputs", flush=True)
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
        default=Path("temp/rfdetr_rle_to_poly_cases"),
    )
    parser.add_argument("--capture-count", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--use-lazy-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When captured cases include uncompressed RLE counts, restore them "
            "onto the replay masks."
        ),
    )
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
