"""Benchmark RF-DETR stream optimization with frame percentile reporting.

This script intentionally leaves ``codeflash-optimization-compare.py`` intact.
It imports that baseline benchmark and patches only the timing/reporting pieces
needed to include p50/p95/p99 latency and FPS percentiles.
"""

import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

_BASE_SCRIPT = Path(__file__).with_name("codeflash-optimization-compare.py")


def _load_base_benchmark():
    spec = importlib.util.spec_from_file_location(
        "codeflash_optimization_compare_base",
        _BASE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base benchmark script: {_BASE_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


_benchmark = _load_base_benchmark()
_benchmark._SELF = Path(__file__).resolve()
_benchmark.LAST_PREDICTION_TIME = None
_benchmark.FRAME_INTERVALS = []


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.2f}"


def _format_percentiles(percentiles: dict[str, float | None]) -> str:
    rendered_percentiles = (
        f"p50={_format_optional_float(percentiles['p_50'])} "
        f"p95={_format_optional_float(percentiles['p_95'])} "
        f"p99={_format_optional_float(percentiles['p_99'])}"
    )

    return rendered_percentiles


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None

    ordered_values = sorted(values)
    if len(ordered_values) == 1:
        return ordered_values[0]

    position = (len(ordered_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    fraction = position - lower_index
    interpolated_value = (
        ordered_values[lower_index] * (1.0 - fraction)
        + ordered_values[upper_index] * fraction
    )

    return interpolated_value


def _percentiles(values: list[float]) -> dict[str, float | None]:
    percentiles = {
        "p_50": _percentile(values, 0.50),
        "p_95": _percentile(values, 0.95),
        "p_99": _percentile(values, 0.99),
    }

    return percentiles


def _summarize_frame_intervals(intervals: list[float]) -> tuple[dict, dict]:
    latency_percentiles_ms = _percentiles([interval * 1000.0 for interval in intervals])
    fps_percentiles = _percentiles(
        [1.0 / interval for interval in intervals if interval > 0]
    )

    return latency_percentiles_ms, fps_percentiles


def _write_json(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _build_benchmark_result(
    *,
    profile: str,
    frame_count: int,
    elapsed: float,
    aggregate_fps: float,
    latency_percentiles_ms: dict[str, float | None],
    fps_percentiles: dict[str, float | None],
    latency_reads_ms: list[float],
) -> dict:
    result = {
        "profile": profile,
        "frames": frame_count,
        "elapsed": elapsed,
        "aggregate_fps": aggregate_fps,
        "fps": aggregate_fps,
        "latency_ms": latency_percentiles_ms,
        "fps_percentiles": fps_percentiles,
        "latency_reads_ms": latency_reads_ms,
        "flags": {
            key: _benchmark.os.environ.get(key)
            for key in _benchmark._OPTIMIZATION_FLAG_KEYS
        },
    }

    return result


def _emit_benchmark_result(
    *,
    result: dict,
    result_out: str | None,
) -> None:
    print(
        f"[benchmark] profile={result['profile']} frames={result['frames']} "
        f"elapsed={result['elapsed']:.2f}s aggregate_fps={result['aggregate_fps']:.2f}",
        flush=True,
    )
    print(
        f"[benchmark] latency_ms {_format_percentiles(result['latency_ms'])}",
        flush=True,
    )
    print(
        f"[benchmark] fps_percentiles {_format_percentiles(result['fps_percentiles'])}",
        flush=True,
    )
    print(
        f"[benchmark] latency_reads_ms count={len(result['latency_reads_ms'])}",
        flush=True,
    )
    if result_out is not None:
        _write_json(path=result_out, payload=result)


def sink(predictions, _video_frames) -> None:
    del _video_frames
    if not isinstance(predictions, list):
        predictions = [predictions]

    current_time = perf_counter()
    frame_count = sum(p is not None for p in predictions)
    if frame_count == 0:
        return None

    if _benchmark.START_TIME is None:
        _benchmark.START_TIME = current_time

    previous_time = (
        _benchmark.LAST_PREDICTION_TIME
        if _benchmark.LAST_PREDICTION_TIME is not None
        else _benchmark.START_TIME
    )
    interval = (current_time - previous_time) / frame_count
    _benchmark.FRAME_INTERVALS.extend([interval] * frame_count)
    _benchmark.LAST_PREDICTION_TIME = current_time
    _benchmark.FRAME_COUNT += frame_count

    if _benchmark.FRAME_COUNT % _benchmark.PROGRESS_EVERY == 0:
        fps = _benchmark.FRAME_COUNT / (perf_counter() - _benchmark.START_TIME)
        print(f"[progress] frames={_benchmark.FRAME_COUNT} fps={fps:.2f}", flush=True)


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
    _benchmark.FRAME_COUNT = 0
    _benchmark.FRAME_INTERVALS = []
    _benchmark.LAST_PREDICTION_TIME = None
    _benchmark.START_TIME = None

    print(f"[benchmark] profile={benchmark_profile}", flush=True)
    print(f"[benchmark] flags: {_benchmark._format_optimization_flags()}", flush=True)
    _benchmark._log_compute_environment()

    resolved_local_package = _benchmark._resolve_local_package(
        backend=backend,
        model_id=model_id,
        model_package_id=model_package_id,
        local_package=local_package,
    )
    if resolved_local_package is not None:
        _benchmark.os.environ.setdefault(
            "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES",
            "True",
        )

    workflow_model_id = _benchmark._resolve_model_id(
        model_id=model_id,
        local_package=resolved_local_package,
    )
    if resolved_local_package is not None:
        _benchmark._prepare_local_workflow_model_bundle(
            workflow_model_id=workflow_model_id,
            local_package=resolved_local_package,
        )
        print(
            f"[model] using package via workflow model id: {workflow_model_id}",
            flush=True,
        )

    inference_pipeline = _benchmark._load_inference_pipeline(backend=backend)
    pipeline = inference_pipeline.init_with_workflow(
        video_reference=_benchmark._resolve_video_reference(video_reference),
        workflow_specification=_benchmark.build_workflow(workflow_model_id, confidence),
        on_prediction=sink,
    )
    _benchmark.START_TIME = perf_counter()
    pipeline.start()
    pipeline.join()

    elapsed = perf_counter() - _benchmark.START_TIME if _benchmark.START_TIME else 0.0
    aggregate_fps = _benchmark.FRAME_COUNT / elapsed if elapsed > 0 else 0.0
    latency_percentiles_ms, fps_percentiles = _summarize_frame_intervals(
        intervals=_benchmark.FRAME_INTERVALS,
    )
    latency_reads_ms = [interval * 1000.0 for interval in _benchmark.FRAME_INTERVALS]
    result = _build_benchmark_result(
        profile=benchmark_profile,
        frame_count=_benchmark.FRAME_COUNT,
        elapsed=elapsed,
        aggregate_fps=aggregate_fps,
        latency_percentiles_ms=latency_percentiles_ms,
        fps_percentiles=fps_percentiles,
        latency_reads_ms=latency_reads_ms,
    )
    _emit_benchmark_result(
        result=result,
        result_out=result_out,
    )

    return result


def _result_p50_fps(result: dict) -> float:
    p50_fps = result.get("fps_percentiles", {}).get("p_50")
    if p50_fps is None:
        return 0.0

    return p50_fps


def _print_compare_result(profile: str, result: dict) -> None:
    print(
        f"  {profile:<9} frames={result['frames']} "
        f"elapsed={result['elapsed']:.2f}s "
        f"aggregate_fps={result['aggregate_fps']:.2f}",
        flush=True,
    )
    print(
        f"    latency_ms     {_format_percentiles(result['latency_ms'])}",
        flush=True,
    )
    print(
        f"    fps_percentile {_format_percentiles(result['fps_percentiles'])}",
        flush=True,
    )


def do_compare(
    *,
    video_reference: str,
    model_id: str,
    confidence: float,
    backend: str,
    model_package_id: str | None,
    local_package: str | None,
    result_out: str | None = None,
) -> None:
    resolved_video_reference = _benchmark._resolve_video_reference(video_reference)
    with _benchmark.tempfile.TemporaryDirectory(
        prefix="rfdetr-nano-seg-benchmark-"
    ) as tmp_dir:
        baseline_result_path = str(Path(tmp_dir) / "baseline.json")
        optimized_result_path = str(Path(tmp_dir) / "optimized.json")
        baseline = _benchmark._run_child_benchmark(
            benchmark_profile="baseline",
            flags=_benchmark._BASELINE_FLAGS,
            video_reference=resolved_video_reference,
            model_id=model_id,
            confidence=confidence,
            backend=backend,
            model_package_id=model_package_id,
            local_package=local_package,
            result_out=baseline_result_path,
        )
        optimized = _benchmark._run_child_benchmark(
            benchmark_profile="optimized",
            flags=_benchmark._OPTIMIZED_FLAGS,
            video_reference=resolved_video_reference,
            model_id=model_id,
            confidence=confidence,
            backend=backend,
            model_package_id=model_package_id,
            local_package=local_package,
            result_out=optimized_result_path,
        )

    baseline_p50_fps = _result_p50_fps(baseline)
    optimized_p50_fps = _result_p50_fps(optimized)
    p50_speedup = (
        optimized_p50_fps / baseline_p50_fps if baseline_p50_fps > 0 else 0.0
    )
    print("\n---- compare ----", flush=True)
    _print_compare_result(profile="baseline", result=baseline)
    _print_compare_result(profile="optimized", result=optimized)
    print(f"  p50_fps_speedup {p50_speedup:.2f}x", flush=True)

    if result_out is not None:
        compare_result = {
            "mode": "compare",
            "baseline": baseline,
            "optimized": optimized,
            "p50_fps_speedup": p50_speedup,
        }
        _write_json(path=result_out, payload=compare_result)
        print(f"[benchmark] wrote compare result: {result_out}", flush=True)


def main() -> None:
    parser = _benchmark.argparse.ArgumentParser()
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
    parser.add_argument("--model_id", default=_benchmark._DEFAULT_MODEL_ID)
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
        help="Optional JSON path for the benchmark result.",
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
            result_out=args.result_out,
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


_benchmark.sink = sink
_benchmark.do_run = do_run
_benchmark.do_compare = do_compare
_benchmark.main = main


if __name__ == "__main__":
    main()
