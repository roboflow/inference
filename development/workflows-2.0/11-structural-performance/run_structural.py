"""Held frames: per-painter vs shared-prep vs batch-painters, rotated windows.

Every case runs the 10 batched V2 workflow on the same held frames, in one
process, around one loaded model, with the drawing of its case::

    case id   <drawing>:b<B>:<execution>     e.g. shared-prep:b8:pipe2
    drawing   per-painter | shared-prep | batch-painters   (drawing_backend)
    B         frames per run (one WorkflowBatchInput, one model batch)
    execution serial (session.run) | pipeN (session.pipeline, N runs in flight)
    capacity  B x max(N, 1) frames in flight: compare cases of equal capacity

    1. build every case (a backend each)
    2. agreement gate, untimed: one frame rotation per case; per frame a
       sha256 of xyxy, class_id, confidence and annotated pixels. Cases of
       one B must agree exactly; across B a real model may differ.
    3. warmup: --warmup iterations per case
    4. windows: --windows rounds; round w runs every case once, starting at
       case w (cyclic rotation), so no case always runs first or last
       window = 10 diagnose repetition of ceil(--frames / B) iterations:
       clock start -> iterations -> torch.cuda.synchronize -> clock stop

Per window: frames/s, process CPU, per-call latencies (raw), engine residual
(call latency minus in-block wall time, serial only), CUDA peak allocated /
reserved, RSS and GC collections. No speedup ratios: compare medians and
ranges of cases with the same B, execution and capacity, from one run.
Uninstrumented: no NVTX or profiler here (use 10 diagnose for those).

    python run_structural.py --fixture /fixtures/parity/fixture.pt \\
        --output /tmp/m46/structural.json
    python run_structural.py --fake-model --device cpu --synthetic-frames 4 \\
        --synthetic-size 360x640 --frames 16 --warmup 2 --windows 2 \\
        --output /tmp/m46/structural-smoke.json
"""

import gc
import hashlib
import math
import statistics
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import structural_imports  # noqa: F401 - installs the 08, 09 and 10 search paths

# isort: split

import diagnose
import drawing_backend
import metrics
from run_benchmark import DEFAULT_MODEL_ID

_STARTED_UTC = datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Case:
    """One drawing at one batch size and execution.

    Args:
        drawing: One of ``drawing_backend.DRAWINGS``.
        batch_size: Frames per run.
        depth: 0 for serial; N for N runs in flight.
    """

    drawing: str
    batch_size: int
    depth: int

    @property
    def id(self) -> str:
        """``<drawing>:b<B>:<execution>``."""
        execution = f"pipe{self.depth}" if self.depth else "serial"
        case_id = f"{self.drawing}:b{self.batch_size}:{execution}"

        return case_id

    @property
    def capacity(self) -> int:
        """Frames in flight at most."""
        return self.batch_size * max(self.depth, 1)


@click.command()
@click.option(
    "--output",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    help="JSON report path.",
)
@click.option(
    "--fixture",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="fixture.pt written by 09 check_parity.py capture.",
)
@click.option(
    "--synthetic-frames",
    type=click.IntRange(
        min=0,
    ),
    default=0,
    show_default=True,
    help="Use N seeded random frames instead of a fixture (smoke tests).",
)
@click.option(
    "--synthetic-size",
    default="1080x1920",
    show_default=True,
    help="HEIGHTxWIDTH of synthetic frames.",
)
@click.option(
    "--fake-model/--trt-model",
    default=False,
    show_default=True,
    help="ScriptedDetectionModel (0-40 boxes per frame) instead of TensorRT.",
)
@click.option(
    "--device",
    default="cuda:0",
    show_default=True,
)
@click.option(
    "--model-id",
    default=DEFAULT_MODEL_ID,
    show_default=True,
)
@click.option(
    "--confidence",
    type=click.FloatRange(
        min=0.0,
        max=1.0,
    ),
    default=0.4,
    show_default=True,
)
@click.option(
    "--drawings",
    default=",".join(drawing_backend.DRAWINGS),
    show_default=True,
    help="Comma-separated drawings.",
)
@click.option(
    "--batch-sizes",
    default="1,8",
    show_default=True,
    help="Comma-separated frames per run.",
)
@click.option(
    "--executions",
    default="serial,pipe2",
    show_default=True,
    help="Comma-separated: serial and/or pipeN (N runs in flight).",
)
@click.option(
    "--cases",
    default="*",
    show_default=True,
    help="Comma-separated glob patterns over case ids.",
)
@click.option(
    "--stages",
    type=click.Choice(
        ["boxes", "full"],
    ),
    default="full",
    show_default=True,
)
@click.option(
    "--warmup",
    type=click.IntRange(
        min=1,
    ),
    default=20,
    show_default=True,
    help="Untimed iterations per case.",
)
@click.option(
    "--frames",
    type=click.IntRange(
        min=1,
    ),
    default=200,
    show_default=True,
    help="Frames per window; rounded up to whole batches.",
)
@click.option(
    "--windows",
    type=click.IntRange(
        min=1,
    ),
    default=5,
    show_default=True,
    help="Rotated rounds; each runs every case once.",
)
@click.option(
    "--list-cases",
    is_flag=True,
    help="Print the selected case ids and exit; imports nothing heavy.",
)
def main(
    output: Optional[Path],
    fixture: Optional[Path],
    synthetic_frames: int,
    synthetic_size: str,
    fake_model: bool,
    device: str,
    model_id: str,
    confidence: float,
    drawings: str,
    batch_sizes: str,
    executions: str,
    cases: str,
    stages: str,
    warmup: int,
    frames: int,
    windows: int,
    list_cases: bool,
) -> None:
    """Time the drawing variants on held frames and write one JSON report."""
    selected = _select_cases(
        drawings=diagnose._str_list(drawings),
        batch_sizes=diagnose._int_list(batch_sizes),
        depths=[diagnose._depth(item) for item in diagnose._str_list(executions)],
        patterns=diagnose._str_list(cases),
    )
    if list_cases:
        click.echo(
            "\n".join(f"{case.id} capacity {case.capacity}" for case in selected)
        )
        return

    if not selected:
        raise click.UsageError("no case selected")
    if output is None:
        raise click.UsageError("--output is required")
    if (fixture is None) == (synthetic_frames == 0):
        raise click.UsageError("give exactly one of --fixture or --synthetic-frames")

    import backends

    backends.configure_mode("v2_serial", device=device)
    import torch

    held, frames_facts = diagnose._load_frames(
        fixture,
        synthetic_frames=synthetic_frames,
        synthetic_size=synthetic_size,
        device=device,
    )
    model = _model(fake_model, model_id=model_id, device=device)
    reports = {case.id: _new_report(case, frames=frames) for case in selected}
    runners, backends_by_case = {}, {}
    try:
        for case in selected:
            backend = _backend(case, model=model, confidence=confidence, stages=stages)
            backends_by_case[case.id] = backend
            runners[case.id] = _runner(case, backend=backend, stages=stages)
            reports[case.id]["facts"] = runners[case.id].facts
        agreement = _agreement(
            selected, runners=runners, backends=backends_by_case, held=held
        )
        for case in selected:
            repeat = diagnose._Repeater(
                runners[case.id], held=held, spec=_spec(case), tracer=_no_tracer()
            )
            reports[case.id]["warmup"] = repeat(warmup, label="warmup")["summary"]
        for window in range(windows):
            for case in _rotated(selected, by=window):
                _time_window(
                    case,
                    runner=runners[case.id],
                    held=held,
                    report=reports[case.id],
                    window=window,
                    device=device,
                )
    finally:
        for runner in runners.values():
            runner.close()

    for report in reports.values():
        report.update(_aggregate(report["windows"]))
    payload = {
        "schema": "m46-structural/1",
        "started_utc": _STARTED_UTC,
        "command": sys.argv,
        "instrumented": False,
        "throughput_validity": (
            "valid when the process ran without nsys/ncu or other profilers; "
            "this script cannot detect an external profiler"
        ),
        "timing": {
            "window": (
                "10 diagnose repetition: ceil(frames / B) iterations, then "
                "torch.cuda.synchronize, then the clock stops; pipelines drain "
                "inside the window"
            ),
            "rotation": "window w runs the cases cyclically from case w",
            "latency": (
                "per run of B frames: serial call to return; pipeline submit-return "
                "to ready-observation, excluding blocking inside submit and any "
                "work completed before submit returns"
            ),
            "residual": "serial only: call latency minus in-block wall time",
            "memory": "per window: CUDA peaks after reset_peak_memory_stats; RSS after",
            "outliers": "none dropped",
        },
        "settings": {
            "confidence": confidence,
            "stages": stages,
            "warmup_iterations": warmup,
            "frames_per_window_requested": frames,
            "windows": windows,
        },
        "frames": frames_facts,
        "model": backends.describe_model(model),
        "fake_model": fake_model,
        "environment": diagnose._environment(torch, device=device),
        "provenance": {
            "modules": diagnose._provenance(model),
            "engine": _engine_provenance(),
        },
        "agreement": agreement,
        "comparison_groups": _groups(selected, reports=reports),
        "cases": list(reports.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_json(output, payload)
    for report in reports.values():
        click.echo(_one_line(report))
    failed = [report["id"] for report in reports.values() if report.get("error")]
    click.echo(
        f"wrote {output}; outputs agree: {agreement['passed']}; failed: {failed or 'none'}"
    )
    if failed or not agreement["passed"]:
        sys.exit(1)


def _select_cases(
    *,
    drawings: List[str],
    batch_sizes: List[int],
    depths: List[int],
    patterns: List[str],
) -> List[Case]:
    unknown = sorted(set(drawings) - set(drawing_backend.DRAWINGS))
    if unknown:
        raise click.BadParameter(f"unknown drawings {unknown}", param_hint="--drawings")

    every = [
        Case(drawing, batch_size, depth)
        for batch_size in batch_sizes
        for depth in depths
        for drawing in drawings
    ]
    selected = [case for case in every if any(fnmatch(case.id, p) for p in patterns)]

    return selected


def _model(fake_model: bool, *, model_id: str, device: str) -> Any:
    if fake_model:
        from scripted_model import ScriptedDetectionModel

        model = ScriptedDetectionModel(device=device)
        return model

    import backends

    model = backends.load_trt_model(model_id, device=device)

    return model


def _backend(case: Case, *, model: Any, confidence: float, stages: str) -> Any:
    backend = drawing_backend.build_for_model(
        "v2_pipeline" if case.depth else "v2_serial",
        model=model,
        model_id="held",
        confidence=confidence,
        pipeline_depth=max(case.depth, 1),
        stages=stages,
        drawing=case.drawing,
    )

    return backend


def _runner(case: Case, *, backend: Any, stages: str) -> Any:
    # A 10 diagnose runner around one backend; block_ms feeds residual_ms.
    def count(rows: List[Any]) -> int:
        return sum(len(backend.predictions(row)) for row in rows)

    def block_ms(rows: List[Any]) -> float:
        # Detector phases and drawing prep are whole-batch times, equal in
        # every row: count them once. Painters are per image.
        detector = sum(rows[0][name] for name in diagnose.DETECTOR_TIMINGS)
        prep = sum(rows[0].get(name) or 0.0 for name in drawing_backend.DRAWING_TIMINGS)
        painters = sum(
            row.get(name) or 0.0 for row in rows for name in diagnose.PAINTER_TIMINGS
        )
        return detector + painters + prep

    runner = diagnose._Runner(
        count=count,
        close=backend.close,
        facts={
            "backend": backend.facts,
            "capacity_frames": case.capacity,
            "engine_calls_per_run": _engine_calls(case, stages=stages),
            "host_transfers_per_run": drawing_backend.HOST_TRANSFERS[case.drawing],
        },
        block_ms=block_ms,
    )
    if case.depth:
        runner.submit = lambda frames, ids: backend.submit_batch(frames, image_ids=ids)
    else:
        runner.call = lambda frames, ids: backend.process_batch(frames, image_ids=ids)

    return runner


def _engine_calls(case: Case, *, stages: str) -> Dict[str, int]:
    # From the compiled plan: a batch-delivering step is called once per run,
    # any other step once per image.
    import drawing_blocks
    from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
    from roboflow_workflows.execution_engine.v2.plan import CompileOptions

    plan = compile_workflow(
        drawing_backend.v2_workflow(stages, drawing=case.drawing),
        catalogue=drawing_blocks.create_catalogue(),
        options=CompileOptions(mutation_conflicts="error"),
    )
    calls = {
        step.path[-1]: 1 if step.delivers_batches else case.batch_size
        for step in plan.steps
    }
    calls["total"] = sum(calls.values())

    return calls


def _agreement(
    cases: List[Case],
    *,
    runners: Dict[str, Any],
    backends: Dict[str, Any],
    held: Any,
) -> Dict[str, Any]:
    """One untimed rotation per case; per frame, a digest of everything drawn."""
    digests: Dict[str, Dict[int, str]] = {}
    for case in cases:
        runner = runners[case.id]
        seen: Dict[int, str] = {}
        for iteration in range(held.rotation(case.batch_size)):
            frames, image_ids = held.batch(iteration, size=case.batch_size)
            rows = (
                runner.call(frames, image_ids)
                if runner.call is not None
                else runner.submit(frames, image_ids).result()
            )
            for row, image_id in zip(rows, image_ids):
                frame_index = int(image_id.rsplit("-f", 1)[1])
                seen[frame_index] = _digest(backends[case.id], row)
        digests[case.id] = seen

    by_batch: Dict[int, Dict[str, Any]] = {}
    for case in cases:
        group = by_batch.setdefault(
            case.batch_size, {"reference": case.id, "differing": []}
        )
        if digests[case.id] != digests[group["reference"]]:
            group["differing"].append(case.id)
    agreement = {
        "compared": "sha256 of xyxy, class_id, confidence and annotated bytes per frame",
        "by_batch_size": by_batch,
        "passed": all(not group["differing"] for group in by_batch.values()),
    }

    return agreement


def _digest(backend: Any, row: Any) -> str:
    predictions = backend.predictions(row)
    annotated = backend.annotated(row)
    digest = hashlib.sha256()
    for tensor in (predictions.xyxy, predictions.class_id, predictions.confidence):
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if annotated is not None:
        digest.update(annotated.detach().cpu().contiguous().numpy().tobytes())
    hexdigest = digest.hexdigest()

    return hexdigest


def _time_window(
    case: Case,
    *,
    runner: Any,
    held: Any,
    report: Dict[str, Any],
    window: int,
    device: str,
) -> None:
    import psutil
    import torch

    is_cuda = torch.device(device).type == "cuda"
    iterations = report["iterations_per_window"]
    repeat = diagnose._Repeater(
        runner, held=held, spec=_spec(case), tracer=_no_tracer()
    )
    gc_before = [stats["collections"] for stats in gc.get_stats()]
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        repetition = repeat(iterations, label=f"window {window}")
    except Exception as error:  # recorded, the other cases still run
        report.setdefault("error", f"window {window}: {error!r}")
        report.setdefault("traceback", traceback.format_exc())
        return

    memory = {"rss_bytes_after": psutil.Process().memory_info().rss}
    if is_cuda:
        memory.update(
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            cuda_allocated_bytes_after=torch.cuda.memory_allocated(device),
        )
    report["windows"].append(
        {
            "window": window,
            **repetition["summary"],
            "memory": memory,
            "gc_collections": [
                stats["collections"] - before
                for stats, before in zip(gc.get_stats(), gc_before)
            ],
            "latencies_ms": repetition["latencies_ms"],
            "residuals_ms": repetition["residuals_ms"],
        }
    )


def _aggregate(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not windows:
        return {}

    fps = [window["frames_per_second"] for window in windows]
    latencies = [value for window in windows for value in window["latencies_ms"]]
    residuals = [value for window in windows for value in window["residuals_ms"]]
    peaks = [window["memory"].get("cuda_peak_allocated_bytes") for window in windows]
    aggregate = {
        "frames_per_second": {
            "median": statistics.median(fps),
            "min": min(fps),
            "max": max(fps),
        },
        "cpu_ms_per_frame_median": statistics.median(
            window["cpu_ms_per_frame"] for window in windows
        ),
        "latency_ms": metrics._distribution(latencies),
        "residual_ms": metrics._distribution(residuals),
        "cuda_peak_allocated_bytes_max": None if None in peaks else max(peaks),
        "rss_bytes_after_max": max(
            window["memory"]["rss_bytes_after"] for window in windows
        ),
        "gc_collections_total": [
            sum(window["gc_collections"][generation] for window in windows)
            for generation in range(len(windows[0]["gc_collections"]))
        ],
    }

    return aggregate


def _groups(
    cases: List[Case], *, reports: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    # Cases that may be compared: same B, same execution, same capacity.
    groups: Dict[str, List[Case]] = {}
    for case in cases:
        execution = f"pipe{case.depth}" if case.depth else "serial"
        groups.setdefault(f"b{case.batch_size}:{execution}", []).append(case)

    summary = [
        {
            "group": name,
            "capacity_frames": members[0].capacity,
            "cases": {
                case.id: {
                    "frames_per_second": reports[case.id].get("frames_per_second"),
                    "latency_ms_p50": (reports[case.id].get("latency_ms") or {}).get(
                        "p50"
                    ),
                    "latency_ms_p99": (reports[case.id].get("latency_ms") or {}).get(
                        "p99"
                    ),
                    "engine_calls_per_run": reports[case.id]["facts"][
                        "engine_calls_per_run"
                    ]["total"],
                    "host_transfers_per_run": reports[case.id]["facts"][
                        "host_transfers_per_run"
                    ],
                }
                for case in members
            },
        }
        for name, members in groups.items()
    ]

    return summary


def _engine_provenance() -> Dict[str, Any]:
    """Where the V2 engine came from, and a hash of all its Python sources."""
    import roboflow_workflows

    root = Path(roboflow_workflows.__file__).resolve().parent
    engine = root / "execution_engine" / "v2"
    digest = hashlib.sha256()
    files = sorted(engine.rglob("*.py"))
    for path in files:
        digest.update(str(path.relative_to(engine)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    provenance = {
        "roboflow_workflows": str(root),
        "execution_engine_v2_files": len(files),
        "execution_engine_v2_sha256": digest.hexdigest(),
        "hash_scheme": "sha256 over sorted (relative path, sha256(file)) of v2/**/*.py",
    }

    return provenance


def _new_report(case: Case, *, frames: int) -> Dict[str, Any]:
    iterations = math.ceil(frames / case.batch_size)
    report = {
        "id": case.id,
        "drawing": case.drawing,
        "batch_size": case.batch_size,
        "depth": case.depth,
        "capacity_frames": case.capacity,
        "iterations_per_window": iterations,
        "frames_per_window": iterations * case.batch_size,
        "windows": [],
    }

    return report


def _spec(case: Case) -> Any:
    # _Repeater reads batch_size, depth and id; stage carries drawing for labels.
    spec = diagnose.CaseSpec("v2b", case.drawing, case.batch_size, case.depth)

    return spec


def _rotated(cases: List[Case], *, by: int) -> List[Case]:
    shift = by % len(cases)
    rotated = cases[shift:] + cases[:shift]

    return rotated


def _one_line(report: Dict[str, Any]) -> str:
    if report.get("error"):
        return f"{report['id']}: FAILED {report['error']}"

    fps = report["frames_per_second"]
    timer = (
        "submit-return to ready-observation"
        if ":pipe" in report["id"]
        else "call latency"
    )
    line = (
        f"{report['id']} (capacity {report['capacity_frames']}): "
        f"{fps['median']:.1f} frames/s ({fps['min']:.1f}-{fps['max']:.1f}), "
        f"{timer} p50 {report['latency_ms']['p50']:.2f} ms "
        f"p99 {report['latency_ms']['p99']:.2f} ms, "
        f"cpu {report['cpu_ms_per_frame_median']:.2f} ms/frame, "
        f"engine calls/run {report['facts']['engine_calls_per_run']['total']}"
    )

    return line


def _no_tracer() -> Any:
    # diagnose._Repeater needs a tracer; this one records nothing.
    tracer = diagnose._Nvtx(enabled=False)

    return tracer


if __name__ == "__main__":
    main()
