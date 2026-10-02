"""Held-frame ablation: where does frame time go, model call to full workflow?

Diagnostic only. Every case runs on the same held GPU frames, in one process,
around one loaded model. No decoding, no sources, no admission::

    held frames ─┬─ forward         model.forward(pre-processed)   network only
                 ├─ model           model(frames): pre, fwd, post  inference_models call
                 ├─ blocks.<stage>  09 gpu_blocks called directly  no engine
                 ├─ v2.<stage>      V2 session, 09 gpu_blocks      09 per-image V2
                 └─ v2b.<stage>     10 batched_backend             physical batches

    stage: detector | boxes (+ box painter) | full (+ label painter)

Case id ``<family>[.<stage>]:b<B>:<execution>``, e.g. ``v2.full:b1:pipe2``.
``serial`` calls and waits; ``pipeN`` keeps N calls in flight (V2 pipeline,
or N threads for ``model``). ``--list-cases`` prints the grid.

Timing per case::

    warmup        --warmup iterations, untimed (summary kept)
    observe       one full frame rotation with model.forward wrapped: the batch
                  size and a per-image checksum of every forward input
    repetitions   --repetitions x ceil(--frames / B) iterations, each:
                    clock start -> iterations -> torch.cuda.synchronize -> clock stop

Iteration ``k`` takes the next B held frames in rotation, in every case. The
``forward`` case pre-processes each distinct rotating batch once, before
timing, and forwards the batch of the frames it is given. The report compares
the observed forward-input checksums of all cases (``forward_inputs``).

A result is ready when the call returns or its future is done: every block and
model phase here waits for its own GPU work. The device synchronize at the end
of a repetition only guards against leaked work. Latency is per iteration
(B frames): call start to return, or submit accepted to future done. Pipelines
drain inside each repetition. No values are dropped. Serial V2 cases also
record, per iteration, the call latency minus the wall time measured inside
that call's blocks (``residual_ms``): engine and wrapping cost of that call.

Instrumentation is off by default. ``--nvtx`` adds NVTX ranges (case,
repetition, iteration, model phases, block ``run``). ``--profile-case ID``
runs that case's warmup, then only one repetition of ``--profile-frames``
between ``cudaProfilerStart`` and ``cudaProfilerStop``, for
``nsys profile --capture-range=cudaProfilerApi``. Never report timings from an
instrumented run as throughput; the JSON says ``instrumented``.

    python diagnose.py --fixture /tmp/thor/parity/fixture.pt \\
        --output /tmp/thor/diagnose.json
    python diagnose.py --fake-model --device cpu --synthetic-frames 4 \\
        --frames 8 --warmup 2 --output /tmp/diagnose-smoke.json
"""

import hashlib
import math
import platform
import statistics
import sys
import threading
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

import click
import thor_imports  # installs the 08 and 09 search paths

# isort: split

import metrics
from run_benchmark import DEFAULT_MODEL_ID

FAMILY_STAGES = (
    ("forward", None),
    ("model", None),
    ("blocks", "detector"),
    ("blocks", "boxes"),
    ("blocks", "full"),
    ("v2", "detector"),
    ("v2", "boxes"),
    ("v2", "full"),
    ("v2b", "detector"),
    ("v2b", "boxes"),
    ("v2b", "full"),
)
# Block wall times (ms) in V2 results: detector phases, then painters.
DETECTOR_TIMINGS = ("pre_ms", "model_ms", "post_ms")
PAINTER_TIMINGS = ("boxes_ms", "labels_ms")
# Environment that changes what the model or blocks do; recorded, never set here.
RECORDED_ENVIRONMENT = (
    "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND",
    "ENABLE_TENSOR_DATA_REPRESENTATION",
    "WORKFLOWS_IMAGE_TENSOR_DEVICE",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
)
# Library modules whose source decides the measured work; hashed into the
# report together with every loaded 08/09/10 example module.
PROVENANCE_MODULES = (
    "inference_models.models.common.trt",
    "roboflow_workflows.execution_engine.v2.pipelining.stages",
    "roboflow_workflows.core_steps.visualizations.label.v1_tensor",
)
EXAMPLES_DIR = Path(__file__).resolve().parent.parent
_STARTED_UTC = datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CaseSpec:
    """One cell of the ablation grid.

    Args:
        family: ``forward``, ``model``, ``blocks``, ``v2`` or ``v2b``.
        stage: ``detector``, ``boxes``, ``full``; ``None`` for model families.
        batch_size: Frames per call.
        depth: 0 for serial calls; N for N calls in flight.
    """

    family: str
    stage: Optional[str]
    batch_size: int
    depth: int

    @property
    def id(self) -> str:
        """``<family>[.<stage>]:b<B>:<execution>``."""
        name = self.family if self.stage is None else f"{self.family}.{self.stage}"
        execution = f"pipe{self.depth}" if self.depth else "serial"
        case_id = f"{name}:b{self.batch_size}:{execution}"

        return case_id


@dataclass(frozen=True)
class _Settings:
    device: str
    model_id: str
    confidence: float
    nms: Dict[str, Any]


@dataclass
class _Runner:
    # Exactly one of call (serial) or submit (pipelined) is set.
    count: Callable[[Any], Optional[int]]
    call: Optional[Callable[[List[Any], List[str]], Any]] = None
    submit: Optional[Callable[[List[Any], List[str]], Future]] = None
    close: Callable[[], None] = lambda: None
    facts: Dict[str, Any] = field(default_factory=dict)
    # Wall ms measured inside the blocks of one call's output; None if unknown.
    block_ms: Callable[[Any], Optional[float]] = lambda _: None


def case_grid(*, batch_sizes: List[int], depths: List[int]) -> List[CaseSpec]:
    """Every supported case for these batch sizes and executions, family first.

    Args:
        batch_sizes: Frames per call, e.g. ``[1, 8]``.
        depths: 0 for serial, N for N in flight.

    Returns:
        Cases in run order. ``forward`` is serial only; ``blocks`` is serial
        and B1 only; ``v2`` (09 per-image blocks) is B1 only.
    """
    specs = []
    for family, stage in FAMILY_STAGES:
        for batch_size in batch_sizes:
            for depth in depths:
                spec = CaseSpec(family, stage, batch_size, depth)
                if _supported(spec):
                    specs.append(spec)

    return specs


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
    help="fixture.pt written by 09 check_parity.py.",
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
    help="Deterministic CPU stand-in model, for smoke tests without TensorRT.",
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
    "--cases",
    default="*",
    show_default=True,
    help="Comma-separated glob patterns over case ids, e.g. 'model:*,v2.full:*'.",
)
@click.option(
    "--batch-sizes",
    default="1,8",
    show_default=True,
    help="Comma-separated frames per call.",
)
@click.option(
    "--executions",
    default="serial,pipe2",
    show_default=True,
    help="Comma-separated: serial and/or pipeN (N calls in flight).",
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
    help="Frames per repetition; rounded up to whole batches.",
)
@click.option(
    "--repetitions",
    type=click.IntRange(
        min=1,
    ),
    default=3,
    show_default=True,
)
@click.option(
    "--nvtx/--no-nvtx",
    default=False,
    show_default=True,
    help="NVTX ranges for Nsight Systems. Makes the run instrumented.",
)
@click.option(
    "--profile-case",
    default=None,
    help="Case id whose capture window is wrapped in cudaProfilerStart/Stop.",
)
@click.option(
    "--profile-frames",
    type=click.IntRange(
        min=1,
    ),
    default=100,
    show_default=True,
    help="Frames inside the capture window of --profile-case.",
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
    cases: str,
    batch_sizes: str,
    executions: str,
    warmup: int,
    frames: int,
    repetitions: int,
    nvtx: bool,
    profile_case: Optional[str],
    profile_frames: int,
    list_cases: bool,
) -> None:
    """Time the held-frame ablation ladder and write one JSON report."""
    grid = case_grid(
        batch_sizes=_int_list(batch_sizes),
        depths=[_depth(execution) for execution in _str_list(executions)],
    )
    patterns = _str_list(cases)
    specs = [spec for spec in grid if any(fnmatch(spec.id, p) for p in patterns)]
    if list_cases:
        click.echo("\n".join(spec.id for spec in specs))
        return

    if not specs:
        raise click.UsageError(f"no case matches {patterns}")
    if output is None:
        raise click.UsageError("--output is required")
    if (fixture is None) == (synthetic_frames == 0):
        raise click.UsageError("give exactly one of --fixture or --synthetic-frames")
    if profile_case is not None and profile_case not in {spec.id for spec in specs}:
        raise click.UsageError(f"--profile-case {profile_case} is not selected")
    if profile_case is not None and not device.startswith("cuda"):
        raise click.UsageError("--profile-case needs a CUDA --device")

    import backends

    backends.configure_mode("v2_serial", device=device)
    import torch

    tracer = _Nvtx(enabled=nvtx)
    held, frames_facts = _load_frames(
        fixture,
        synthetic_frames=synthetic_frames,
        synthetic_size=synthetic_size,
        device=device,
    )
    model = (
        _FakeDetectionModel(device=device)
        if fake_model
        else backends.load_trt_model(model_id, device=device)
    )
    settings = _Settings(
        device=device,
        model_id=model_id,
        confidence=confidence,
        nms=backends.post_process_parameters(),
    )
    if nvtx:
        _instrument(model, tracer=tracer)

    reports = []
    for spec in specs:
        report = _run_case(
            spec,
            model=model,
            held=held,
            settings=settings,
            warmup=warmup,
            frames=frames if spec.id != profile_case else profile_frames,
            repetitions=repetitions,
            profiled=spec.id == profile_case,
            tracer=tracer,
        )
        reports.append(report)
        click.echo(_one_line(report))

    payload = {
        "schema": "m45-diagnose/1",
        "started_utc": _STARTED_UTC,
        "command": sys.argv,
        "instrumented": nvtx or profile_case is not None,
        "instrumentation": {"nvtx": tracer.active, "profile_case": profile_case},
        "throughput_validity": (
            "valid only when instrumented is false and the process ran without "
            "nsys/ncu; this script cannot detect an external profiler"
        ),
        "timing": {
            "readiness": (
                "a result is ready when the call returns or its future is done; "
                "each repetition ends with torch.cuda.synchronize, then the clock stops"
            ),
            "latency": (
                "per iteration of B frames: serial call start to return; "
                "pipelined submit accepted to future done"
            ),
            "pipelines": "drained inside each repetition; drain is in the denominator",
            "cpu": "process user+system CPU seconds, all threads",
            "outliers": "none dropped",
        },
        "settings": {
            "confidence": confidence,
            "nms": settings.nms,
            "warmup_iterations": warmup,
            "frames_per_repetition_requested": frames,
            "repetitions": repetitions,
            "profile_frames": profile_frames if profile_case else None,
        },
        "frames": frames_facts,
        "model": backends.describe_model(model),
        "fake_model": fake_model,
        "environment": _environment(torch, device=device),
        "provenance": _provenance(model),
        "forward_inputs": _compare_forward_inputs(reports, frames=len(held.frames)),
        "cases": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_json(output, payload)
    failed = [report["id"] for report in reports if report.get("error")]
    click.echo(f"wrote {output}; failed cases: {failed or 'none'}")
    if failed:
        sys.exit(1)


def _run_case(
    spec: CaseSpec,
    *,
    model: Any,
    held: "_HeldFrames",
    settings: _Settings,
    warmup: int,
    frames: int,
    repetitions: int,
    profiled: bool,
    tracer: "_Nvtx",
) -> Dict[str, Any]:
    iterations = math.ceil(frames / spec.batch_size)
    report: Dict[str, Any] = {
        "id": spec.id,
        "family": spec.family,
        "stage": spec.stage,
        "batch_size": spec.batch_size,
        "depth": spec.depth,
        "iterations_per_repetition": iterations,
        "frames_per_repetition": iterations * spec.batch_size,
    }
    runner = None
    try:
        runner = _BUILDERS[spec.family](spec, model=model, held=held, settings=settings)
        report["facts"] = runner.facts
        repeat = _Repeater(runner, held=held, spec=spec, tracer=tracer)

        report["warmup"] = repeat(warmup, label="warmup")["summary"]
        rotation = held.rotation(spec.batch_size)
        observation = _observe_forward(model, repeat=repeat, iterations=rotation)
        report["forward_batch_sizes"] = observation["batch_sizes"]
        report["forward_input_checksums"] = observation["checksums"]
        if profiled:
            report["profiled_repetition"] = _profiled(repeat, iterations=iterations)
            return report

        timed = [
            repeat(iterations, label=f"repetition {index}")
            for index in range(repetitions)
        ]
        report["repetitions"] = [rep["summary"] for rep in timed]
        report.update(_aggregate(timed))
    except Exception as error:  # recorded, the next case still runs
        report["error"] = "".join(traceback.format_exception_only(error)).strip()
        report["traceback"] = traceback.format_exc()
    finally:
        if runner is not None:
            runner.close()

    return report


class _Repeater:
    """Run ``iterations`` iterations of one case and time them as one repetition."""

    def __init__(
        self, runner: _Runner, *, held: "_HeldFrames", spec: CaseSpec, tracer: "_Nvtx"
    ):
        self._runner = runner
        self._held = held
        self._spec = spec
        self._tracer = tracer
        self._synchronize = _synchronizer(held.device)

    def __call__(self, iterations: int, *, label: str) -> Dict[str, Any]:
        latencies_ms: List[float] = []
        residuals_ms: List[float] = []
        detections: List[Optional[int]] = []

        with self._tracer.range(f"{self._spec.id} {label}"):
            cpu_started = metrics.process_cpu_seconds()
            started = perf_counter()
            if self._runner.call is not None:
                self._serial(
                    iterations,
                    latencies_ms=latencies_ms,
                    residuals_ms=residuals_ms,
                    detections=detections,
                )
            else:
                self._pipelined(
                    iterations, latencies_ms=latencies_ms, detections=detections
                )
            self._synchronize()
            elapsed = perf_counter() - started
            cpu_seconds = metrics.process_cpu_seconds() - cpu_started

        frames = iterations * self._spec.batch_size
        counted = None if None in detections else sum(detections)
        summary = {
            "elapsed_s": elapsed,
            "frames": frames,
            "frames_per_second": frames / elapsed,
            "detections": counted,
            "cpu_s": cpu_seconds,
            "cpu_ms_per_frame": cpu_seconds * 1000.0 / frames,
            "latency_ms_p50": statistics.median(latencies_ms),
        }
        repetition = {
            "summary": summary,
            "latencies_ms": latencies_ms,
            "residuals_ms": residuals_ms,
        }

        return repetition

    def _serial(
        self,
        iterations: int,
        *,
        latencies_ms: List[float],
        residuals_ms: List[float],
        detections: List,
    ) -> None:
        for iteration in range(iterations):
            frames, image_ids = self._held.batch(iteration, size=self._spec.batch_size)
            with self._tracer.range(f"iteration {iteration}"):
                called = perf_counter()
                output = self._runner.call(frames, image_ids)
                latency_ms = (perf_counter() - called) * 1000.0
            latencies_ms.append(latency_ms)
            detections.append(self._runner.count(output))
            # The same call's latency minus its own in-block wall time.
            block_ms = self._runner.block_ms(output)
            if block_ms is not None:
                residuals_ms.append(latency_ms - block_ms)

    def _pipelined(
        self, iterations: int, *, latencies_ms: List[float], detections: List
    ) -> None:
        # Retire finished submissions as we go, so held results stay bounded.
        pending: Deque[_Submission] = deque()
        for iteration in range(iterations):
            frames, image_ids = self._held.batch(iteration, size=self._spec.batch_size)
            range_id = self._tracer.start(f"iteration {iteration}")
            future = self._runner.submit(frames, image_ids)
            pending.append(_Submission(future, tracer=self._tracer, range_id=range_id))
            while pending and pending[0].future.done():
                self._retire(pending.popleft(), latencies_ms, detections)
        while pending:
            self._retire(pending.popleft(), latencies_ms, detections)

    def _retire(
        self, submission: "_Submission", latencies_ms: List[float], detections: List
    ) -> None:
        output, latency_ms = submission.retire()
        latencies_ms.append(latency_ms)
        detections.append(self._runner.count(output))


class _Submission:
    """A submitted iteration: accepted when ``submit`` returned, done when ready."""

    def __init__(self, future: Future, *, tracer: "_Nvtx", range_id: Optional[int]):
        self.future = future
        self._accepted = perf_counter()
        self._done_at: Optional[float] = None
        self._stamped = threading.Event()
        self._tracer = tracer
        self._range_id = range_id
        future.add_done_callback(self._stamp)

    def retire(self) -> Tuple[Any, float]:
        output = self.future.result()
        # The callback runs right after the result is set; wait for its stamp.
        self._stamped.wait()
        latency_ms = (self._done_at - self._accepted) * 1000.0

        return output, latency_ms

    def _stamp(self, _: Future) -> None:
        self._done_at = perf_counter()
        self._tracer.end(self._range_id)
        self._stamped.set()


def _observe_forward(
    model: Any, *, repeat: _Repeater, iterations: int
) -> Dict[str, List]:
    """What ``model.forward`` receives during ``iterations`` untimed iterations.

    Returns:
        ``batch_sizes``, one per forward call, and ``checksums``: per image,
        in call order, the float64 sum of its network input. Reading the
        checksums synchronizes; that is why this runs outside timing.
    """
    sizes: List[int] = []
    checksums: List[float] = []
    previous = model.__dict__.get("forward")
    original = model.forward

    def observed(network_input: Any, *args: Any, **kwargs: Any) -> Any:
        sizes.append(int(network_input.shape[0]))
        checksums.extend(network_input.flatten(1).double().sum(dim=1).tolist())
        return original(network_input, *args, **kwargs)

    model.forward = observed
    try:
        repeat(iterations, label="observe forward")
    finally:
        if previous is None:
            del model.forward
        else:
            model.forward = previous
    observation = {"batch_sizes": sizes, "checksums": checksums}

    return observation


def _profiled(repeat: _Repeater, *, iterations: int) -> Dict[str, Any]:
    import torch

    torch.cuda.profiler.start()
    try:
        repetition = repeat(iterations, label="profiled")
    finally:
        torch.cuda.profiler.stop()
    summary = {**repetition["summary"], "instrumented": True}

    return summary


def _compare_forward_inputs(
    reports: List[Dict[str, Any]], *, frames: int
) -> Dict[str, Any]:
    """Did every case forward the same held frames, in the same order?

    Each observation covers one rotation from frame 0, so its first ``frames``
    checksums are frames 0..F-1. They are compared with the first observed
    case. Differences are reported as numbers, not hidden: batched
    pre-processing may legitimately differ in the last bits.
    """
    observed = [
        (report["id"], report["forward_input_checksums"][:frames])
        for report in reports
        if report.get("forward_input_checksums")
    ]
    if not observed:
        return {"reference": None, "cases": {}}

    reference_id, reference = observed[0]
    cases = {}
    for case_id, checksums in observed:
        same_length = len(checksums) == len(reference)
        cases[case_id] = {
            # Plain threads (model:*:pipeN) may reach forward in any order.
            "same_frames": sorted(checksums) == sorted(reference),
            "same_order": checksums == reference,
            "max_abs_diff": (
                max(abs(a - b) for a, b in zip(checksums, reference))
                if same_length
                else None
            ),
        }
    comparison = {
        "reference": reference_id,
        "per_image": "float64 sum of the network input of each held frame",
        "cases": cases,
    }

    return comparison


def _aggregate(timed: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [rep["summary"] for rep in timed]
    fps = [summary["frames_per_second"] for summary in summaries]
    cpu = [summary["cpu_ms_per_frame"] for summary in summaries]
    latencies = [value for rep in timed for value in rep["latencies_ms"]]
    residuals = [value for rep in timed for value in rep["residuals_ms"]]
    aggregate = {
        "frames_per_second": {
            "median": statistics.median(fps),
            "min": min(fps),
            "max": max(fps),
        },
        "cpu_ms_per_frame_median": statistics.median(cpu),
        # Reuses 09's private quantile helper: count, mean, p50..p99, max.
        "latency_ms": metrics._distribution(latencies),
        "residual_ms": metrics._distribution(residuals),
        "detections_per_repetition": [summary["detections"] for summary in summaries],
    }

    return aggregate


def _build_forward(
    spec: CaseSpec, *, model: Any, held: "_HeldFrames", settings: _Settings
) -> _Runner:
    # Pre-process every distinct rotating batch once, here, outside timing;
    # a call forwards the cached input of exactly the frames it is given.
    network_inputs = {}
    for iteration in range(held.rotation(spec.batch_size)):
        frames, _ = held.batch(iteration, size=spec.batch_size)
        network_input, _ = model.pre_process(frames, input_color_format="rgb")
        network_inputs[_identity(frames)] = network_input
    _synchronizer(held.device)()

    def call(frames: List[Any], image_ids: List[str]) -> Any:
        return model.forward(network_inputs[_identity(frames)])

    runner = _Runner(
        call=call,
        count=lambda _: None,
        facts={
            "runs": "model.forward on the cached pre-processed input of each batch",
            "cached_batches": len(network_inputs),
            "network_input_shape": list(network_input.shape),
        },
    )

    return runner


def _build_model(
    spec: CaseSpec, *, model: Any, held: "_HeldFrames", settings: _Settings
) -> _Runner:
    kwargs = {
        "confidence": settings.confidence,
        "input_color_format": "rgb",
        **settings.nms,
    }

    def call(frames: List[Any], image_ids: List[str]) -> Any:
        return model(frames, **kwargs)

    facts = {"runs": "model(frames): pre_process, forward, post_process (NMS)"}
    count = _detections_in_list
    if not spec.depth:
        runner = _Runner(call=call, count=count, facts=facts)
        return runner

    threads = _BoundedThreads(call, depth=spec.depth)
    facts["pipelining"] = f"{spec.depth} threads; submit blocks while all are busy"
    runner = _Runner(
        submit=threads.submit, count=count, close=threads.close, facts=facts
    )

    return runner


def _build_blocks(
    spec: CaseSpec, *, model: Any, held: "_HeldFrames", settings: _Settings
) -> _Runner:
    import gpu_blocks
    from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData

    detector = gpu_blocks.ObjectDetector(detection_model=model)
    boxes = gpu_blocks.GpuBoxVisualization()
    labels = gpu_blocks.GpuLabelVisualization()

    def call(frames: List[Any], image_ids: List[str]) -> Any:
        (frame,), (image_id,) = frames, image_ids
        image = ImageData.from_tensor(frame, image_id=image_id)
        detected = detector.run(image=image, confidence=settings.confidence)
        predictions = detected["predictions"]
        if spec.stage != "detector":
            painted = boxes.run(image=image, predictions=predictions)
            if spec.stage == "full":
                labels.run(image=painted["image"], predictions=predictions)
        return predictions

    runner = _Runner(
        call=call,
        count=len,
        facts={
            "runs": "09 gpu_blocks run() called in workflow order, no engine",
            "nms": "model defaults (equal to settings.nms)",
        },
    )

    return runner


def _build_v2(
    spec: CaseSpec, *, model: Any, held: "_HeldFrames", settings: _Settings
) -> _Runner:
    import backends
    import batched_backend
    import gpu_blocks
    from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
    from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
    from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions
    from roboflow_workflows.execution_engine.v2.plan import CompileOptions

    definition = batched_backend.v2_workflow(spec.stage, batched=False)
    # Same compile options as 09 V2Backend: whole calls serially, phases piped.
    block_execution = "phases" if spec.depth else "run"
    plan = compile_workflow(
        definition,
        catalogue=gpu_blocks.create_catalogue(),
        options=CompileOptions(
            block_execution=block_execution,
            mutation_conflicts="error",
        ),
    )
    session = plan.create_session({"detection_model": model})

    def inputs(frames: List[Any], image_ids: List[str]) -> Dict[str, Any]:
        (frame,), (image_id,) = frames, image_ids
        image = ImageData.from_tensor(frame, image_id=image_id)
        return {"image": image, "confidence": settings.confidence}

    def count(result: Any) -> int:
        return len(backends._v2_output(result, "predictions"))

    outputs = {output["name"] for output in definition["outputs"]}
    timing_names = [
        name for name in DETECTOR_TIMINGS + PAINTER_TIMINGS if name in outputs
    ]

    def block_ms(result: Any) -> float:
        return sum(backends._v2_output(result, name) for name in timing_names)

    facts = {
        "runs": "V2 session, 09 gpu_blocks catalogue",
        "workflow": "batched_backend.v2_workflow(stage, batched=False)",
        "block_execution": block_execution,
        "steps": [step["name"] for step in definition["steps"]],
        "block_ms": timing_names,
    }
    if not spec.depth:
        runner = _Runner(
            call=lambda frames, ids: session.run(inputs(frames, ids)),
            count=count,
            facts=facts,
            block_ms=block_ms,
        )
        return runner

    pipeline = session.pipeline(options=PipelineOptions(max_in_flight=spec.depth))
    pipeline.__enter__()
    runner = _Runner(
        submit=lambda frames, ids: pipeline.submit(inputs(frames, ids)),
        count=count,
        close=lambda: pipeline.__exit__(None, None, None),
        facts={**facts, "pipeline_depth": spec.depth},
    )

    return runner


def _build_v2b(
    spec: CaseSpec, *, model: Any, held: "_HeldFrames", settings: _Settings
) -> _Runner:
    import batched_backend

    backend = batched_backend.build_for_model(
        "v2_pipeline" if spec.depth else "v2_serial",
        model=model,
        model_id=settings.model_id,
        confidence=settings.confidence,
        pipeline_depth=max(spec.depth, 1),
        stages=spec.stage,
    )

    def count(rows: List[Any]) -> int:
        return sum(len(backend.predictions(row)) for row in rows)

    def block_ms(rows: List[Any]) -> float:
        # Detector phase times are per batch (equal in every row); painter
        # times are per image.
        detector = sum(rows[0][name] for name in DETECTOR_TIMINGS)
        painters = sum(row.get(name) or 0.0 for row in rows for name in PAINTER_TIMINGS)
        return detector + painters

    runner = _Runner(
        count=count,
        close=backend.close,
        facts={"runs": "10 batched_backend", "backend": backend.facts},
        block_ms=block_ms,
    )
    if spec.depth:
        runner.submit = lambda frames, ids: backend.submit_batch(frames, image_ids=ids)
    else:
        runner.call = lambda frames, ids: backend.process_batch(frames, image_ids=ids)

    return runner


_BUILDERS = {
    "forward": _build_forward,
    "model": _build_model,
    "blocks": _build_blocks,
    "v2": _build_v2,
    "v2b": _build_v2b,
}


def _supported(spec: CaseSpec) -> bool:
    if spec.family == "forward":
        return spec.depth == 0
    if spec.family == "blocks":
        # Direct calls are serial: the label painter's sprite cache is unlocked.
        return spec.batch_size == 1 and spec.depth == 0
    if spec.family == "v2":
        return spec.batch_size == 1

    return True


def _identity(frames: List[Any]) -> Tuple[int, ...]:
    # Held frames are long-lived tensors, so object identity names the batch.
    identity = tuple(id(frame) for frame in frames)

    return identity


def _detections_in_list(detections: List[Any]) -> int:
    count = sum(len(item) for item in detections)

    return count


class _BoundedThreads:
    """``depth`` threads; ``submit`` blocks while all are busy, like a V2 pipeline."""

    def __init__(self, call: Callable, *, depth: int):
        self._call = call
        self._free = threading.Semaphore(depth)
        self._executor = ThreadPoolExecutor(
            max_workers=depth, thread_name_prefix="diagnose"
        )

    def submit(self, frames: List[Any], image_ids: List[str]) -> Future:
        self._free.acquire()
        future = self._executor.submit(self._call, frames, image_ids)
        future.add_done_callback(lambda _: self._free.release())

        return future

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class _HeldFrames:
    """Frames on the device; iteration ``k`` takes the next ``size`` in rotation."""

    def __init__(self, frames: List[Any], *, device: str):
        self.frames = frames
        self.device = device

    def rotation(self, size: int) -> int:
        """Iterations until batches of ``size`` repeat: one pass over the frames."""
        iterations = len(self.frames) // math.gcd(len(self.frames), size)

        return iterations

    def batch(self, iteration: int, *, size: int) -> Tuple[List[Any], List[str]]:
        indices = [(iteration * size + slot) % len(self.frames) for slot in range(size)]
        frames = [self.frames[index] for index in indices]
        # Unique within a batch even when a batch is larger than the fixture.
        image_ids = [
            f"k{iteration}-s{slot}-f{index}" for slot, index in enumerate(indices)
        ]

        return frames, image_ids


def _load_frames(
    fixture: Optional[Path],
    *,
    synthetic_frames: int,
    synthetic_size: str,
    device: str,
) -> Tuple[_HeldFrames, Dict[str, Any]]:
    import torch

    if fixture is not None:
        load_fixture = thor_imports.load_09_module("check_parity").load_fixture
        stored = load_fixture(fixture)["frames"]
        facts = {
            "source": str(fixture),
            "sha256": _sha256(fixture),
        }
    else:
        height, width = (int(value) for value in synthetic_size.split("x"))
        generator = torch.Generator().manual_seed(0)
        stored = [
            torch.randint(
                0, 256, (3, height, width), dtype=torch.uint8, generator=generator
            )
            for _ in range(synthetic_frames)
        ]
        facts = {"source": "synthetic, torch.Generator seed 0"}

    frames = [frame.to(device).contiguous() for frame in stored]
    # Uploads are complete before any case reads the frames on another stream.
    _synchronizer(device)()
    facts.update(
        {
            "count": len(frames),
            "shape": list(frames[0].shape),
            "dtype": str(frames[0].dtype),
            "device": device,
        }
    )
    held = _HeldFrames(frames, device=device)

    return held, facts


def _synchronizer(device: str) -> Callable[[], None]:
    import torch

    if torch.device(device).type != "cuda":
        return lambda: None

    return lambda: torch.cuda.synchronize(device)


class _Nvtx:
    """NVTX ranges when enabled and CUDA is present; otherwise no-ops."""

    def __init__(self, *, enabled: bool):
        import torch

        self.active = enabled and torch.cuda.is_available()
        self._nvtx = torch.cuda.nvtx if self.active else None

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        if not self.active:
            yield
            return

        self._nvtx.range_push(name)
        try:
            yield
        finally:
            self._nvtx.range_pop()

    def start(self, name: str) -> Optional[int]:
        """Open a range another thread may close (pipelined iterations)."""
        range_id = self._nvtx.range_start(name) if self.active else None

        return range_id

    def end(self, range_id: Optional[int]) -> None:
        if range_id is not None:
            self._nvtx.range_end(range_id)


def _instrument(model: Any, *, tracer: _Nvtx) -> None:
    """NVTX around the model phases (this instance) and block ``run`` (classes).

    In-process diagnostic wrappers only; product code is unchanged.
    """
    for name in ("pre_process", "forward", "post_process"):
        setattr(model, name, _ranged(getattr(model, name), f"model.{name}", tracer))

    import gpu_blocks

    classes = list(gpu_blocks.BLOCKS)
    try:
        import batched_blocks

        classes.extend(getattr(batched_blocks, "BLOCKS", ()))
    except ImportError:
        pass
    for block_class in classes:
        if not getattr(block_class.run, "nvtx_wrapped", False):
            label = getattr(block_class, "type", block_class.__name__)
            block_class.run = _ranged(block_class.run, label, tracer)


def _ranged(function: Callable, label: str, tracer: _Nvtx) -> Callable:
    @wraps(function)
    def ranged(*args: Any, **kwargs: Any) -> Any:
        with tracer.range(label):
            return function(*args, **kwargs)

    ranged.nvtx_wrapped = True

    return ranged


class _FakeDetectionModel:
    """CPU stand-in for smoke tests: real tensor work, fixed detections.

    Three boxes per image, classes 0..2, confidence 0.9, whatever the pixels.
    """

    class_names = ["person", "bicycle", "car"]

    def __init__(self, *, device: str):
        import torch

        self._device = torch.device(device)

    def pre_process(
        self, images: Any, input_color_format: Optional[str] = None, **kwargs: Any
    ) -> Tuple[Any, List[Tuple[int, int]]]:
        import torch
        import torch.nn.functional as functional

        images = [images] if isinstance(images, torch.Tensor) else list(images)
        batch = torch.stack(
            [
                functional.interpolate(image[None].float(), size=(64, 64))[0]
                for image in images
            ]
        )
        sizes = [(int(image.shape[1]), int(image.shape[2])) for image in images]

        return batch / 255.0, sizes

    def forward(self, network_input: Any, **kwargs: Any) -> Any:
        output = network_input.mean(dim=(1, 2, 3))

        return output

    def post_process(
        self, raw: Any, sizes: List[Tuple[int, int]], **kwargs: Any
    ) -> List[Any]:
        import torch

        from inference_models.models.base.object_detection import Detections

        detections = []
        for height, width in sizes:
            corners = torch.tensor([[40, 60], [300, 200], [width - 120, height - 90]])
            xyxy = torch.cat([corners, corners + torch.tensor([100, 80])], dim=1)
            xyxy = xyxy.clamp(min=0).minimum(torch.tensor([width - 1, height - 1] * 2))
            detections.append(
                Detections(
                    xyxy=xyxy.int().to(self._device),
                    class_id=torch.arange(3, dtype=torch.int32, device=self._device),
                    confidence=torch.full((3,), 0.9, device=self._device),
                )
            )

        return detections

    def __call__(self, images: Any, **kwargs: Any) -> List[Any]:
        network_input, sizes = self.pre_process(images, **kwargs)
        detections = self.post_process(self.forward(network_input), sizes, **kwargs)

        return detections


def _environment(torch: Any, *, device: str) -> Dict[str, Any]:
    import os

    is_cuda = torch.device(device).type == "cuda"
    environment = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if is_cuda else None,
        "variables": {name: os.environ.get(name) for name in RECORDED_ENVIRONMENT},
        "clocks": "not read by this script; record them around the run",
    }

    return environment


def _provenance(model: Any) -> Dict[str, Any]:
    """Import path and sha256 of each loaded module that decides the work."""
    sources = {name: _module_path(module) for name, module in list(sys.modules.items())}
    wanted = {*PROVENANCE_MODULES, type(model).__module__}
    examples = EXAMPLES_DIR.resolve()
    provenance = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in sorted(sources.items())
        if path is not None and (name in wanted or examples in path.parents)
    }

    return provenance


def _module_path(module: Any) -> Optional[Path]:
    # Some extension modules report a bare relative file name; skip those.
    path = Path(getattr(module, "__file__", None) or ".")
    source = path if path.is_absolute() and path.is_file() else None

    return source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    return digest


def _one_line(report: Dict[str, Any]) -> str:
    if report.get("error"):
        return f"{report['id']}: FAILED {report['error']}"
    if "profiled_repetition" in report:
        return f"{report['id']}: profiled {report['frames_per_repetition']} frames"

    fps = report["frames_per_second"]
    line = (
        f"{report['id']}: {fps['median']:.1f} frames/s "
        f"({fps['min']:.1f}-{fps['max']:.1f}), "
        f"latency p50 {report['latency_ms']['p50']:.2f} ms, "
        f"cpu {report['cpu_ms_per_frame_median']:.2f} ms/frame, "
        f"forward batch sizes {sorted(set(report['forward_batch_sizes']))}"
    )

    return line


def _str_list(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]

    return items


def _int_list(value: str) -> List[int]:
    numbers = [int(item) for item in _str_list(value)]

    return numbers


def _depth(execution: str) -> int:
    if execution == "serial":
        return 0
    if (
        execution.startswith("pipe")
        and execution[4:].isdigit()
        and int(execution[4:]) > 0
    ):
        return int(execution[4:])

    raise click.BadParameter(f"execution must be serial or pipeN, got {execution!r}")


if __name__ == "__main__":
    main()
