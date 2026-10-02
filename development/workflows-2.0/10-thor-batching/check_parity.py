"""Do physical batches compute what one frame at a time computes, frame by frame?

Two commands. Both cut the frames into the same batch plan (``batch_plan``):
at least one full batch of B whenever there are B frames, plus single-frame and
partial batches where they fit, e.g. 16 frames at ``--batch-size 8`` ->
``[1, 3, 8, 4]``.

```
fake  (CPU, seconds)   FakeDetectionModel: deterministic, one box per frame at that
                       frame's marker pixel value, so a row paired with the wrong
                       frame shows. Records every forward input shape and the most
                       concurrent calls of each model phase.
                       Frames of three sizes, mixed inside batches.
                       reference: 09 V2Backend, one frame per call (08 detector)
                       v2_serial and v2_pipeline (depth 2) in batches, stages cut too
run   (Thor, TRT)      fixture from 09 check_parity capture (or --fixture), in a
                       seeded shuffled order, so batches mix sources
                       one subprocess per mode: v1_tensor, v2_serial, v2_pipeline
                       each: batched pass; reference pass with the 09 backend of the
                       same engine, one frame per call; pipeline stress
                       then 09 compare_modes on the batched outputs (exact across modes)
```

Checks per batched pass:

    forward_batches   sizes of the inputs reaching ``model.forward`` == the plan
    identity          every row's ``predictions.image_metadata["parent_id"]`` is the
                      id submitted at its position (and V2 ``annotated.image_id``);
                      its ``image_dimensions`` are its own frame's size
    batch keys        rows of one batch share one ``batch_index`` and its size
    alignment         (fake) every row's box sits at its own frame's marker
    single            batched rows vs the 09 backend, one frame per call: ``fake``
                      needs identical; ``run`` allows identical / within_tolerance /
                      renderer_agree under ``TOLERANCES`` (TRT may pick other kernels
                      at another batch size) and records every non-identical frame
    stress            v2_pipeline only: the same pipeline backend with ``depth``
                      batches pending in a shuffled order must return, per batch,
                      exactly the submitted rows (count and ids) with payloads
                      identical to its own rows when batches went one at a time;
                      overlap observed. Cross-mode agreement is ``single``/``run``.
    phases            (fake) no model phase ever ran twice at once; >= 2 phases did

``run --device cpu`` loads the ONNX model instead of TRT. It exists for local
checks of the V1 comparator only and says so in its report; it is not TRT parity.

    python check_parity.py fake --output-dir /tmp/thor/batched-parity-fake
    python check_parity.py run --batch-size 8 --output-dir /tmp/thor/batched-parity
"""

import json
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import click
import thor_imports  # installs the 08 and 09 search paths

# isort: split

import batched_backend
from run_benchmark import DEFAULT_MODEL_ID, DEFAULT_URL_TEMPLATE

SCRIPT = Path(__file__).resolve()
THOR_PARITY_SCRIPT = thor_imports.THOR_MULTISTREAM_DIR / "check_parity.py"
# 09 check_parity defaults; v1_tensor is its reference, as there.
TOLERANCES = {
    "box_tolerance_px": 2.0,
    "confidence_tolerance": 0.02,
    "threshold_margin": 0.02,
    "shift_px": 1,
    "color_tolerance": 0,
    "max_unexplained_pixels": 0,
}
FAKE_FRAMES = 20
PROBE_BATCH_SIZES = (1, 3, 5)  # single and partial batches, when they fit
FAKE_FRAME_SIZES_HW = ((360, 640), (288, 512), (540, 960))
FAKE_FORWARD_DELAY_S = 0.02
SINGLE_ALLOWED = ("identical", "within_tolerance", "renderer_agree")


@click.group()
def main() -> None:
    """Batched V2 / V1 parity with one-frame-at-a-time results."""


@main.command()
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Receives fake.json.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(
        min=2,
    ),
    default=8,
    show_default=True,
)
@click.option(
    "--depth",
    type=click.IntRange(
        min=2,
    ),
    default=2,
    show_default=True,
    help="v2_pipeline batches in flight.",
)
@click.option(
    "--rounds",
    type=click.IntRange(
        min=1,
    ),
    default=3,
    show_default=True,
    help="Shuffled passes over the batch plan in the pipeline stress.",
)
def fake(output_dir: Path, batch_size: int, depth: int, rounds: int) -> None:
    """CPU proof with a deterministic fake model; no download, no GPU."""
    import backends

    # v2_serial and v2_pipeline need the same process configuration.
    backends.configure_mode("v2_serial", device="cpu")

    frames = marker_frames(FAKE_FRAMES, sizes_hw=FAKE_FRAME_SIZES_HW)
    originals = [frame.clone() for frame in frames]
    ids = [f"s{index % 4}-f{index}" for index in range(len(frames))]
    plan = batch_plan(len(frames), batch_size=batch_size)

    single_model = FakeDetectionModel()
    single = backends._build_for_model(
        "v2_serial",
        model=single_model,
        model_id="fake",
        confidence=0.4,
        max_in_flight=1,
    )
    reference = [
        _comparable(single, single.process(frame, image_id=image_id))
        for frame, image_id in zip(frames, ids)
    ]

    passes = {}
    for mode in ("v2_serial", "v2_pipeline"):
        model = FakeDetectionModel(forward_delay_s=FAKE_FORWARD_DELAY_S)
        backend = batched_backend.build_for_model(
            mode, model=model, model_id="fake", confidence=0.4, pipeline_depth=depth
        )
        try:
            rows = run_plan(backend, frames, ids=ids, plan=plan)
            forward_sizes = [shape[0] for shape in model.forward_shapes]
            stress = None
            if mode == "v2_pipeline":
                expected = [_comparable(backend, row) for row in rows]
                stress = stress_pipeline(
                    backend,
                    frames,
                    ids=ids,
                    plan=plan,
                    expected=expected,
                    depth=depth,
                    rounds=rounds,
                )
        finally:
            backend.close()
        passes[mode] = {
            **batch_checks(
                backend,
                rows,
                frames=frames,
                ids=ids,
                plan=plan,
                forward_sizes=forward_sizes,
            ),
            "alignment_errors": _alignment_errors(backend, rows, frames=frames),
            "single": _single_statuses(
                backend, rows, reference=reference, sources=frames, exact=True
            ),
            "stress": stress,
            "phase_concurrency": dict(model.max_concurrent),
        }

    stages = {}
    for stage in ("detector", "boxes"):
        backend = batched_backend.build_for_model(
            "v2_serial",
            model=FakeDetectionModel(),
            model_id="fake",
            confidence=0.4,
            stages=stage,
        )
        rows = run_plan(backend, frames, ids=ids, plan=plan)
        stages[stage] = {
            "row_keys": sorted(rows[0]),
            "annotated_present": all(
                backend.annotated(row) is not None for row in rows
            ),
            "predictions_identical": all(
                _same_predictions(_comparable(backend, row), expected)
                for row, expected in zip(rows, reference)
            ),
        }

    report = {
        "model": "FakeDetectionModel (CPU, deterministic); not TRT",
        "frames": len(frames),
        "plan": plan,
        "depth": depth,
        "passes": passes,
        "stages": stages,
        "source_frames_changed": _changed(frames, originals=originals),
    }
    report["checks"] = _fake_checks(report)
    report["passed"] = all(report["checks"].values())
    click.echo(f"checks {report['checks']} -> {'PASS' if report['passed'] else 'FAIL'}")
    _write_report(output_dir / "fake.json", report)


@main.command()
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Receives fixture.pt (when captured), <mode>/ and parity.json.",
)
@click.option(
    "--fixture",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    default=None,
    help="A 09 check_parity fixture; captured from the sources when omitted.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(
        min=2,
    ),
    default=8,
    show_default=True,
)
@click.option(
    "--depth",
    type=click.IntRange(
        min=2,
    ),
    default=2,
    show_default=True,
    help="v2_pipeline batches in flight, also in the stress pass.",
)
@click.option(
    "--rounds",
    type=click.IntRange(
        min=1,
    ),
    default=4,
    show_default=True,
    help="Shuffled passes over the batch plan in the pipeline stress.",
)
@click.option(
    "--modes",
    default=",".join(batched_backend.MODES),
    show_default=True,
    help="Comma-separated modes; the first is the reference.",
)
@click.option(
    "--sources",
    type=click.IntRange(
        min=1,
    ),
    default=4,
    show_default=True,
    help="Sources to capture from (09 capture).",
)
@click.option(
    "--frames",
    type=click.IntRange(
        min=3,
    ),
    default=24,
    show_default=True,
    help="Frames to capture (09 capture).",
)
@click.option(
    "--url-template",
    default=DEFAULT_URL_TEMPLATE,
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
    "--device",
    default="cuda:0",
    show_default=True,
    help="cuda:N loads TRT strictly; cpu loads ONNX for local checks only.",
)
def run(output_dir: Path, fixture: Optional[Path], **options: Any) -> None:
    """Real model: one subprocess per mode, then compare."""
    modes = [mode.strip() for mode in options["modes"].split(",") if mode.strip()]
    unknown = sorted(set(modes) - set(batched_backend.MODES))
    if unknown:
        raise click.BadParameter(f"unknown modes {unknown}", param_hint="--modes")
    output_dir.mkdir(parents=True, exist_ok=True)

    if fixture is None:
        fixture = output_dir / "fixture.pt"
        _run_script(
            THOR_PARITY_SCRIPT,
            "capture",
            "--out",
            str(fixture),
            "--url-template",
            options["url_template"],
            "--sources",
            str(options["sources"]),
            "--frames",
            str(options["frames"]),
            "--stride",
            "15",
            "--device",
            options["device"],
        )
    for mode in modes:
        _run_script(
            SCRIPT,
            "worker",
            "--mode",
            mode,
            "--fixture",
            str(fixture),
            "--out-dir",
            str(output_dir / mode),
            "--model-id",
            options["model_id"],
            "--confidence",
            str(options["confidence"]),
            "--device",
            options["device"],
            "--batch-size",
            str(options["batch_size"]),
            "--depth",
            str(options["depth"]),
            "--rounds",
            str(options["rounds"]),
        )

    thor = _thor_parity()
    compare_options = {**TOLERANCES, **options, "reference": modes[0]}
    report = thor.compare_modes(
        output_dir, fixture=fixture, modes=modes, options=compare_options
    )
    report["checks"].update(
        {
            "batches": all(
                worker["batches"]["passed"] for worker in report["workers"].values()
            ),
            "single_within_tolerance": all(
                worker["single"]["passed"] for worker in report["workers"].values()
            ),
        }
    )
    report["passed"] = all(report["checks"].values())
    thor._print_report(report)
    _write_report(output_dir / "parity.json", report)


@main.command(hidden=True)
@click.option(
    "--mode",
    type=click.Choice(batched_backend.MODES),
    required=True,
)
@click.option(
    "--fixture",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--out-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--model-id",
    required=True,
)
@click.option(
    "--confidence",
    type=float,
    required=True,
)
@click.option(
    "--device",
    required=True,
)
@click.option(
    "--batch-size",
    type=int,
    required=True,
)
@click.option(
    "--depth",
    type=int,
    required=True,
)
@click.option(
    "--rounds",
    type=int,
    required=True,
)
def worker(
    mode: str,
    *,
    fixture: Path,
    out_dir: Path,
    model_id: str,
    confidence: float,
    device: str,
    batch_size: int,
    depth: int,
    rounds: int,
) -> None:
    """Run one mode on the fixture (runs in its own process)."""
    import backends

    backends.configure_mode(mode, device=device)

    import torch

    thor = _thor_parity()
    fixture_frames = thor.load_fixture(fixture)["frames"]
    # The same seeded order in every worker; outputs are saved in fixture order.
    order = random.Random(0).sample(range(len(fixture_frames)), len(fixture_frames))
    originals = [fixture_frames[index] for index in order]
    held = [frame.to(device) for frame in originals]
    if torch.device(device).type == "cuda":
        # Uploads must be complete before other streams read the frames.
        torch.cuda.current_stream(device).synchronize()
    ids = [f"parity-{index}" for index in order]
    plan = batch_plan(len(held), batch_size=batch_size)

    model, model_backend = _load_model(model_id, device=device)
    forward_sizes = _record_forward_sizes(model)
    backend = batched_backend.build_for_model(
        mode,
        model=model,
        model_id=model_id,
        confidence=confidence,
        pipeline_depth=depth,
    )
    try:
        run_plan(backend, held, ids=ids, plan=plan)  # warm-up, all batch sizes
        forward_sizes.clear()
        rows = run_plan(backend, held, ids=ids, plan=plan)
        batched_forward_sizes = list(forward_sizes)
        batched = [_comparable(backend, row) for row in rows]
        forward_sizes.clear()
        singles = _single_frame_reference(
            mode, model=model, model_id=model_id, confidence=confidence, frames=held
        )
        single_forward_sizes = list(forward_sizes)
        stress = None
        if mode == "v2_pipeline":
            stress = stress_pipeline(
                backend,
                held,
                ids=ids,
                plan=plan,
                expected=batched,
                depth=depth,
                rounds=rounds,
            )
        worker_report = {
            "mode": mode,
            "model_backend": model_backend,
            "backend": backend.facts,
            "frames": len(held),
            "frame_order": order,
            "batches": batch_checks(
                backend,
                rows,
                frames=held,
                ids=ids,
                plan=plan,
                forward_sizes=batched_forward_sizes,
            ),
            "single": {
                **_single_statuses(
                    backend, rows, reference=singles, sources=held, exact=False
                ),
                "forward_batch_sizes": single_forward_sizes,
            },
            "stress": stress,
        }
    finally:
        backend.close()

    worker_report["source_frames_changed"] = _changed(held, originals=originals)
    in_fixture_order = [view for _, view in sorted(zip(order, batched))]
    worker_report["detections"] = [len(view["class_id"]) for view in in_fixture_order]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            {name: value.cpu() for name, value in view.items()}
            for view in in_fixture_order
        ],
        out_dir / "outputs.pt",
    )
    (out_dir / "worker.json").write_text(
        json.dumps(worker_report, indent=2, default=str)
    )
    click.echo(
        f"{mode} ({model_backend}): plan {plan}, forward {batched_forward_sizes}, "
        f"batches {'ok' if worker_report['batches']['passed'] else 'FAIL'}, "
        f"single {worker_report['single']['counts']}"
    )


class FakeDetectionModel:
    """Deterministic CPU stand-in for the TRT detector, with call records.

    The output of a frame is its marker, the value of pixel ``(0, 0, 0)``:
    one detection with box ``(m, m, m + 60, m + 40)`` and class ``m % 3``.
    Like the TRT model, ``pre_process`` takes one tensor or a list and
    ``forward`` takes one stacked batch.

    Args:
        forward_delay_s: Sleep inside ``forward``, so phases of pipelined
            batches have time to overlap on a CPU.
    """

    class_names = ["person", "car", "dog"]

    def __init__(self, *, forward_delay_s: float = 0.0):
        import torch

        self._device = torch.device("cpu")
        self.forward_shapes: List[Tuple[int, ...]] = []
        self.max_concurrent: Counter = Counter()  # per phase, and "any"
        self._active: Counter = Counter()
        self._lock = threading.Lock()
        self._forward_delay_s = forward_delay_s

    def pre_process(self, images: Any, input_color_format: Optional[str] = None) -> Any:
        """Stack the top-left 8 x 8 pixels of every image; metadata is its size."""
        import torch

        with self._tracking("pre_process"):
            images = images if isinstance(images, list) else [images]
            batch = torch.stack([image[:, :8, :8].float() for image in images])
            metadata = [tuple(image.shape[1:]) for image in images]

        return batch, metadata

    def forward(self, batch: Any) -> Any:
        """Record the input shape; return each image's marker."""
        with self._tracking("forward"):
            self.forward_shapes.append(tuple(batch.shape))
            time.sleep(self._forward_delay_s)
            markers = batch[:, 0, 0, 0].clone()

        return markers

    def post_process(self, markers: Any, metadata: Any, confidence: float) -> list:
        """One ``Detections`` per marker, in batch order."""
        with self._tracking("post_process"):
            detections = [_marker_detections(int(marker)) for marker in markers]

        return detections

    @contextmanager
    def _tracking(self, phase: str) -> Iterator[None]:
        with self._lock:
            self._active[phase] += 1
            self._active["any"] += 1
            for name in (phase, "any"):
                self.max_concurrent[name] = max(
                    self.max_concurrent[name], self._active[name]
                )
        try:
            yield
        finally:
            with self._lock:
                self._active[phase] -= 1
                self._active["any"] -= 1


def marker_frames(
    count: int, *, sizes_hw: Sequence[Tuple[int, int]] = FAKE_FRAME_SIZES_HW
) -> List[Any]:
    """CPU CHW RGB frames; frame ``k`` has marker ``10 + 7k`` at pixel (0, 0, 0).

    Args:
        count: Number of frames, at most 30 (markers stay inside the frame).
        sizes_hw: Frame heights and widths, used in turn; at least 300 x 300.

    Returns:
        Distinct frames: a per-frame gradient plus the marker.
    """
    import torch

    frames = []
    for index in range(count):
        height, width = sizes_hw[index % len(sizes_hw)]
        rows = torch.arange(height).view(1, height, 1)
        columns = torch.arange(width).view(1, 1, width)
        frame = ((rows + columns * (index + 1)) % 200).to(torch.uint8).expand(3, -1, -1)
        frame = frame.contiguous()
        frame[0, 0, 0] = 10 + 7 * index
        frames.append(frame)

    return frames


def batch_plan(count: int, *, batch_size: int) -> List[int]:
    """Batch sizes covering ``count`` frames: probes, one full batch, the rest.

    Probe batches of 1, 3 and 5 frames (those below B) come first, but only
    while one full batch of B still fits after them; then come full batches
    and one remainder. Each frame is in exactly one batch, in order.

    Args:
        count: Number of frames.
        batch_size: Largest batch, B.

    Returns:
        Positive sizes summing to ``count``. At B = 8: 16 frames ->
        ``[1, 3, 8, 4]``, 20 -> ``[1, 3, 5, 8, 3]``, 24 -> ``[1, 3, 5, 8, 7]``,
        8 -> ``[8]``, 7 -> ``[1, 3, 3]`` (no full batch fits).
    """
    probe_room = count - batch_size if count >= batch_size else count
    plan: List[int] = []
    for size in PROBE_BATCH_SIZES:
        if size < batch_size and size <= probe_room:
            plan.append(size)
            probe_room -= size
    remaining = count - sum(plan)
    while remaining > 0:
        plan.append(min(batch_size, remaining))
        remaining -= plan[-1]

    return plan


def run_plan(
    backend: Any, frames: Sequence[Any], *, ids: Sequence[str], plan: Sequence[int]
) -> List[dict]:
    """Run consecutive chunks of ``frames`` as batches of the planned sizes.

    Args:
        backend: A ``batched_backend.BatchedBackend``.
        frames: Frames, as many as the plan covers.
        ids: Their image ids.
        plan: Batch sizes in order.

    Returns:
        Every row, in frame order.
    """
    rows: List[dict] = []
    for start, size in _chunks(plan):
        rows.extend(
            backend.process_batch(
                frames[start : start + size], image_ids=ids[start : start + size]
            )
        )

    return rows


def batch_checks(
    backend: Any,
    rows: List[dict],
    *,
    frames: Sequence[Any],
    ids: Sequence[str],
    plan: Sequence[int],
    forward_sizes: Sequence[int],
) -> Dict[str, Any]:
    """Forward batch sizes, batch keys and per-row identity of one pass.

    Args:
        backend: The backend that produced ``rows``.
        rows: Rows of ``run_plan``.
        frames: The submitted frames, in order.
        ids: The submitted ids, in order.
        plan: The batch plan of the pass.
        forward_sizes: Sizes of the inputs ``model.forward`` received.

    Returns:
        The observations, the errors found and ``passed``.
    """
    identity_errors = []
    for position, (row, frame, image_id) in enumerate(zip(rows, frames, ids)):
        metadata = backend.predictions(row).image_metadata or {}
        seen = {
            "ids": _row_ids(backend, row),
            "image_dimensions": metadata.get("image_dimensions"),
        }
        expected = {
            "ids": (image_id, image_id),
            "image_dimensions": list(frame.shape[1:]),
        }
        if seen != expected:
            identity_errors.append({"position": position, "seen": seen})
    # Per planned batch: the distinct (batch_index, batch_size) its rows carry.
    seen_batches = [
        sorted(
            {
                (row["batch_index"], row[batched_backend.BATCH_SIZE_KEY])
                for row in rows[start : start + size]
            }
        )
        for start, size in _chunks(plan)
    ]
    checks = {
        "plan": list(plan),
        "forward_batch_sizes": list(forward_sizes),
        "rows": len(rows),
        "identity_errors": identity_errors,
        # One batch_index per planned batch, distinct, with the planned size.
        "batch_keys_match_plan": (
            all(len(keys) == 1 for keys in seen_batches)
            and [keys[0][1] for keys in seen_batches] == list(plan)
            and len({keys[0][0] for keys in seen_batches}) == len(plan)
        ),
    }
    checks["passed"] = (
        checks["forward_batch_sizes"] == checks["plan"]
        and len(rows) == len(ids)
        and not identity_errors
        and checks["batch_keys_match_plan"]
    )

    return checks


def stress_pipeline(
    backend: Any,
    frames: Sequence[Any],
    *,
    ids: Sequence[str],
    plan: Sequence[int],
    expected: List[Dict[str, Any]],
    depth: int,
    rounds: int,
) -> Dict[str, Any]:
    """Keep ``depth`` batches pending over a shuffled batch order; compare.

    Batches keep their composition. Each future must return exactly its
    batch's rows: one per submitted frame, carrying the submitted ids in
    order. Only then are payloads compared, each with ``expected`` for its frame.

    Args:
        backend: A ``v2_pipeline`` backend.
        frames: The frames of ``expected``, in order.
        ids: Their ids; stress submissions use ``stress-<id>``.
        plan: The batch plan; each batch keeps its frames.
        expected: The same backend's views per frame, batches one at a time.
        depth: Batches kept pending.
        rounds: Shuffled passes over the plan.

    Returns:
        Submissions, overlap evidence, row-count and identity errors,
        payload mismatches and ``passed``.
    """
    chunks = list(_chunks(plan))
    shuffler = random.Random(0)
    order = [
        chunk for _ in range(rounds) for chunk in shuffler.sample(chunks, len(chunks))
    ]
    pending: List[Tuple[int, List[str], Future]] = []
    not_done_after_submit: List[int] = []
    row_count_errors: List[dict] = []
    identity_errors: List[dict] = []
    mismatches: List[dict] = []

    def check_oldest() -> None:
        start, submitted_ids, future = pending.pop(0)
        rows = future.result()
        if len(rows) != len(submitted_ids):
            row_count_errors.append(
                {"start": start, "expected": len(submitted_ids), "returned": len(rows)}
            )
            return

        seen_ids = [_row_ids(backend, row) for row in rows]
        if seen_ids != [(image_id, image_id) for image_id in submitted_ids]:
            identity_errors.append({"start": start, "seen": seen_ids})
            return

        for offset, row in enumerate(rows):
            actual = _comparable(backend, row)
            for name, value in expected[start + offset].items():
                if not _same(actual[name], value):
                    mismatches.append({"frame": start + offset, "output": name})

    for start, size in order:
        while len(pending) >= depth:
            check_oldest()
        submitted_ids = [f"stress-{image_id}" for image_id in ids[start : start + size]]
        future = backend.submit_batch(
            frames[start : start + size], image_ids=submitted_ids
        )
        pending.append((start, submitted_ids, future))
        not_done_after_submit.append(sum(not future.done() for _, _, future in pending))
    while pending:
        check_oldest()

    stress = {
        "submissions": len(order),
        "depth": depth,
        "max_pending_not_done_after_submit": max(not_done_after_submit),
        "row_count_errors": row_count_errors,
        "identity_errors": identity_errors,
        "mismatches": mismatches,
        "overlap_observed": max(not_done_after_submit) >= 2,
    }
    stress["passed"] = (
        not row_count_errors
        and not identity_errors
        and not mismatches
        and stress["overlap_observed"]
    )

    return stress


def _row_ids(backend: Any, row: dict) -> Tuple[Any, Any]:
    # (predictions parent_id, annotated image_id); V1 rows have no annotated
    # image_id, so the parent_id stands in for it.
    parent_id = (backend.predictions(row).image_metadata or {}).get("parent_id")
    annotated_id = getattr(row.get("annotated"), "image_id", parent_id)

    return parent_id, annotated_id


def _single_frame_reference(
    mode: str,
    *,
    model: Any,
    model_id: str,
    confidence: float,
    frames: Sequence[Any],
) -> List[Dict[str, Any]]:
    # The 09 backend of the same engine, one frame per call: V1Backend for
    # v1_tensor, else V2Backend (08 detector) as v2_serial.
    import backends

    old_mode = "v1_tensor" if mode == "v1_tensor" else "v2_serial"
    old = backends._build_for_model(
        old_mode,
        model=model,
        model_id=model_id,
        confidence=confidence,
        max_in_flight=1,
    )
    try:
        views = [
            _comparable(old, old.process(frame, image_id=f"single-{position}"))
            for position, frame in enumerate(frames)
        ]
    finally:
        old.close()

    return views


def _chunks(plan: Sequence[int]) -> Iterator[Tuple[int, int]]:
    start = 0
    for size in plan:
        yield start, size
        start += size


def _comparable(backend: Any, row: Any) -> Dict[str, Any]:
    # 09 comparable() without the annotated image when a stage has none.
    predictions = backend.predictions(row)
    view = {
        "xyxy": predictions.xyxy.float(),
        "class_id": predictions.class_id.long(),
        "confidence": predictions.confidence.float(),
    }
    annotated = backend.annotated(row)
    if annotated is not None:
        view["annotated"] = annotated

    return view


def _single_statuses(
    backend: Any,
    rows: List[dict],
    *,
    reference: List[dict],
    sources: Sequence[Any],
    exact: bool,
) -> Dict[str, Any]:
    # Batched rows vs one frame per call, with 09's status levels.
    thor = _thor_parity()
    options = {**TOLERANCES, "confidence": 0.4}
    statuses = []
    for row, expected, source in zip(rows, reference, sources):
        actual = _comparable(backend, row)
        prediction = thor.compare_predictions(expected, actual, options=options)
        image, _ = thor.compare_images(
            expected["annotated"],
            actual["annotated"],
            source=source.to(expected["annotated"].device),
            options=options,
        )
        statuses.append((prediction, image))
    allowed = ("identical",) if exact else SINGLE_ALLOWED
    counts = Counter(f"{p['status']}/{i['status']}" for p, i in statuses)
    single = {
        "required": "identical" if exact else "/".join(SINGLE_ALLOWED),
        "tolerances": TOLERANCES,
        "counts": dict(sorted(counts.items())),
        "not_identical": [
            {"position": position, "predictions": prediction, "image": image}
            for position, (prediction, image) in enumerate(statuses)
            if (prediction["status"], image["status"]) != ("identical", "identical")
        ],
        "passed": all(
            summary["status"] in allowed for pair in statuses for summary in pair
        ),
    }

    return single


def _alignment_errors(
    backend: Any, rows: List[dict], *, frames: Sequence[Any]
) -> List[int]:
    # Fake model only: the one box of row k must sit at frame k's marker.
    errors = []
    for position, (row, frame) in enumerate(zip(rows, frames)):
        marker = int(frame[0, 0, 0])
        if backend.predictions(row).xyxy.tolist() != _marker_box(marker):
            errors.append(position)

    return errors


def _marker_box(marker: int) -> List[List[int]]:
    box = [[marker, marker, marker + 60, marker + 40]]

    return box


def _marker_detections(marker: int) -> Any:
    import torch

    from inference_models.models.base.object_detection import Detections

    detections = Detections(
        xyxy=torch.tensor(_marker_box(marker), dtype=torch.int32),
        class_id=torch.tensor([marker % 3], dtype=torch.int32),
        confidence=torch.tensor([0.5 + (marker % 50) / 100.0]),
    )

    return detections


def _fake_checks(report: Dict[str, Any]) -> Dict[str, bool]:
    passes = report["passes"]
    stages = report["stages"]
    checks = {
        "batches": all(entry["passed"] for entry in passes.values()),
        "alignment": all(not entry["alignment_errors"] for entry in passes.values()),
        "single_identical": all(entry["single"]["passed"] for entry in passes.values()),
        "stress": passes["v2_pipeline"]["stress"]["passed"],
        "no_concurrent_same_phase": all(
            entry["phase_concurrency"][phase] == 1
            for entry in passes.values()
            for phase in ("pre_process", "forward", "post_process")
        ),
        "pipeline_phases_overlap": passes["v2_pipeline"]["phase_concurrency"]["any"]
        >= 2,
        "stages": (
            not stages["detector"]["annotated_present"]
            and stages["boxes"]["annotated_present"]
            and all(entry["predictions_identical"] for entry in stages.values())
        ),
        "sources_unchanged": not report["source_frames_changed"],
    }

    return checks


def _same_predictions(actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    same = all(
        _same(actual[name], expected[name])
        for name in ("xyxy", "class_id", "confidence")
    )

    return same


def _same(a: Any, b: Any) -> bool:
    same = a.shape == b.shape and bool((a == b.to(a.device)).all())

    return same


def _changed(frames: Sequence[Any], *, originals: Sequence[Any]) -> List[int]:
    import torch

    changed = [
        index
        for index, (frame, original) in enumerate(zip(frames, originals))
        if not torch.equal(frame.cpu(), original.cpu())
    ]

    return changed


def _load_model(model_id: str, *, device: str) -> Tuple[Any, str]:
    # TRT, strictly, on CUDA; ONNX on the CPU for local checks only.
    import backends
    import torch

    if torch.device(device).type == "cuda":
        model = backends.load_trt_model(model_id, device=device)
        return model, "trt"

    from inference_models import AutoModel

    model = AutoModel.from_pretrained(
        model_id,
        backend="onnx",
        device=torch.device("cpu"),
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    return model, "onnx-cpu (local check, not TRT)"


def _record_forward_sizes(model: Any) -> List[int]:
    # Diagnostic instance-attribute wrap, as 09 run_batched._observe_forward_batches.
    sizes: List[int] = []
    original = model.forward

    def observed(pre_processed: Any, *args: Any, **kwargs: Any) -> Any:
        sizes.append(int(pre_processed.shape[0]))
        return original(pre_processed, *args, **kwargs)

    model.forward = observed

    return sizes


def _thor_parity() -> Any:
    thor = thor_imports.load_09_module("check_parity")

    return thor


def _run_script(script: Path, *arguments: str) -> None:
    command = [sys.executable, str(script), *arguments]
    click.echo(
        f"$ {script.parent.name}/{script.name} {' '.join(arguments[:3])} ...", err=True
    )
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise click.ClickException(
            f"{arguments[0]} failed: exit {completed.returncode}"
        )


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    click.echo(f"wrote {path}")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
