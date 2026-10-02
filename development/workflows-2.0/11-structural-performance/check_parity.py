"""Do the shared drawings produce exactly the per-painter rows?

One pass, two commands::

    fake  ScriptedDetectionModel (0..40 boxes per frame, ties, edge labels,
          80 classes), 20 marker frames of three sizes, on --device
    run   a real model on the frames of a 09 fixture.pt, in a seeded shuffled
          order: TensorRT on cuda, ONNX on cpu (a local check, not TRT parity)

    for mode in --modes (v2_serial: run; v2_pipeline: phases, depth D):
        for drawing in per-painter, shared-prep, batch-painters:
            run the 10 batch plan (e.g. 20 frames at B 8 -> [1, 3, 5, 8, 3])
    reference: per-painter rows of the first mode

Checks, all exact (``torch.equal``), no tolerance:

    rows        every row's xyxy, class_id, confidence and annotated pixels
                equal the reference row of the same frame
    batches     10 batch_checks: forward batch sizes == plan, row identity
                (parent_id, annotated image_id, image size), batch keys
    stress      v2_pipeline: 10 stress_pipeline, depth batches pending in a
                shuffled order, rows equal to the same backend's own rows
    sources     no source frame changed

All modes and drawings share one process and one loaded model: the drawings
differ only after the detector, so the detector runs the same batches.

    python check_parity.py fake --device cpu --output-dir /tmp/m46/parity-fake
    python check_parity.py run --fixture /fixtures/parity/fixture.pt \\
        --output-dir /tmp/m46/parity-trt
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import click
import structural_imports

# isort: split

import drawing_backend
from run_benchmark import DEFAULT_MODEL_ID

PARITY_10 = structural_imports.load_10_module("check_parity")
FAKE_FRAMES = 20
FAKE_FRAME_SIZES_HW = ((360, 640), (288, 512), (540, 960))
V2_MODES = ("v2_serial", "v2_pipeline")


@click.group()
def main() -> None:
    """Exact parity of the shared drawings with the per-painter workflow."""


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
    "--device",
    default="cpu",
    show_default=True,
    help="Device of the frames and the scripted detections.",
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
    default=3,
    show_default=True,
    help="Shuffled passes over the batch plan in the pipeline stress.",
)
def fake(
    output_dir: Path, device: str, batch_size: int, depth: int, rounds: int
) -> None:
    """Scripted model: no download; CPU or GPU."""
    import backends

    backends.configure_mode("v2_serial", device=device)
    from scripted_model import ScriptedDetectionModel

    frames = [
        frame.to(device)
        for frame in PARITY_10.marker_frames(FAKE_FRAMES, sizes_hw=FAKE_FRAME_SIZES_HW)
    ]
    _wait_for_uploads(device)
    ids = [f"s{index % 4}-f{index}" for index in range(len(frames))]

    report = parity_pass(
        ScriptedDetectionModel(device=device),
        frames,
        ids=ids,
        modes=V2_MODES,
        batch_size=batch_size,
        depth=depth,
        rounds=rounds,
    )
    report = {"model": "ScriptedDetectionModel (deterministic); not TRT", **report}
    _write_report(output_dir / "fake.json", report)


@main.command()
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Receives parity.json.",
)
@click.option(
    "--fixture",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="fixture.pt written by 09 check_parity.py capture.",
)
@click.option(
    "--device",
    default="cuda:0",
    show_default=True,
    help="cuda:N loads TRT strictly; cpu loads ONNX for local checks only.",
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
    "--modes",
    default=",".join(V2_MODES),
    show_default=True,
    help="Comma-separated V2 modes; per-painter of the first is the reference.",
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
)
@click.option(
    "--rounds",
    type=click.IntRange(
        min=1,
    ),
    default=4,
    show_default=True,
)
def run(
    output_dir: Path,
    fixture: Path,
    device: str,
    model_id: str,
    confidence: float,
    modes: str,
    batch_size: int,
    depth: int,
    rounds: int,
) -> None:
    """Real model on held fixture frames."""
    selected = [mode.strip() for mode in modes.split(",") if mode.strip()]
    unknown = sorted(set(selected) - set(V2_MODES))
    if unknown or not selected:
        raise click.BadParameter(f"modes must be from {V2_MODES}", param_hint="--modes")

    import backends

    backends.configure_mode("v2_serial", device=device)
    load_fixture = structural_imports.thor_imports.load_09_module(
        "check_parity"
    ).load_fixture
    stored = load_fixture(fixture)["frames"]
    # The 10 worker's seeded order, so batches mix sources.
    order = random.Random(0).sample(range(len(stored)), len(stored))
    frames = [stored[index].to(device).contiguous() for index in order]
    _wait_for_uploads(device)
    ids = [f"parity-{index}" for index in order]
    model, model_backend = PARITY_10._load_model(model_id, device=device)

    report = parity_pass(
        model,
        frames,
        ids=ids,
        modes=selected,
        batch_size=batch_size,
        depth=depth,
        rounds=rounds,
        confidence=confidence,
    )
    report = {
        "model": backends.describe_model(model),
        "model_backend": model_backend,
        "fixture": str(fixture),
        "frame_order": order,
        **report,
    }
    _write_report(output_dir / "parity.json", report)


def parity_pass(
    model: Any,
    frames: Sequence[Any],
    *,
    ids: Sequence[str],
    modes: Sequence[str],
    batch_size: int,
    depth: int,
    rounds: int,
    confidence: float = 0.4,
) -> Dict[str, Any]:
    """Every mode and drawing on the same frames; exact comparison.

    Args:
        model: Loaded detection model with ``_device``; shared by all passes.
        frames: CHW RGB uint8 frames on the model's device.
        ids: Their distinct image ids.
        modes: V2 modes; per-painter of the first is the reference.
        batch_size: Largest batch of the 10 batch plan.
        depth: ``v2_pipeline`` batches in flight.
        rounds: Shuffled passes of the pipeline stress.
        confidence: Detection confidence threshold.

    Returns:
        ``plan``, per pass the 10 batch checks, row mismatches and stress,
        detection counts, ``source_frames_changed``, ``checks``, ``passed``.
    """
    originals = [frame.clone() for frame in frames]
    plan = PARITY_10.batch_plan(len(frames), batch_size=batch_size)
    forward_sizes = PARITY_10._record_forward_sizes(model)

    reference: Optional[List[Dict[str, Any]]] = None
    passes = {}
    for mode in modes:
        for drawing in drawing_backend.DRAWINGS:
            backend = drawing_backend.build_for_model(
                mode,
                model=model,
                model_id="parity",
                confidence=confidence,
                pipeline_depth=depth,
                drawing=drawing,
            )
            try:
                PARITY_10.run_plan(backend, frames, ids=ids, plan=plan)  # warm-up
                forward_sizes.clear()
                rows = PARITY_10.run_plan(backend, frames, ids=ids, plan=plan)
                views = [PARITY_10._comparable(backend, row) for row in rows]
                batches = PARITY_10.batch_checks(
                    backend,
                    rows,
                    frames=frames,
                    ids=ids,
                    plan=plan,
                    forward_sizes=list(forward_sizes),
                )
                stress = None
                if mode == "v2_pipeline":
                    stress = PARITY_10.stress_pipeline(
                        backend,
                        frames,
                        ids=ids,
                        plan=plan,
                        expected=views,
                        depth=depth,
                        rounds=rounds,
                    )
            finally:
                backend.close()
            if reference is None:
                reference = views
            passes[f"{mode}/{drawing}"] = {
                "facts": backend.facts,
                "batches": batches,
                "mismatches": _mismatches(views, reference=reference),
                "stress": stress,
                "detections": [len(view["class_id"]) for view in views],
            }

    report = {
        "reference": f"{modes[0]}/per-painter",
        "frames": len(frames),
        "plan": plan,
        "batch_size": batch_size,
        "depth": depth,
        "comparison": "exact: torch.equal of xyxy, class_id, confidence, annotated",
        "passes": passes,
        "source_frames_changed": PARITY_10._changed(frames, originals=originals),
    }
    report["checks"] = {
        "rows_identical": all(not one["mismatches"] for one in passes.values()),
        "batches": all(one["batches"]["passed"] for one in passes.values()),
        "stress": all(
            one["stress"]["passed"] for one in passes.values() if one["stress"]
        ),
        "sources_unchanged": not report["source_frames_changed"],
    }
    report["passed"] = all(report["checks"].values())

    return report


def _mismatches(
    views: List[Dict[str, Any]], *, reference: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    # Every (frame, output) that is not exactly the reference's.
    import torch

    mismatches = [
        {"frame": position, "output": name}
        for position, (view, expected) in enumerate(zip(views, reference))
        for name in expected
        if name not in view
        or view[name].shape != expected[name].shape
        or not torch.equal(view[name], expected[name])
    ]
    if len(views) != len(reference):
        mismatches.append({"rows": len(views), "expected_rows": len(reference)})

    return mismatches


def _wait_for_uploads(device: str) -> None:
    # Frames are read on other streams; their uploads must be complete.
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    passes = {
        name: "ok" if not one["mismatches"] else f"{len(one['mismatches'])} mismatches"
        for name, one in report["passes"].items()
    }
    click.echo(f"plan {report['plan']}; rows vs {report['reference']}: {passes}")
    click.echo(f"checks {report['checks']} -> {'PASS' if report['passed'] else 'FAIL'}")
    click.echo(f"wrote {path}")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
