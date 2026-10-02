"""Do the four modes compute the same thing on the same GPU frames?

```
capture (subprocess)      NVDEC sources -> every --stride-th frame -> fixture.pt (CPU uint8 CHW RGB)
worker  (one subprocess   fixture -> GPU frames, held for the whole run
         per mode)          serial:  process() each frame once      -> outputs.pt
                            stress:  v2_pipeline only, --depth futures pending,
                                     --rounds shuffled passes, each result
                                     compared with the serial one in-process
                            source:  held frames still equal the fixture
run     (this process)    compare every mode with --reference -> parity.json
```

One mode per process, because ``backends.configure_mode`` fixes the Workflows
configuration once per process. ``v1_numpy`` copies the frame to the host
inside the backend, exactly as in the benchmark.

Agreement levels, per frame and mode pair:

    predictions  identical       all tensors equal
                 within_tolerance every detection matched (same class, IoU > 0.5)
                                 with box and confidence differences in bounds;
                                 unmatched only below confidence + margin
                 differ          anything else
    image        identical       all pixels equal
                 renderer_agree  every differing pixel's colour, in both images,
                                 appears in the other image within --shift-px
                                 (default 1 px) and --color-tolerance (default 0);
                                 at most --max-unexplained-pixels (default 0)
                                 pixels fail this. A box or label drawn 1 px off
                                 passes; another colour, font or size does not
                 differ          anything else

Every non-identical image pair writes ``diffs/<pair>/frame_NN_diff.png``
(yellow: differs but explained by a <= shift-px move; red: unexplained). A
failing frame also writes its ``_expected.png`` and ``_actual.png``.

Pairs of GPU modes must be ``identical``. Pairs with ``v1_numpy`` may be
``within_tolerance`` / ``renderer_agree``: the stock numpy renderer draws boxes
one pixel off the GPU painters. All numbers are written; nothing is dropped.

    python check_parity.py run --output-dir /tmp/thor/parity
    python check_parity.py run --fixture /tmp/thor/parity/fixture.pt --output-dir ...
"""

import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from run_benchmark import DEFAULT_MODEL_ID, DEFAULT_URL_TEMPLATE, MODES

SCRIPT = Path(__file__).resolve()
GPU_MODES = ("v1_tensor", "v2_serial", "v2_pipeline")
CAPTURE_TIMEOUT_SECONDS = 60.0
STOP_TIMEOUT_SECONDS = 10.0
MATCH_IOU = 0.5


@click.group()
def main() -> None:
    """Prediction and image parity across modes on held GPU frames."""


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
        dir_okay=False,
        path_type=Path,
    ),
    default=None,
    help="Existing fixture.pt to reuse instead of capturing new frames.",
)
@click.option(
    "--url-template",
    default=DEFAULT_URL_TEMPLATE,
    show_default=True,
    help="Stream reference with {index}; used only when capturing.",
)
@click.option(
    "--sources",
    type=click.IntRange(
        min=1,
    ),
    default=4,
    show_default=True,
    help="Sources to capture from.",
)
@click.option(
    "--frames",
    type=click.IntRange(
        min=1,
    ),
    default=16,
    show_default=True,
    help="Frames to capture, spread evenly over the sources.",
)
@click.option(
    "--stride",
    type=click.IntRange(
        min=1,
    ),
    default=15,
    show_default=True,
    help="Keep every stride-th decoded frame of a source, so frames differ.",
)
@click.option(
    "--modes",
    default=",".join(MODES),
    show_default=True,
    help="Comma-separated modes to run.",
)
@click.option(
    "--reference",
    type=click.Choice(MODES),
    default="v1_tensor",
    show_default=True,
    help="Mode every other mode is compared with.",
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
)
@click.option(
    "--depth",
    type=click.IntRange(
        min=2,
    ),
    default=4,
    show_default=True,
    help="v2_pipeline max_in_flight and pending futures in the stress pass.",
)
@click.option(
    "--rounds",
    type=click.IntRange(
        min=1,
    ),
    default=4,
    show_default=True,
    help="Shuffled passes over the fixture in the v2_pipeline stress pass.",
)
@click.option(
    "--box-tolerance-px",
    type=float,
    default=2.0,
    show_default=True,
    help="Largest box coordinate difference for within_tolerance.",
)
@click.option(
    "--confidence-tolerance",
    type=float,
    default=0.02,
    show_default=True,
    help="Largest confidence difference for within_tolerance.",
)
@click.option(
    "--threshold-margin",
    type=float,
    default=0.02,
    show_default=True,
    help="Unmatched detections are tolerated only below confidence + margin.",
)
@click.option(
    "--shift-px",
    type=click.IntRange(
        min=0,
    ),
    default=1,
    show_default=True,
    help="How far a differing pixel's colour may move between renderers.",
)
@click.option(
    "--color-tolerance",
    type=click.IntRange(
        min=0,
        max=255,
    ),
    default=0,
    show_default=True,
    help="Largest per-channel difference that counts as the same colour.",
)
@click.option(
    "--max-unexplained-pixels",
    type=click.IntRange(
        min=0,
    ),
    default=0,
    show_default=True,
    help="Differing pixels per frame allowed without a colour match nearby.",
)
def run(output_dir: Path, fixture: Optional[Path], **options: Any) -> None:
    """Capture or load a fixture, run each mode in its own process, compare."""
    modes = [mode.strip() for mode in options["modes"].split(",") if mode.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise click.BadParameter(f"unknown modes {unknown}", param_hint="--modes")
    if options["reference"] not in modes:
        raise click.BadParameter("must be one of --modes", param_hint="--reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    if fixture is None:
        fixture = output_dir / "fixture.pt"
        _run_step(
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
            str(options["stride"]),
            "--device",
            options["device"],
        )
    for mode in modes:
        _run_step(
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
            "--depth",
            str(options["depth"]),
            "--rounds",
            str(options["rounds"]),
        )

    report = compare_modes(output_dir, fixture=fixture, modes=modes, options=options)
    (output_dir / "parity.json").write_text(json.dumps(report, indent=2))
    _print_report(report)
    click.echo(f"wrote {output_dir / 'parity.json'}")
    if not report["passed"]:
        sys.exit(1)


@main.command(hidden=True)
@click.option(
    "--out",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--url-template",
    required=True,
)
@click.option(
    "--sources",
    type=int,
    required=True,
)
@click.option(
    "--frames",
    type=int,
    required=True,
)
@click.option(
    "--stride",
    type=int,
    required=True,
)
@click.option(
    "--device",
    required=True,
)
def capture(
    out: Path, *, url_template: str, sources: int, frames: int, stride: int, device: str
) -> None:
    """Save ``frames`` NVDEC frames as a CPU fixture (runs in its own process)."""
    import time

    import torch
    from sources import JetsonSourceSet

    urls = [url_template.format(index=index) for index in range(sources)]
    per_source = -(-frames // sources)
    source_set = JetsonSourceSet(urls, device=device)
    seen = [0] * sources
    kept: List[List[Any]] = [[] for _ in range(sources)]
    deadline = time.monotonic() + CAPTURE_TIMEOUT_SECONDS
    try:
        source_set.start()
        while sum(len(frames_of_source) for frames_of_source in kept) < frames:
            if time.monotonic() > deadline:
                raise click.ClickException("capture timed out")
            with source_set.wakeup:
                arrived = source_set.take_next_locked()
                if arrived is None:
                    source_set.wakeup.wait(timeout=0.1)
                    end = source_set.first_end_locked()
                    if end is not None:
                        raise click.ClickException(f"source ended: {end.detail}")
                    continue
            index = arrived.source_index
            seen[index] += 1
            if seen[index] % stride == 0 and len(kept[index]) < per_source:
                kept[index].append((arrived.frame.frame_id, arrived.frame.image.cpu()))
            del arrived
    finally:
        stop_report = source_set.stop(timeout_seconds=STOP_TIMEOUT_SECONDS)

    # Interleave sources so consecutive fixture frames come from different streams.
    ordered = []
    for position in range(per_source):
        for index in range(sources):
            if position < len(kept[index]):
                frame_id, image = kept[index][position]
                ordered.append((index, int(frame_id), image))
    ordered = ordered[:frames]
    fixture = {
        "frames": [image for _, _, image in ordered],
        "source_index": [index for index, _, _ in ordered],
        "frame_id": [frame_id for _, frame_id, _ in ordered],
        "urls": [source.display_url for source in source_set.sources],
        "stride": stride,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, out)
    click.echo(f"captured {len(ordered)} frames to {out}; stop {stop_report}")


@main.command(hidden=True)
@click.option(
    "--mode",
    type=click.Choice(MODES),
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
    depth: int,
    rounds: int,
) -> None:
    """Run one mode on the fixture (runs in its own process)."""
    import backends

    backends.configure_mode(mode, device=device)

    import torch

    originals = load_fixture(fixture)["frames"]
    held = [frame.to(device) for frame in originals]
    # The model and painters read frames on their own streams; the uploads must
    # be complete first. Waits on the upload stream only, not the device.
    torch.cuda.current_stream(device).synchronize()
    backend = backends.build_backend(
        mode, model_id, confidence, device=device, max_in_flight=depth
    )
    try:
        backend.warm_up(size_hw=tuple(held[0].shape[1:]))
        serial = [
            comparable(backend, backend.process(frame, image_id=f"serial-{index}"))
            for index, frame in enumerate(held)
        ]
        stress = None
        if mode == "v2_pipeline":
            stress = stress_pipeline(
                backend, held, serial=serial, depth=depth, rounds=rounds
            )
        facts = backend.facts
    finally:
        backend.close()

    changed = [
        index
        for index, (frame, original) in enumerate(zip(held, originals))
        if not torch.equal(frame.cpu(), original)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        [{name: value.cpu() for name, value in entry.items()} for entry in serial],
        out_dir / "outputs.pt",
    )
    worker_report = {
        "mode": mode,
        "backend": facts,
        "frames": len(held),
        "detections": [len(entry["class_id"]) for entry in serial],
        "source_frames_changed": changed,
        "stress": stress,
    }
    (out_dir / "worker.json").write_text(
        json.dumps(worker_report, indent=2, default=str)
    )
    click.echo(f"{mode}: {len(held)} frames, detections {worker_report['detections']}")


def load_fixture(path: Path) -> Dict[str, Any]:
    """Load a fixture written by ``capture``; tensors and plain values only."""
    import torch

    fixture = torch.load(path, map_location="cpu", weights_only=True)

    return fixture


def comparable(backend: Any, result: Any) -> Dict[str, Any]:
    """Predictions and annotated image of ``result`` as CHW RGB / float tensors.

    GPU modes stay on the GPU; ``v1_numpy`` (``sv.Detections``, HWC BGR numpy)
    is converted to CPU tensors of the same layout.
    """
    import torch

    predictions = backend.predictions(result)
    annotated = backend.annotated(result)
    if isinstance(annotated, torch.Tensor):
        view = {
            "xyxy": predictions.xyxy.float(),
            "class_id": predictions.class_id.long(),
            "confidence": predictions.confidence.float(),
            "annotated": annotated,
        }
        return view

    view = {
        "xyxy": torch.as_tensor(predictions.xyxy, dtype=torch.float32),
        "class_id": torch.as_tensor(predictions.class_id, dtype=torch.long),
        "confidence": torch.as_tensor(predictions.confidence, dtype=torch.float32),
        "annotated": torch.from_numpy(annotated[..., ::-1].copy()).permute(2, 0, 1),
    }

    return view


def stress_pipeline(
    backend: Any, held: List[Any], *, serial: List[dict], depth: int, rounds: int
) -> Dict[str, Any]:
    """Keep ``depth`` futures pending over shuffled frames; compare with serial.

    Every round submits each held frame once, in a seeded shuffled order, so
    neighbours and repeats of one frame meet in flight. Each result must equal
    the serial result of its frame exactly.
    """
    import torch

    generator = torch.Generator().manual_seed(0)
    order = [
        int(index)
        for _ in range(rounds)
        for index in torch.randperm(len(held), generator=generator)
    ]
    pending: deque = deque()
    not_done_after_submit: List[int] = []
    mismatches: List[dict] = []

    def check_oldest() -> None:
        index, future = pending.popleft()
        actual = comparable(backend, future.result())
        for name, expected in serial[index].items():
            if not _same(actual[name], expected):
                mismatches.append({"frame": index, "output": name})

    for position, index in enumerate(order):
        while len(pending) >= depth:
            check_oldest()
        future = backend.submit(held[index], image_id=f"stress-{position}-{index}")
        pending.append((index, future))
        not_done_after_submit.append(sum(not future.done() for _, future in pending))
    while pending:
        check_oldest()

    # Without two unfinished futures at some submit, nothing ran concurrently.
    overlap_observed = max(not_done_after_submit) >= 2
    stress = {
        "submissions": len(order),
        "depth": depth,
        "max_pending_not_done_after_submit": max(not_done_after_submit),
        "mean_pending_not_done_after_submit": sum(not_done_after_submit) / len(order),
        "overlap_observed": overlap_observed,
        "mismatches": mismatches,
        "passed": not mismatches and overlap_observed,
    }

    return stress


def compare_modes(
    output_dir: Path, *, fixture: Path, modes: List[str], options: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare every mode's outputs with the reference mode's, frame by frame."""
    import torch

    sources = load_fixture(fixture)["frames"]
    outputs = {
        mode: torch.load(output_dir / mode / "outputs.pt", weights_only=True)
        for mode in modes
    }
    workers = {
        mode: json.loads((output_dir / mode / "worker.json").read_text())
        for mode in modes
    }
    reference = options["reference"]
    pairs = {}
    for mode in modes:
        if mode == reference:
            continue
        exact = mode in GPU_MODES and reference in GPU_MODES
        pair_name = f"{reference}_vs_{mode}"
        frames = []
        for index, (expected, actual) in enumerate(
            zip(outputs[reference], outputs[mode])
        ):
            image, diff_map = compare_images(
                expected["annotated"],
                actual["annotated"],
                source=sources[index],
                options=options,
            )
            failed = image["status"] not in (
                ("identical",) if exact else ("identical", "renderer_agree")
            )
            if diff_map is not None:
                image["artifacts"] = save_diff_artifacts(
                    output_dir / "diffs" / pair_name / f"frame_{index:02d}",
                    diff_map=diff_map,
                    images=(
                        (expected["annotated"], actual["annotated"]) if failed else None
                    ),
                )
            frames.append(
                {
                    "predictions": compare_predictions(
                        expected, actual, options=options
                    ),
                    "image": image,
                }
            )
        allowed_predictions = (
            {"identical"} if exact else {"identical", "within_tolerance"}
        )
        allowed_images = {"identical"} if exact else {"identical", "renderer_agree"}
        pairs[f"{reference} vs {mode}"] = {
            "required": "identical" if exact else "within_tolerance/renderer_agree",
            "passed": all(
                frame["predictions"]["status"] in allowed_predictions
                and frame["image"]["status"] in allowed_images
                for frame in frames
            ),
            "frames": frames,
        }

    detections = sum(workers[reference]["detections"])
    checks = {
        "pairs": all(pair["passed"] for pair in pairs.values()),
        "sources_unchanged": all(
            not workers[mode]["source_frames_changed"] for mode in modes
        ),
        "stress": all(
            workers[mode]["stress"]["passed"]
            for mode in modes
            if workers[mode]["stress"] is not None
        ),
        "fixture_has_detections": detections > 0,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "fixture": str(fixture),
        "frames": len(sources),
        "reference_detections": detections,
        "options": {key: value for key, value in options.items()},
        "workers": workers,
        "pairs": pairs,
    }

    return report


def compare_predictions(
    expected: Dict[str, Any], actual: Dict[str, Any], *, options: Dict[str, Any]
) -> Dict[str, Any]:
    """Match detections by class and IoU; report the largest differences."""
    import torch
    from torchvision.ops import box_iou

    names = ("xyxy", "class_id", "confidence")
    if all(_same(expected[name], actual[name]) for name in names):
        summary = {"status": "identical", "count": len(expected["class_id"])}
        return summary

    iou = box_iou(expected["xyxy"], actual["xyxy"])
    iou[expected["class_id"][:, None] != actual["class_id"][None, :]] = 0.0
    matched_expected, matched_actual = [], []
    for row in torch.argsort(expected["confidence"], descending=True).tolist():
        if iou.shape[1] == 0:
            break
        best = int(iou[row].argmax())
        if iou[row, best] > MATCH_IOU:
            matched_expected.append(row)
            matched_actual.append(best)
            iou[:, best] = 0.0

    unmatched = [
        confidence
        for detections, matched in (
            (expected, matched_expected),
            (actual, matched_actual),
        )
        for position, confidence in enumerate(detections["confidence"].tolist())
        if position not in matched
    ]
    box_diff = conf_diff = 0.0
    if matched_expected:
        box_diff = float(
            (expected["xyxy"][matched_expected] - actual["xyxy"][matched_actual])
            .abs()
            .max()
        )
        conf_diff = float(
            (
                expected["confidence"][matched_expected]
                - actual["confidence"][matched_actual]
            )
            .abs()
            .max()
        )
    near_threshold = options["confidence"] + options["threshold_margin"]
    within = (
        box_diff <= options["box_tolerance_px"]
        and conf_diff <= options["confidence_tolerance"]
        and all(confidence < near_threshold for confidence in unmatched)
    )
    summary = {
        "status": "within_tolerance" if within else "differ",
        "count_expected": len(expected["class_id"]),
        "count_actual": len(actual["class_id"]),
        "matched": len(matched_expected),
        "unmatched_confidences": unmatched,
        "max_box_diff_px": box_diff,
        "max_confidence_diff": conf_diff,
    }

    return summary


def compare_images(
    expected: Any, actual: Any, *, source: Any, options: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Compare two annotated CHW RGB uint8 images, colour by colour.

    A differing pixel is explained when, for each image, its colour there is
    either the unpainted ``source`` pixel or occurs in the other image within
    ``shift_px``. So a box border or label drawn one pixel off is explained
    (the uncovered pixel shows the source); a different colour, glyph or size
    is not.

    Returns:
        The summary, and an RGB diff map (yellow explained, red unexplained)
        or None when the images are identical or differ in shape.
    """
    import torch

    if expected.shape != actual.shape:
        summary = {
            "status": "differ",
            "shapes": [list(expected.shape), list(actual.shape)],
        }
        return summary, None

    if torch.equal(expected, actual):
        summary = {"status": "identical"}
        return summary, None

    tolerance = options["color_tolerance"]
    differing = (expected.int() - actual.int()).abs().amax(dim=0) > tolerance
    shift = options["shift_px"]
    explained = (
        _unpainted(expected, source=source, tolerance=tolerance)
        | _colour_nearby(expected, actual, shift=shift, tolerance=tolerance)
    ) & (
        _unpainted(actual, source=source, tolerance=tolerance)
        | _colour_nearby(actual, expected, shift=shift, tolerance=tolerance)
    )
    unexplained = differing & ~explained
    unexplained_count = int(unexplained.sum())
    agree = unexplained_count <= options["max_unexplained_pixels"]
    summary = {
        "status": "renderer_agree" if agree else "differ",
        "pixels": int(differing.numel()),
        "differing_pixels": int(differing.sum()),
        "unexplained_pixels": unexplained_count,
        "max_abs_diff": int((expected.int() - actual.int()).abs().max()),
        "max_abs_diff_unexplained": (
            int((expected.int() - actual.int()).abs().amax(dim=0)[unexplained].max())
            if unexplained_count
            else 0
        ),
    }
    diff_map = torch.zeros_like(expected)
    diff_map[:2, differing] = 255
    diff_map[1, unexplained] = 0

    return summary, diff_map


def save_diff_artifacts(
    prefix: Path, *, diff_map: Any, images: Optional[Tuple[Any, Any]]
) -> List[str]:
    """Write ``<prefix>_diff.png`` and, for failures, both annotated images."""
    from torchvision.io import write_png

    prefix.parent.mkdir(parents=True, exist_ok=True)
    named = {"diff": diff_map}
    if images is not None:
        named["expected"], named["actual"] = images
    paths = []
    for name, image in named.items():
        path = prefix.parent / f"{prefix.name}_{name}.png"
        write_png(image.contiguous(), str(path))
        paths.append(str(path))

    return paths


def _unpainted(image: Any, *, source: Any, tolerance: int) -> Any:
    unpainted = (image.int() - source.int()).abs().amax(dim=0) <= tolerance

    return unpainted


def _colour_nearby(image: Any, other: Any, *, shift: int, tolerance: int) -> Any:
    # True where image's colour occurs in other within ``shift`` pixels.
    import torch.nn.functional as functional

    height, width = image.shape[1:]
    padded = functional.pad(other.int(), (shift, shift, shift, shift), value=-1000)
    found = image.new_zeros((height, width), dtype=bool)
    for dy in range(2 * shift + 1):
        for dx in range(2 * shift + 1):
            window = padded[:, dy : dy + height, dx : dx + width]
            found |= (image.int() - window).abs().amax(dim=0) <= tolerance

    return found


def _same(a: Any, b: Any) -> bool:
    same = a.shape == b.shape and bool((a == b.to(a.device)).all())

    return same


def _run_step(*arguments: str) -> None:
    command = [sys.executable, str(SCRIPT), *arguments]
    click.echo(f"$ {' '.join(command[1:3])} ...", err=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise click.ClickException(
            f"{arguments[0]} failed: exit {completed.returncode}"
        )


def _print_report(report: Dict[str, Any]) -> None:
    click.echo(
        f"frames {report['frames']}, reference detections {report['reference_detections']}"
    )
    for name, pair in report["pairs"].items():
        predictions = [frame["predictions"]["status"] for frame in pair["frames"]]
        images = [frame["image"]["status"] for frame in pair["frames"]]
        click.echo(
            f"{name}: {'ok' if pair['passed'] else 'FAIL'} (needs {pair['required']}); "
            f"predictions {_counts(predictions)}; images {_counts(images)}"
        )
    for mode, worker_report in report["workers"].items():
        stress = worker_report["stress"]
        if stress is not None:
            click.echo(
                f"{mode} stress: {'ok' if stress['passed'] else 'FAIL'}, "
                f"{stress['submissions']} submissions, depth {stress['depth']}, "
                f"max not-done after submit {stress['max_pending_not_done_after_submit']}, "
                f"mismatches {len(stress['mismatches'])}"
            )
    click.echo(f"checks {report['checks']} -> {'PASS' if report['passed'] else 'FAIL'}")


def _counts(statuses: List[str]) -> Dict[str, int]:
    counts = {status: statuses.count(status) for status in sorted(set(statuses))}

    return counts


if __name__ == "__main__":
    main()
