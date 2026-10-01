"""Live object detection with the V2 engine: camera, video file or one image.

    camera / video / image -> detector -> boxes -> labels -> window (or headless)

The model is loaded once through ``AutoModel.from_pretrained`` (ONNX Runtime,
CPUExecutionProvider, CPU tensors) and passed to the session as the
``detection_model`` resource. ``--mode serial`` compiles block ``run`` and calls
``session.run``; ``--mode pipeline`` compiles block phases and submits to
``session.pipeline``. Both run on a background thread; the window stays on the
main thread.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import cv2
import torch
from detection_blocks import create_catalogue
from live import LiveSummary, StillImage, image_to_bgr, run_live
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    CompileOptions,
)

DEMO_DIR = Path(__file__).resolve().parent
WORKFLOW_PATH = DEMO_DIR / "workflows" / "live_detection.json"
DEVICE = "cpu"
ONNX_PROVIDERS = ["CPUExecutionProvider"]
WARMUP_RUNS = 3
WARMUP_SIZE_HW = (480, 640)
PRINTED_DETECTIONS = 20


@click.command()
@click.option(
    "--camera",
    type=click.IntRange(
        min=0,
    ),
    default=0,
    show_default=True,
    help="Camera index; used when neither --video nor --image is given.",
)
@click.option(
    "--video",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
    help="Video file instead of the camera; every frame is processed.",
)
@click.option(
    "--image",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
    help="One image instead of the camera; always headless.",
)
@click.option(
    "--mode",
    type=click.Choice(
        ["serial", "pipeline"],
    ),
    default="pipeline",
    show_default=True,
    help="session.run one frame at a time, or session.pipeline.",
)
@click.option(
    "--max-in-flight",
    type=click.IntRange(
        min=1,
    ),
    default=2,
    show_default=True,
    help="Frames in the pipeline at once (pipeline mode); more can add age.",
)
@click.option(
    "--model",
    "model_id",
    type=str,
    default="yolov8n-640",
    show_default=True,
    help="Roboflow model ID loaded with AutoModel.",
)
@click.option(
    "--backend",
    type=click.Choice(
        ["onnx", "auto"],
    ),
    default="onnx",
    show_default=True,
    help="Requested model backend; auto lets AutoModel choose.",
)
@click.option(
    "--confidence",
    type=click.FloatRange(
        min=0.0,
        max=1.0,
    ),
    default=0.4,
    show_default=True,
    help="Minimum detection confidence.",
)
@click.option(
    "--headless",
    is_flag=True,
    default=False,
    help="No window; print statistics at the end.",
)
@click.option(
    "--frames",
    "max_results",
    type=click.IntRange(
        min=1,
    ),
    default=None,
    help="Stop after this many results.",
)
@click.option(
    "--duration",
    "duration_seconds",
    type=click.FloatRange(
        min=0.0,
        min_open=True,
    ),
    default=None,
    help="Stop after this many seconds.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Write stats.json here, and annotated.png for --image.",
)
def main(
    camera: int,
    video: Optional[Path],
    image: Optional[Path],
    mode: str,
    max_in_flight: int,
    model_id: str,
    backend: str,
    confidence: float,
    headless: bool,
    max_results: Optional[int],
    duration_seconds: Optional[float],
    output_dir: Optional[Path],
) -> None:
    """Detect objects live and show the speed of every stage.

    \f
    Args:
        camera: Camera index.
        video: Video file to process instead of the camera.
        image: Image to process instead of the camera.
        mode: ``serial`` or ``pipeline``.
        max_in_flight: Pipeline bound.
        model_id: Model to load.
        backend: Requested backend, ``onnx`` or ``auto``.
        confidence: Detection confidence threshold.
        headless: Run without a window.
        max_results: Optional result limit.
        duration_seconds: Optional time limit.
        output_dir: Optional directory for ``stats.json`` and ``annotated.png``.
    """
    if video is not None and image is not None:
        raise click.UsageError("Use either --video or --image, not both.")

    model = load_model(model_id, backend=backend)
    actual = describe_backend(model)
    click.echo(
        f"model {model_id}: {actual['model_class']} | providers "
        f"{actual['providers']} | device {actual['device']} "
        f"(requested backend {backend}, device {DEVICE})"
    )
    click.echo(
        f"threads: ORT intra-op {actual['ort_intra_op_num_threads']}, inter-op "
        f"{actual['ort_inter_op_num_threads']} (configured; 0 = ORT chooses) | "
        f"torch {torch.get_num_threads()}"
    )

    plan = compile_live_workflow(mode=mode)
    session = plan.create_session({"detection_model": model})
    warmup_seconds = warm_up(session, confidence=confidence)
    click.echo(
        f"warmup: {WARMUP_RUNS} runs on a blank {WARMUP_SIZE_HW[1]}x"
        f"{WARMUP_SIZE_HW[0]} image in {warmup_seconds:.2f} s, excluded from "
        "statistics (no detections: first real frames still pay cold drawing "
        "and resize costs)"
    )

    # The camera opens only after the model is ready, so a permission prompt
    # is never stuck behind model loading.
    capture, lossless = open_source(camera=camera, video=video, image=image)
    pipelined = mode == "pipeline"
    header = (
        f"{mode}{f' x{max_in_flight}' if pipelined else ''} | "
        f"{actual['model_class']} | {actual['providers']} | "
        f"model {actual['device']} | images cpu"
    )
    display = not headless and image is None
    click.echo(header)
    click.echo("Press q or Esc in the window to stop." if display else "Headless.")

    summary = run_live(
        session,
        capture=capture,
        lossless=lossless,
        confidence=confidence,
        max_in_flight=max_in_flight if pipelined else None,
        header=header,
        display=display,
        max_results=max_results,
        duration_seconds=duration_seconds,
    )

    report = summarize(summary, display=display, header=header, backend=actual)
    click.echo(json.dumps(report, indent=2))
    if image is not None and summary.last_result is not None:
        _print_detections(summary, class_names=model.class_names)
    if output_dir is not None:
        _write_outputs(
            summary,
            report=report,
            output_dir=output_dir,
            save_annotated=image is not None,
        )
    if summary.error is not None:
        click.echo(f"Error: {type(summary.error).__name__}: {summary.error}", err=True)
        sys.exit(1)


def load_model(model_id: str, *, backend: str) -> Any:
    """Load the detector once on the CPU with ONNX Runtime's CPU provider.

    Trusted platform packages only: untrusted packages are never allowed.

    Args:
        model_id: Roboflow model ID.
        backend: ``onnx``, or ``auto`` to let AutoModel choose.

    Returns:
        The loaded ``inference_models`` object detection model.

    Raises:
        click.ClickException: When a dependency is missing or loading fails.
    """
    requested = f"backend {backend}, device {DEVICE}, providers {ONNX_PROVIDERS}"
    try:
        from inference_models import AutoModel
        from inference_models.errors import BaseInferenceModelsError
    except ImportError as error:
        raise click.ClickException(
            f"inference_models is not importable ({error}). Run from the "
            "repository root with PYTHONPATH=.:workflows:inference_models:stream_vision"
        ) from error

    try:
        model = AutoModel.from_pretrained(
            model_id,
            backend=None if backend == "auto" else backend,
            device=DEVICE,
            onnx_execution_providers=ONNX_PROVIDERS,
            allow_untrusted_packages=False,
        )
    except (BaseInferenceModelsError, ImportError) as error:
        raise click.ClickException(
            f"Could not load {model_id} ({requested}): "
            f"{type(error).__name__}: {error}"
        ) from error

    return model


def describe_backend(model: Any) -> Dict[str, Any]:
    """Read the backend the loaded model actually uses, not the request.

    ``inference_models`` keeps these as private attributes; this is the only
    place that reads them.

    Args:
        model: Loaded model.

    Returns:
        ``model_class``, ``providers`` (ONNX Runtime session providers, or
        ``not ONNX Runtime``), ``device``, and the session's configured
        ``ort_intra_op_num_threads`` / ``ort_inter_op_num_threads`` (0 lets
        ONNX Runtime choose; None without an ONNX Runtime session).
    """
    session = getattr(model, "_session", None)
    providers = "not ONNX Runtime"
    intra_op = inter_op = None
    if hasattr(session, "get_providers"):
        providers = ",".join(session.get_providers())
        options = session.get_session_options()
        intra_op = options.intra_op_num_threads
        inter_op = options.inter_op_num_threads
    description = {
        "model_class": type(model).__name__,
        "providers": providers,
        "device": str(getattr(model, "_device", "unknown")),
        "ort_intra_op_num_threads": intra_op,
        "ort_inter_op_num_threads": inter_op,
    }

    return description


def compile_live_workflow(*, mode: str) -> CompiledWorkflow:
    """Compile ``workflows/live_detection.json`` for one mode.

    Args:
        mode: ``serial`` compiles block ``run``; ``pipeline`` compiles phases.

    Returns:
        The compiled plan; mutation conflicts are errors, not warnings.
    """
    definition = json.loads(WORKFLOW_PATH.read_text())
    plan = compile_workflow(
        definition,
        catalogue=create_catalogue(),
        options=CompileOptions(
            block_execution="phases" if mode == "pipeline" else "run",
            mutation_conflicts="error",
        ),
    )

    return plan


def warm_up(session: Any, *, confidence: float) -> float:
    """Run the workflow ``WARMUP_RUNS`` times on a blank image.

    Args:
        session: Session of the compiled workflow; no pipeline open yet.
        confidence: Workflow input ``confidence``.

    Returns:
        Seconds the warmup took.
    """
    started = time.perf_counter()
    for run in range(WARMUP_RUNS):
        blank = ImageData.from_tensor(
            torch.zeros((3, *WARMUP_SIZE_HW), dtype=torch.uint8),
            image_id=f"warmup-{run}",
        )
        session.run({"image": blank, "confidence": confidence})
    seconds = time.perf_counter() - started

    return seconds


def open_source(
    *, camera: int, video: Optional[Path], image: Optional[Path]
) -> Tuple[Any, bool]:
    """Open the image, video or camera.

    Args:
        camera: Camera index, used when no file is given.
        video: Video file.
        image: Image file.

    Returns:
        The capture and whether every frame must be processed (files).

    Raises:
        click.ClickException: When the source cannot be opened.
    """
    if image is not None:
        pixels_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if pixels_bgr is None:
            raise click.ClickException(f"Could not read image {image}")
        return StillImage(pixels_bgr), True

    if video is not None:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise click.ClickException(f"Could not open video {video}")
        return capture, True

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise click.ClickException(
            f"Could not open camera {camera}. On macOS, allow camera access for "
            "this terminal in System Settings > Privacy & Security > Camera, "
            "then run again. Use --camera to pick another device."
        )

    return capture, False


def summarize(
    summary: LiveSummary,
    *,
    display: bool,
    header: str,
    backend: Dict[str, Any],
) -> Dict[str, Any]:
    """Final statistics as JSON-friendly data with explicit units.

    Args:
        summary: Result of ``run_live``.
        display: Whether results were shown in a window.
        header: Mode, backend and device line.
        backend: Result of ``describe_backend``.

    Returns:
        The report printed at the end and written to ``stats.json``.
    """
    snapshot = summary.snapshot
    report = {
        "stopped": summary.reason,
        "setup": header,
        "window": display,
        "frame_size_hw": snapshot["size_hw"],
        "frames_read": snapshot["frames_read"],
        "results_completed": snapshot["results_completed"],
        "results_presented": snapshot["results_presented"] if display else None,
        "last_window_fps": {
            "output": snapshot["output_fps"] if display else None,
            "processed": snapshot["processed_fps"],
            "source": snapshot["source_fps"],
        },
        "capture_to_present_ms": snapshot["capture_to_present_ms"] if display else None,
        "capture_to_result_ms": snapshot["capture_to_result_ms"],
        "stage_median_ms": snapshot["stage_median_ms"],
        "drops": snapshot["drops"],
        "warmup_blank_runs_excluded": WARMUP_RUNS,
        "ort_intra_op_num_threads": backend["ort_intra_op_num_threads"],
        "ort_inter_op_num_threads": backend["ort_inter_op_num_threads"],
        "torch_threads": torch.get_num_threads(),
    }
    if summary.reader_still_blocked:
        report["note"] = "the capture read had not returned when the run ended"

    return report


def _print_detections(summary: LiveSummary, *, class_names: Any) -> None:
    predictions = summary.last_result.predictions
    click.echo(f"{len(predictions)} detections")
    rows = zip(
        predictions.class_id.tolist(),
        predictions.confidence.tolist(),
        predictions.xyxy.tolist(),
    )
    for class_id, confidence, xyxy in list(rows)[:PRINTED_DETECTIONS]:
        click.echo(f"  {class_names[class_id]:<16} {confidence:.2f} xyxy {xyxy}")


def _write_outputs(
    summary: LiveSummary,
    *,
    report: Dict[str, Any],
    output_dir: Path,
    save_annotated: bool,
) -> None:
    # Camera and video frames are never written; only the --image result.
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(report, indent=2) + "\n")
    click.echo(f"Statistics: {stats_path}")
    if not save_annotated or summary.last_result is None:
        return

    annotated_path = output_dir / "annotated.png"
    cv2.imwrite(str(annotated_path), image_to_bgr(summary.last_result.annotated))
    click.echo(f"Annotated image: {annotated_path}")


if __name__ == "__main__":
    main()
