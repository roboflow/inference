"""Compare the V1 ExecutionEngine with V2 (serial and pipeline) on one detector.

    one CHW RGB uint8 image (default dogs.jpg resized to 1920x1080)
      ├─ V1: ExecutionEngine + stock tensor blocks
      │      roboflow_object_detection_model@v1 -> bounding_box@v1 -> label@v1
      ├─ V2 serial:   live_detection.json, session.run
      └─ V2 pipeline: live_detection.json, session.pipeline(max_in_flight=2)

All modes use the same already loaded ``inference_models`` YOLOv8n ONNX model
on the CPU. V1 reaches it through ``ForwardingModelsProvider``, a thin
``ModelsProvider`` that calls the real model; there is no mock inference.

Before timing, every mode runs once on the image and once on a blank image;
predictions and annotated pixels must match V1 exactly, and the source tensor
must be unchanged. The script stops on any disagreement.

Timed per frame: from creating the input wrapper (WorkflowImageData / ImageData)
until the engine returns the result (serial) or the future is done (pipeline).
Model loading, compilation, warmup and checks are excluded.
"""

import json
import platform
import statistics
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, NamedTuple, Optional, Tuple

import click
import cv2
import numpy as np
import torch
from roboflow_workflows.configuration import (
    TensorConfiguration,
    WorkflowsConfiguration,
    configure_process,
)

# Process-wide and set once: must happen before any other roboflow_workflows
# import, because block discovery and WorkflowImageData read it at import time.
configure_process(
    WorkflowsConfiguration(
        tensor=TensorConfiguration(
            representation_enabled=True,
            image_tensor_device=torch.device("cpu"),
        )
    )
)

from roboflow_workflows.core_steps.common.entities import (  # noqa: E402
    StepExecutionMode,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine  # noqa: E402
from roboflow_workflows.execution_engine.entities.base import (  # noqa: E402
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import (  # noqa: E402
    ImageData,
)
from roboflow_workflows.execution_engine.v2.pipelining import (  # noqa: E402
    PipelineOptions,
)
from run_demo import (  # noqa: E402
    compile_live_workflow,
    describe_backend,
    load_model,
)

from inference_models import configuration as models_configuration  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[2]
DEFAULT_IMAGE = REPO_ROOT / "workflows" / "tests" / "assets" / "dogs.jpg"
MODEL_ID = "yolov8n-640"
MAX_IN_FLIGHT = 2
WARMUP_RUNS = 5
# Frames per warmup run; 5 runs x 2 frames = 10 warmup frames per mode.
WARMUP_FRAMES = 2
MODES = ("v1", "v2_serial", "v2_pipeline")

# Explicit model post-process values for V1, equal to what the V2 detector gets
# by calling post_process with confidence only (the model's own defaults).
IOU_THRESHOLD = (
    models_configuration.INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD
)
MAX_DETECTIONS = (
    models_configuration.INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS
)
CLASS_AGNOSTIC_NMS = (
    models_configuration.INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS
)
# Kwargs the model server consumes itself (active learning), never the model.
SERVER_ONLY_KWARGS = ("disable_active_learning", "active_learning_target_dataset")

V1_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "detector",
            "images": "$inputs.image",
            "model_id": MODEL_ID,
            "confidence": "$inputs.confidence",
            "iou_threshold": IOU_THRESHOLD,
            "max_detections": MAX_DETECTIONS,
            "class_agnostic_nms": CLASS_AGNOSTIC_NMS,
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "boxes",
            "image": "$inputs.image",
            "predictions": "$steps.detector.predictions",
        },
        {
            "type": "roboflow_core/label_visualization@v1",
            "name": "labels",
            "image": "$steps.boxes.image",
            "predictions": "$steps.detector.predictions",
            "text": "Class and Confidence",
            "copy_image": False,
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "annotated", "selector": "$steps.labels.image"},
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.detector.predictions",
        },
    ],
}


class ForwardingModelsProvider:
    """The three ``ModelsProvider`` calls the V1 tensor detector makes.

    ``run_tensor_native_inference`` calls the real model like the server's
    adapter does (``model(images, **kwargs)``), minus the server-only
    active-learning kwargs. No model manager, cache or usage tracking.
    """

    def __init__(self, model: Any):
        self.model = model

    def add_model(self, model_id: str, api_key: Optional[str] = None, **_: Any) -> None:
        if model_id != MODEL_ID:
            raise ValueError(f"Only {MODEL_ID} is loaded, not {model_id}")

    def run_tensor_native_inference(self, model_id: str, **kwargs: Any) -> Any:
        model_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name not in SERVER_ONLY_KWARGS
        }
        images = model_kwargs.pop("images")
        detections = self.model(images, **model_kwargs)

        return detections

    def get_class_names(self, model_id: str) -> List[str]:
        return list(self.model.class_names)


def load_source(image_path: Path, *, width: int, height: int) -> torch.Tensor:
    pixels_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if pixels_bgr is None:
        raise click.ClickException(f"Could not read image {image_path}")

    resized = cv2.resize(pixels_bgr, (width, height), interpolation=cv2.INTER_AREA)
    chw_rgb = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)[::-1]))

    return chw_rgb


class Runners(NamedTuple):
    """What each mode runs.

    ``v1`` calls ``v1_request``; ``v2_serial`` calls ``v2_serial_request``;
    ``v2_pipeline`` submits ``v2_inputs`` to a pipeline of
    ``v2_pipeline_session``. Requests and ``v2_inputs`` take the source tensor
    and a request id, and wrap the tensor in a new wrapper with that id.
    """

    v1_request: Callable[[torch.Tensor, str], Any]
    v2_serial_request: Callable[[torch.Tensor, str], Any]
    v2_inputs: Callable[[torch.Tensor, str], Dict[str, Any]]
    v2_pipeline_session: Any


def build_runners(model: Any, *, confidence: float) -> Runners:
    """Initialize V1 and compile V2 (serial and phased) on the loaded model."""
    v1_engine = ExecutionEngine.init(
        workflow_definition=V1_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": ForwardingModelsProvider(model),
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
    )
    v2_serial = compile_live_workflow(mode="serial").create_session(
        {"detection_model": model}
    )
    v2_phased = compile_live_workflow(mode="pipeline").create_session(
        {"detection_model": model}
    )

    def v1_request(source: torch.Tensor, request_id: str) -> Dict[str, Any]:
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id=request_id),
            tensor_image=source,
        )
        (result,) = v1_engine.run(
            runtime_parameters={"image": image, "confidence": confidence}
        )

        return result

    def v2_inputs(source: torch.Tensor, request_id: str) -> Dict[str, Any]:
        image = ImageData.from_tensor(source, image_id=request_id)
        inputs = {"image": image, "confidence": confidence}

        return inputs

    def v2_serial_request(source: torch.Tensor, request_id: str) -> Any:
        result = v2_serial.run(v2_inputs(source, request_id))

        return result

    runners = Runners(
        v1_request=v1_request,
        v2_serial_request=v2_serial_request,
        v2_inputs=v2_inputs,
        v2_pipeline_session=v2_phased,
    )

    return runners


def output_value(result: Any, name: str) -> Any:
    if isinstance(result, dict):
        return result[name]

    (entry,) = result.selections[name].values()
    value = result.outputs.data[entry]

    return value


def comparable(result: Any) -> Dict[str, Any]:
    predictions = output_value(result, "predictions")
    annotated = output_value(result, "annotated").tensor_image
    view = {
        "xyxy": predictions.xyxy.cpu(),
        "class_id": predictions.class_id.cpu(),
        "confidence": predictions.confidence.cpu(),
        "annotated": annotated.cpu(),
    }

    return view


def run_once_pipelined(runners: Runners, source: torch.Tensor, request_id: str):
    with runners.v2_pipeline_session.pipeline(
        options=PipelineOptions(max_in_flight=MAX_IN_FLIGHT)
    ) as pipeline:
        result = pipeline.submit(runners.v2_inputs(source, request_id)).result()

    return result


def verify(runners: Runners, source: torch.Tensor, *, label: str) -> int:
    """Run every mode once on ``source``; raise unless all agree with V1."""
    original = source.clone()
    results = {
        "v1": comparable(runners.v1_request(source, f"check-{label}-v1")),
        "v2_serial": comparable(runners.v2_serial_request(source, f"check-{label}-s")),
        "v2_pipeline": comparable(
            run_once_pipelined(runners, source, f"check-{label}-p")
        ),
    }
    reference = results["v1"]
    for mode in ("v2_serial", "v2_pipeline"):
        for field, expected in reference.items():
            actual = results[mode][field]
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                differing = (
                    int((actual != expected).any(dim=0).sum())
                    if field == "annotated" and actual.shape == expected.shape
                    else None
                )
                raise click.ClickException(
                    f"{label}: {mode} {field} differs from V1 "
                    f"(shapes {tuple(actual.shape)} vs {tuple(expected.shape)}, "
                    f"differing pixels {differing})"
                )
    if not torch.equal(source, original):
        raise click.ClickException(f"{label}: a mode modified the source image")

    detections = len(reference["class_id"])

    return detections


def measure_serial(request: Callable, source: torch.Tensor, *, frames: int, tag: str):
    latencies_ms = []
    started = time.perf_counter()
    for index in range(frames):
        request_started = time.perf_counter()
        request(source, f"{tag}-{index}")
        latencies_ms.append((time.perf_counter() - request_started) * 1000.0)
    elapsed = time.perf_counter() - started

    return elapsed, latencies_ms


def measure_pipeline(runners: Runners, source: torch.Tensor, *, frames: int, tag: str):
    # At most MAX_IN_FLIGHT pending; the oldest result is taken and dropped
    # before the next submit. Latency: input wrapper creation to future done,
    # stamped by a done callback in the completing thread.
    latencies_ms: List[float] = []
    pending: Deque[Future] = deque()

    def stamp_latency(submitted_at: float) -> Callable[[Future], None]:
        def stamp(_: Future) -> None:
            latencies_ms.append((time.perf_counter() - submitted_at) * 1000.0)

        return stamp

    options = PipelineOptions(max_in_flight=MAX_IN_FLIGHT)
    started = time.perf_counter()
    with runners.v2_pipeline_session.pipeline(options=options) as pipeline:
        for index in range(frames):
            if len(pending) == MAX_IN_FLIGHT:
                pending.popleft().result()
            submitted_at = time.perf_counter()
            future = pipeline.submit(runners.v2_inputs(source, f"{tag}-{index}"))
            future.add_done_callback(stamp_latency(submitted_at))
            pending.append(future)
        while pending:
            pending.popleft().result()
    elapsed = time.perf_counter() - started
    # result() can return just before the done callback has stamped.
    while len(latencies_ms) < frames:
        time.sleep(0.001)

    return elapsed, latencies_ms


def measure_mode(
    runners: Runners, source: torch.Tensor, *, mode: str, frames: int, tag: str
) -> Tuple[float, List[float]]:
    if mode == "v1":
        measured = measure_serial(runners.v1_request, source, frames=frames, tag=tag)
    elif mode == "v2_serial":
        measured = measure_serial(
            runners.v2_serial_request, source, frames=frames, tag=tag
        )
    else:
        measured = measure_pipeline(runners, source, frames=frames, tag=tag)

    return measured


def summarize(elapsed: float, latencies_ms: List[float]) -> Dict[str, float]:
    ordered = sorted(latencies_ms)
    summary = {
        "frames": len(ordered),
        "seconds": round(elapsed, 4),
        "fps": round(len(ordered) / elapsed, 2),
        "latency_median_ms": round(statistics.median(ordered), 2),
        "latency_p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
    }

    return summary


@click.command()
@click.option(
    "--image",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=DEFAULT_IMAGE,
    show_default=True,
    help="Still image to process; resized to --width x --height.",
)
@click.option(
    "--width",
    type=click.IntRange(
        min=32,
    ),
    default=1920,
    show_default=True,
)
@click.option(
    "--height",
    type=click.IntRange(
        min=32,
    ),
    default=1080,
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
    "--frames",
    type=click.IntRange(
        min=2,
    ),
    default=100,
    show_default=True,
    help="Measured frames per mode and repeat.",
)
@click.option(
    "--repeats",
    type=click.IntRange(
        min=1,
    ),
    default=3,
    show_default=True,
    help="Repeats; the mode order rotates each repeat.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write the results JSON here.",
)
def main(
    image: Path,
    width: int,
    height: int,
    confidence: float,
    frames: int,
    repeats: int,
    output: Optional[Path],
) -> None:
    """Measure V1 vs V2 serial vs V2 pipeline x2 on one image (CPU ONNX)."""
    model = load_model(MODEL_ID, backend="onnx")
    backend = describe_backend(model)
    source = load_source(image, width=width, height=height)
    runners = build_runners(model, confidence=confidence)

    detections = verify(runners, source, label="image")
    if detections == 0:
        raise click.ClickException("No detections on the image; parity check is weak")
    blank_detections = verify(runners, torch.zeros_like(source), label="blank")
    click.echo(f"parity ok: {detections} detections; blank {blank_detections}")

    for mode in MODES:
        for run in range(WARMUP_RUNS):
            measure_mode(
                runners, source, mode=mode, frames=WARMUP_FRAMES, tag=f"warm{run}"
            )

    runs = []
    for repeat in range(repeats):
        order = MODES[repeat % len(MODES) :] + MODES[: repeat % len(MODES)]
        for mode in order:
            elapsed, latencies = measure_mode(
                runners, source, mode=mode, frames=frames, tag=f"r{repeat}"
            )
            runs.append(
                {"repeat": repeat, "mode": mode, **summarize(elapsed, latencies)}
            )
            click.echo(json.dumps(runs[-1]))

    report = {
        "runs": runs,
        "fps_median_by_mode": {
            mode: statistics.median(run["fps"] for run in runs if run["mode"] == mode)
            for mode in MODES
        },
        "setup": setup_facts(
            image,
            source=source,
            backend=backend,
            confidence=confidence,
            detections=detections,
            frames=frames,
            repeats=repeats,
        ),
    }
    click.echo(json.dumps(report["fps_median_by_mode"], indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        click.echo(f"Results: {output}")


def setup_facts(image, *, source, backend, confidence, detections, frames, repeats):
    facts = {
        "image": str(image),
        "image_chw": list(source.shape),
        "confidence": confidence,
        "detections_on_image": detections,
        "frames_per_run": frames,
        "repeats": repeats,
        "warmup_runs_per_mode": WARMUP_RUNS,
        "warmup_frames_per_run": WARMUP_FRAMES,
        "model": {"id": MODEL_ID, **backend},
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "machine": platform.platform(),
        "post_process": {
            "iou_threshold": IOU_THRESHOLD,
            "max_detections": MAX_DETECTIONS,
            "class_agnostic_nms": CLASS_AGNOSTIC_NMS,
        },
        "v1_blocks": [
            "roboflow_workflows.core_steps.models.roboflow.object_detection.v1_tensor",
            "roboflow_workflows.core_steps.visualizations.bounding_box.v1_tensor",
            "roboflow_workflows.core_steps.visualizations.label.v1_tensor",
        ],
        "v2_blocks": "development/workflows-2.0/08-live-detection/detection_blocks.py",
        "v2_pipeline_max_in_flight": MAX_IN_FLIGHT,
        "timing_boundary": (
            "per frame: input wrapper creation (WorkflowImageData / ImageData) to "
            "engine.run / session.run return, or pipeline future done (measured "
            "by a done callback); FPS = frames / wall time of the run loop. "
            "Excludes model loading, compilation, warmup, parity checks."
        ),
        "v1_model_access": (
            "ForwardingModelsProvider: model(images, **kwargs) on the same loaded "
            "model, like the server adapter; drops only "
            f"{list(SERVER_ONLY_KWARGS)}; no model manager, cache or usage tracking"
        ),
        "known_differences": [
            "V1 mints a uuid inference_id and per-box detection_id; V2 does not",
            "V1 block runs batch-oriented with a one-image batch; V2 runs one image",
            "V2 blocks also time themselves (perf_counter per stage)",
            "V1 boxes clone the image (copy_image default True); labels draw in "
            "place (copy_image=False); V2 boxes clone once, labels in place",
        ],
    }

    return facts


if __name__ == "__main__":
    main()
