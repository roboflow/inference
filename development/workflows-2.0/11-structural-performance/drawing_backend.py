"""The 10 batched backend with a choice of how detections are drawn.

    drawing          after the batched detector                      engine calls  transfers
                                                                     per run of N  per run
    per-painter      10 as is: boxes(predictions), labels(predictions)   1 + 2N    6N
    shared-prep      prep -> boxes(drawing), labels(drawing): per image  2 + 2N    1
    batch-painters   prep -> batch boxes, batch labels: one call each    4         1

    (N = 8: 17, 18 and 4 calls; per-painter transfers count images with
    detections, 2 per empty image.)

Both shared drawings use the same prep step, one call and one transfer per
run. They differ only in painter delivery; every image keeps its own paint
primitives and readiness wait (``drawing_blocks``).

``per-painter`` is 10 ``batched_backend.build_for_model`` itself, unchanged.
The other two compile ``v2_workflow(stages, drawing=...)`` with
``drawing_blocks.create_catalogue()`` and reuse the 10 ``V2BatchedBackend``
calls, rows and pipeline. ``v1_tensor`` exists only as ``per-painter``.

Rows carry the 10 keys plus ``prep_ms`` (``DRAWING_TIMINGS``): the whole
batch's prep time, repeated in every row. Count it once per batch.

    import drawing_backend, backends
    backends.configure_mode("v2_pipeline", device="cuda:0")
    backend = drawing_backend.build_for_model(
        "v2_pipeline", model=model, model_id="yolov8n-640", confidence=0.4,
        pipeline_depth=2, drawing="shared-prep",
    )
    rows = backend.process_batch(frames, image_ids=ids)       # or submit_batch
"""

from typing import Any, Dict, List

import structural_imports  # noqa: F401 - installs the 08, 09 and 10 search paths

# isort: split

import batched_backend

DRAWINGS = ("per-painter", "shared-prep", "batch-painters")
STAGES = batched_backend.STAGES
MODES = batched_backend.MODES
# Row timing added by the shared drawings, ms; one value per batch.
DRAWING_TIMINGS = ("prep_ms",)
# Device-to-host transfers of prediction data per run, stages "full", from the code.
HOST_TRANSFERS = {
    "per-painter": "6 per image with detections, 2 per empty image",
    "shared-prep": "1 per run",
    "batch-painters": "1 per run",
}

_DRAWING_STEPS = {
    "shared-prep": (
        "structural/batched_detection_drawing_prep@v1",
        "structural/box_visualization@v1",
        "structural/label_visualization@v1",
    ),
    "batch-painters": (
        "structural/batched_detection_drawing_prep@v1",
        "structural/batched_box_visualization@v1",
        "structural/batched_label_visualization@v1",
    ),
}


def build_for_model(
    mode: str,
    *,
    model: Any,
    model_id: str,
    confidence: float,
    pipeline_depth: int = 2,
    stages: str = "full",
    drawing: str = "per-painter",
) -> batched_backend.BatchedBackend:
    """Build the 10 batched backend of ``mode`` with the ``drawing`` variant.

    Same arguments and result as 10 ``batched_backend.build_for_model``, so a
    10 runner can call this in its place.

    Args:
        mode: One of ``MODES``; 09 ``backends.configure_mode(mode)`` must
            have run.
        model: Loaded ``inference_models`` object detection model with
            ``_device``.
        model_id: Key under which the V1 models provider serves ``model``.
        confidence: Detection confidence threshold.
        pipeline_depth: ``v2_pipeline`` only: batches executing at once.
        stages: One of ``STAGES``; the shared drawings need a painter stage.
        drawing: One of ``DRAWINGS``.

    Returns:
        The backend, not warmed up.

    Raises:
        ValueError: On an unknown drawing, a shared drawing with ``v1_tensor``
            or with ``stages="detector"``, or what 10 rejects.
    """
    if drawing not in DRAWINGS:
        raise ValueError(f"drawing must be one of {DRAWINGS}, got {drawing!r}")
    if drawing == "per-painter":
        backend = batched_backend.build_for_model(
            mode,
            model=model,
            model_id=model_id,
            confidence=confidence,
            pipeline_depth=pipeline_depth,
            stages=stages,
        )
        backend.facts["drawing"] = drawing
        return backend

    if mode not in ("v2_serial", "v2_pipeline"):
        raise ValueError(f"drawing {drawing!r} needs a V2 mode, got {mode!r}")

    backend = DrawingBackend(
        mode,
        model=model,
        confidence=confidence,
        pipeline_depth=pipeline_depth,
        stages=stages,
        drawing=drawing,
    )

    return backend


def v2_workflow(stages: str, *, drawing: str) -> Dict[str, Any]:
    """The 10 batched V2 workflow, cut after ``stages``, drawn by ``drawing``.

    Step names other than ``drawing`` and every output name are the 10 ones,
    so rows of all variants compare key by key.

    Args:
        stages: ``boxes`` or ``full``; ``detector`` only with ``per-painter``.
        drawing: One of ``DRAWINGS``.

    Returns:
        A V2 definition compiling with ``drawing_blocks.create_catalogue()``.
        Shared drawings add the step ``drawing`` and the output ``prep_ms``.

    Raises:
        ValueError: On an unknown stage or drawing, or ``detector`` with a
            shared drawing.
    """
    if drawing not in DRAWINGS:
        raise ValueError(f"drawing must be one of {DRAWINGS}, got {drawing!r}")

    definition = batched_backend.v2_workflow(stages)
    if drawing == "per-painter":
        return definition
    if stages == "detector":
        raise ValueError(f"drawing {drawing!r} needs stages 'boxes' or 'full'")

    prep_type, boxes_type, labels_type = _DRAWING_STEPS[drawing]
    painter_types = {"boxes": boxes_type, "labels": labels_type}
    steps: List[Dict[str, Any]] = []
    for step in definition["steps"]:
        if step["name"] not in painter_types:
            steps.append(step)
            continue

        painter = {key: value for key, value in step.items() if key != "predictions"}
        painter.update(
            type=painter_types[step["name"]], drawing="$steps.drawing.drawing"
        )
        steps.append(painter)
    steps.insert(
        1,
        {
            "type": prep_type,
            "name": "drawing",
            "predictions": "$steps.detector.predictions",
        },
    )
    outputs = [
        *definition["outputs"],
        {"type": "JsonField", "name": "prep_ms", "selector": "$steps.drawing.prep_ms"},
    ]
    shared = {**definition, "steps": steps, "outputs": outputs}

    return shared


class DrawingBackend(batched_backend.V2BatchedBackend):
    """10 ``V2BatchedBackend`` running a shared-drawing workflow.

    Copies 10's constructor because it hard-codes the workflow and catalogue.
    Changes: ``v2_workflow(..., drawing=)``, ``drawing_blocks.create_catalogue()``,
    an exposed ``plan``, and the drawing, batching, readiness, host-transfer and
    paint-event facts. Calls, rows, pipelining and ``close`` are inherited.
    """

    def __init__(
        self,
        mode: str,
        *,
        model: Any,
        confidence: float,
        pipeline_depth: int,
        stages: str,
        drawing: str,
    ) -> None:
        import backends
        import drawing_blocks
        from roboflow_workflows.execution_engine.v2.compilation import (
            compile_workflow,
        )
        from roboflow_workflows.execution_engine.v2.pipelining import (
            PipelineOptions,
        )
        from roboflow_workflows.execution_engine.v2.plan import CompileOptions

        self.mode = mode
        self.device = model._device
        self._confidence = confidence
        pipelined = mode == "v2_pipeline"
        definition = v2_workflow(stages, drawing=drawing)
        self.plan = compile_workflow(
            definition,
            catalogue=drawing_blocks.create_catalogue(),
            options=CompileOptions(
                block_execution="phases" if pipelined else "run",
                mutation_conflicts="error",
            ),
        )
        self._session = self.plan.create_session({"detection_model": model})
        self._pipeline = None
        if pipelined:
            self._pipeline = self._session.pipeline(
                options=PipelineOptions(max_in_flight=pipeline_depth)
            )
            self._pipeline.__enter__()
        step_types = {step["type"] for step in definition["steps"]}
        self.facts = {
            "mode": mode,
            "engine": "V2 " + ("session.pipeline" if pipelined else "session.run"),
            "block_execution": "phases" if pipelined else "run",
            "stages": stages,
            "drawing": drawing,
            "pipeline_depth": pipeline_depth if pipelined else None,
            "model": backends.describe_model(model),
            "blocks": {
                block.type: f"{block.__module__}.{block.__name__}"
                for block in drawing_blocks.BLOCKS
                if block.type in step_types
            },
            "batching": (
                "one run per batch: WorkflowBatchInput of N images; detector "
                "and drawing prep called once each; "
                + (
                    "boxes and labels called once each (batch painters)"
                    if drawing == "batch-painters"
                    else "boxes and labels called N times each"
                )
            ),
            "host_transfers": (
                "drawing prep: one device-to-host transfer per batch; "
                "painters read no prediction tensor"
            ),
            "paint_events": "one readiness event wait per image and painter",
            "readiness": (
                "each block returns ready: TRT phases synchronize their streams; "
                "prep and painters wait on an event recorded on their own stream"
            ),
            "post_process": backends.post_process_parameters(),
        }

    def summarize(self, row: dict) -> Dict[str, Any]:
        """10 ``summarize`` plus ``prep_ms``.

        Args:
            row: One row of ``process_batch`` or ``submit_batch``.

        Returns:
            The 10 summary and the row's ``DRAWING_TIMINGS``.
        """
        summary = super().summarize(row)
        summary.update({name: row[name] for name in DRAWING_TIMINGS if name in row})

        return summary
