"""One detector workload in physical batches of N frames, three ways.

    N CUDA CHW RGB frames + N image ids
      ├─ v1_tensor    one V1 engine.run(image=[N WorkflowImageData])   (09 v1_batch_runner)
      ├─ v2_serial    one V2 session.run({"image": [N ImageData]})
      └─ v2_pipeline  one V2 pipeline.submit({"image": [N ImageData]}) per batch,
                      up to pipeline_depth batches executing at once
      -> N rows, in input order, all ready

The V2 workflow feeds the images as a ``WorkflowBatchInput``. The batched
detector (``batched_blocks``) is called once per batch; the 09 painters once
per image. ``stages`` cuts the V2 workflow after the detector or the box
painter, for held-frame ablations; the V1 comparator is always the full one.
``v2_workflow(stages, batched=False)`` builds the per-image 09 workflow with the
same cuts, for comparisons.

Importing this module imports ``thor_imports`` (08/09 search paths) and nothing
heavy, so ``backends.configure_mode`` can still run first::

    import batched_backend, backends
    backends.configure_mode("v2_pipeline", device="cuda:0")
    backend = batched_backend.build_backend("v2_pipeline", "yolov8n-640", 0.4)
    rows = backend.process_batch(frames, image_ids=ids)      # or submit_batch
"""

import itertools
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Sequence

import thor_imports

MODES = ("v1_tensor", "v2_serial", "v2_pipeline")
STAGES = ("detector", "boxes", "full")
BATCH_SIZE_KEY = "batch_size"  # the same key as 09 run_batched
# Every row names its batch: a process-wide index, N, and time.monotonic_ns()
# when the backend received the batch. With frames.csv admit/ready times this
# splits a frame's latency into collection wait and batch execution.
BATCH_KEYS = ("batch_index", BATCH_SIZE_KEY, "batch_started_ns")
# Row timings, ms, when the stage exists. Detector phases: the whole batch.
# Painters: this image only, including the wait for its GPU work.
ROW_TIMINGS = ("pre_ms", "model_ms", "post_ms", "boxes_ms", "labels_ms")


def build_backend(
    mode: str,
    model_id: str,
    confidence: float,
    *,
    device: str = "cuda:0",
    pipeline_depth: int = 2,
    stages: str = "full",
) -> "BatchedBackend":
    """Load the TRT model (09, strict) and build the batched backend around it.

    Args:
        mode: One of ``MODES``; ``backends.configure_mode(mode)`` must have run.
        model_id: Roboflow model id, e.g. ``yolov8n-640``.
        confidence: Detection confidence threshold.
        device: CUDA device of the model and the frames.
        pipeline_depth: ``v2_pipeline`` only: batches executing at once;
            other modes ignore it.
        stages: One of ``STAGES``.

    Returns:
        The backend, not warmed up.
    """
    import backends

    model = backends.load_trt_model(model_id, device=device)
    backend = build_for_model(
        mode,
        model=model,
        model_id=model_id,
        confidence=confidence,
        pipeline_depth=pipeline_depth,
        stages=stages,
    )

    return backend


def build_for_model(
    mode: str,
    *,
    model: Any,
    model_id: str,
    confidence: float,
    pipeline_depth: int = 2,
    stages: str = "full",
) -> "BatchedBackend":
    """Build the batched backend of ``mode`` around an already loaded model.

    Args:
        mode: One of ``MODES``.
        model: Loaded ``inference_models`` object detection model with
            ``_device``; any backend (TRT on Thor, a fake or ONNX in CPU checks).
        model_id: Key under which the V1 models provider serves ``model``.
        confidence: Detection confidence threshold.
        pipeline_depth: ``v2_pipeline`` only: batches executing at once;
            other modes ignore it.
        stages: One of ``STAGES``; ``v1_tensor`` supports ``full`` only.

    Returns:
        The backend.

    Raises:
        ValueError: On an unknown mode or stage, or a cut V1 workflow.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if stages not in STAGES:
        raise ValueError(f"stages must be one of {STAGES}, got {stages!r}")
    if mode == "v1_tensor":
        if stages != "full":
            raise ValueError(
                "v1_tensor runs the full 09 V1 workflow; use stages='full'"
            )
        backend = V1TensorBatchedBackend(
            model=model, model_id=model_id, confidence=confidence
        )
        return backend

    backend = V2BatchedBackend(
        mode,
        model=model,
        confidence=confidence,
        pipeline_depth=pipeline_depth,
        stages=stages,
    )

    return backend


class BatchedBackend:
    """What every mode offers: ready rows for a list of frames.

    A row is a dict with ``predictions``, ``annotated`` (absent before the box
    stage), the ``BATCH_KEYS`` and the mode's timings (module doc).
    """

    mode: str
    device: Any
    facts: Dict[str, Any]

    def process_batch(
        self, frames: Sequence[Any], *, image_ids: Sequence[str]
    ) -> List[dict]:
        """Run the workflow once on ``frames`` and return their ready rows.

        Args:
            frames: N >= 1 CHW RGB uint8 tensors with complete pixels, on the
                model's device. Not modified; not referenced after return.
            image_ids: N distinct identities, in the order of ``frames``.

        Returns:
            N rows, in input order; their GPU work has finished.
        """
        raise NotImplementedError

    def submit_batch(
        self, frames: Sequence[Any], *, image_ids: Sequence[str]
    ) -> Future:
        """Submit ``frames`` as one pipelined batch (``v2_pipeline`` only).

        Args:
            frames: As for ``process_batch``; alive until the future is done.
            image_ids: As for ``process_batch``.

        Returns:
            A future of the N ready rows.

        Raises:
            RuntimeError: In any other mode.
        """
        raise RuntimeError(f"submit_batch needs mode v2_pipeline, not {self.mode}")

    def summarize(self, row: dict) -> Dict[str, Any]:
        """Small JSON-friendly view of one row.

        Args:
            row: One row of ``process_batch`` or ``submit_batch``.

        Returns:
            ``image_id`` (from the predictions' metadata), ``detections``,
            the ``BATCH_KEYS`` and the row's timings.
        """
        predictions = self.predictions(row)
        summary = {
            "image_id": (predictions.image_metadata or {}).get("parent_id"),
            "detections": len(predictions),
        }
        summary.update({name: row[name] for name in BATCH_KEYS})
        summary.update({name: row[name] for name in ROW_TIMINGS if name in row})

        return summary

    def predictions(self, row: dict) -> Any:
        """The row's native ``Detections``."""
        return row["predictions"]

    def annotated(self, row: dict) -> Optional[Any]:
        """The row's annotated CHW RGB tensor, or None without a painter stage."""
        image = row.get("annotated")
        pixels = None if image is None else image.tensor_image

        return pixels

    def close(self) -> None:
        """Release what the backend opened."""


class V1TensorBatchedBackend(BatchedBackend):
    """09's batched V1 comparator: stock engine and blocks, one run per batch."""

    def __init__(self, *, model: Any, model_id: str, confidence: float) -> None:
        import backends

        v1_batch_runner = thor_imports.load_09_module("run_batched").v1_batch_runner
        single = backends._build_for_model(
            "v1_tensor",
            model=model,
            model_id=model_id,
            confidence=confidence,
            max_in_flight=1,
        )
        self.mode = "v1_tensor"
        self.device = single.device
        self._run_batch = v1_batch_runner(single)
        self.facts = {
            **single.facts,
            "stages": "full",
            "batching": "09 run_batched.v1_batch_runner: one engine.run per batch",
        }

    def process_batch(
        self, frames: Sequence[Any], *, image_ids: Sequence[str]
    ) -> List[dict]:
        batch = _new_batch(frames, image_ids=image_ids)
        rows = self._run_batch(list(frames), list(image_ids))
        for row in rows:
            row.update(batch)

        return rows


class V2BatchedBackend(BatchedBackend):
    """The V2 engine with ``batched_blocks``: ``session.run`` or ``session.pipeline``."""

    def __init__(
        self,
        mode: str,
        *,
        model: Any,
        confidence: float,
        pipeline_depth: int,
        stages: str,
    ) -> None:
        import backends
        import batched_blocks
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
        definition = v2_workflow(stages)
        plan = compile_workflow(
            definition,
            catalogue=batched_blocks.create_catalogue(),
            options=CompileOptions(
                block_execution="phases" if pipelined else "run",
                mutation_conflicts="error",
            ),
        )
        self._session = plan.create_session({"detection_model": model})
        self._pipeline = None
        if pipelined:
            self._pipeline = self._session.pipeline(
                options=PipelineOptions(max_in_flight=pipeline_depth)
            )
            self._pipeline.__enter__()
        self.facts = {
            "mode": mode,
            "engine": "V2 " + ("session.pipeline" if pipelined else "session.run"),
            "block_execution": "phases" if pipelined else "run",
            "stages": stages,
            "pipeline_depth": pipeline_depth if pipelined else None,
            "model": backends.describe_model(model),
            "blocks": {
                block.type: f"{block.__module__}.{block.__name__}"
                for block in batched_blocks.BLOCKS
                if block.type in {step["type"] for step in definition["steps"]}
            },
            "batching": (
                "one run per batch: WorkflowBatchInput of N images; the detector "
                "step is called once (pre_process, forward, post_process on N), "
                "each painter step N times"
            ),
            "readiness": (
                "each block returns ready: TRT phases synchronize their streams; "
                "painters wait on an event recorded on their own stream"
            ),
            "post_process": backends.post_process_parameters(),
        }

    def process_batch(
        self, frames: Sequence[Any], *, image_ids: Sequence[str]
    ) -> List[dict]:
        if self._pipeline is not None:
            rows = self.submit_batch(frames, image_ids=image_ids).result()
            return rows

        batch = _new_batch(frames, image_ids=image_ids)
        result = self._session.run(self._inputs(frames, image_ids=image_ids))
        rows = _rows(result, batch=batch)

        return rows

    def submit_batch(
        self, frames: Sequence[Any], *, image_ids: Sequence[str]
    ) -> Future:
        if self._pipeline is None:
            return super().submit_batch(frames, image_ids=image_ids)

        batch = _new_batch(frames, image_ids=image_ids)
        rows: Future = Future()
        run = self._pipeline.submit(self._inputs(frames, image_ids=image_ids))
        run.add_done_callback(lambda done: _complete_rows(rows, run=done, batch=batch))

        return rows

    def close(self) -> None:
        if self._pipeline is not None:
            pipeline, self._pipeline = self._pipeline, None
            pipeline.__exit__(None, None, None)

    def _inputs(self, frames: Sequence[Any], *, image_ids: Sequence[str]) -> dict:
        from roboflow_workflows.execution_engine.v2.blocks.image_data import (
            ImageData,
        )

        inputs = {
            "image": [
                ImageData.from_tensor(frame, image_id=image_id)
                for frame, image_id in zip(frames, image_ids)
            ],
            "confidence": self._confidence,
        }

        return inputs


def v2_workflow(stages: str, *, batched: bool = True) -> Dict[str, Any]:
    """The 09 V2 workflow, cut after ``stages``, batched or per image.

    The two variants differ only in the image input and the detector step type;
    step names, painters and outputs are the same. ``batched=False`` with
    ``stages="full"`` is 09 ``backends.V2_WORKFLOW``. Both compile with
    ``batched_blocks.create_catalogue()``.

    Args:
        stages: ``detector``, ``boxes`` (adds the box painter; ``annotated``
            is its image) or ``full`` (adds the label painter).
        batched: True: a ``WorkflowBatchInput`` image and the batched
            detector, one model call per run. False: one image per run
            (``WorkflowParameter``) and the 08 detector.

    Returns:
        A V2 definition. Outputs: ``predictions``, the detector timings, and
        per painter stage ``annotated`` plus its ``*_ms`` timing.

    Raises:
        ValueError: On an unknown stage.
    """
    if stages not in STAGES:
        raise ValueError(f"stages must be one of {STAGES}, got {stages!r}")

    image_input = "WorkflowBatchInput" if batched else "WorkflowParameter"
    detector_type = (
        "thor_batching/object_detector@v1"
        if batched
        else "live_detection/object_detector@v1"
    )
    steps = [
        {
            "type": detector_type,
            "name": "detector",
            "image": "$inputs.image",
            "confidence": "$inputs.confidence",
        }
    ]
    outputs = {
        "predictions": "$steps.detector.predictions",
        "pre_ms": "$steps.detector.pre_ms",
        "model_ms": "$steps.detector.model_ms",
        "post_ms": "$steps.detector.post_ms",
    }
    if stages in ("boxes", "full"):
        steps.append(
            {
                "type": "thor_detection/box_visualization@v1",
                "name": "boxes",
                "image": "$inputs.image",
                "predictions": "$steps.detector.predictions",
            }
        )
        outputs.update(annotated="$steps.boxes.image", boxes_ms="$steps.boxes.draw_ms")
    if stages == "full":
        steps.append(
            {
                "type": "thor_detection/label_visualization@v1",
                "name": "labels",
                "image": "$steps.boxes.image",
                "predictions": "$steps.detector.predictions",
            }
        )
        outputs.update(
            annotated="$steps.labels.image", labels_ms="$steps.labels.draw_ms"
        )
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": image_input, "name": "image", "kind": ["image"]},
            {"type": "WorkflowParameter", "name": "confidence", "kind": ["float"]},
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


_batch_indices = itertools.count()


def _new_batch(frames: Sequence[Any], *, image_ids: Sequence[str]) -> Dict[str, int]:
    # The BATCH_KEYS values of one batch, stamped on arrival.
    if not frames or len(frames) != len(image_ids):
        raise ValueError(
            f"need N >= 1 frames and N image ids, got {len(frames)} and {len(image_ids)}"
        )

    batch = {
        "batch_index": next(_batch_indices),
        BATCH_SIZE_KEY: len(frames),
        "batch_started_ns": time.monotonic_ns(),
    }

    return batch


def _rows(result: Any, *, batch: Dict[str, int]) -> List[dict]:
    # One row per input image, in input order (RunResult.rows); payloads as is.
    rows = result.rows()
    for row in rows:
        row.update(batch)

    return rows


def _complete_rows(rows: Future, *, run: Future, batch: Dict[str, int]) -> None:
    # Runs on the pipeline worker that finished the run; builds rows only.
    try:
        rows.set_result(_rows(run.result(), batch=batch))
    except BaseException as error:  # noqa: BLE001 - the caller's future fails
        rows.set_exception(error)
