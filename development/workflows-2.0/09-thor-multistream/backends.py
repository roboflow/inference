"""One detector workload, four ways to run it, for the Thor multi-stream runner.

    CUDA CHW RGB uint8 frame (from the runner)
      ├─ v1_numpy    frame -> host BGR numpy; V1 ExecutionEngine, numpy blocks
      ├─ v1_tensor   frame as is;             V1 ExecutionEngine, tensor blocks
      │              roboflow_object_detection_model@v1 -> bounding_box@v1 -> label@v1
      ├─ v2_serial   frame as is; V2 session.run,      gpu_blocks catalogue
      └─ v2_pipeline frame as is; V2 session.pipeline, gpu_blocks catalogue

Every mode uses the same TRT YOLOv8 model, loaded once per process. V1 reaches
it through the server's own stack:

    ModelManagerModelsProvider -> ModelManager -> InferenceModelsObjectDetectionAdapter

The adapter is the stock class; only its constructor is bypassed, so it wraps
the already loaded model instead of loading a second one. The plain V1
detector therefore gets real response dicts, the tensor one native Detections.

Readiness: a result is returned (or its future is done) only when its GPU work
has finished. v1_numpy ends on the host; v1_tensor waits on an event recorded
after ``engine.run`` (V1 tensor painters are asynchronous on CUDA); V2 blocks
are ready at return (see ``gpu_blocks``).

Call order::

    configure_mode(mode)            # before any workflows / inference import
    backend = build_backend(mode, model_id, confidence, device=..., max_in_flight=...)
    backend.process(frame, image_id=...)   # all modes; or submit() for v2_pipeline
    backend.summarize(result); backend.facts; backend.close()
"""

import os
from concurrent.futures import Future
from time import perf_counter
from typing import Any, Dict, Optional

MODES = ("v1_numpy", "v1_tensor", "v2_serial", "v2_pipeline")
TRT_MODEL_CLASS = "YOLOv8ForObjectDetectionTRT"
# Result key the V1 backend adds: v1_numpy frame-to-numpy time, ms; else None.
HOST_CONVERSION_MS = "host_conversion_ms"
# V2 block timings in every V2 result, ms. Model phases synchronize and the
# painters wait for their GPU work, so these include GPU time.
V2_TIMINGS = ("pre_ms", "model_ms", "post_ms", "boxes_ms", "labels_ms")

V1_DETECTOR = "roboflow_core/roboflow_object_detection_model@v1"
V1_BOXES = "roboflow_core/bounding_box_visualization@v1"
V1_LABELS = "roboflow_core/label_visualization@v1"

_configured_mode: Optional[str] = None


def configure_mode(mode: str, *, device: str = "cuda:0") -> None:
    """Install the process configuration for ``mode``, the way the server does.

    Sets ``ENABLE_TENSOR_DATA_REPRESENTATION`` (off for ``v1_numpy`` only) and
    ``WORKFLOWS_IMAGE_TENSOR_DEVICE``, then imports ``inference.core``, which
    builds the Workflows configuration from these variables and installs it.
    Call once, before anything imports ``roboflow_workflows`` or
    ``inference``: block loading reads the configuration at import time.

    Args:
        mode: One of ``MODES``.
        device: Torch device of the frames and the model.

    Raises:
        ValueError: On an unknown mode, a second call with another mode, or a
            preset environment variable that disagrees with ``mode``.
    """
    global _configured_mode
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if _configured_mode is not None and _configured_mode != mode:
        raise ValueError(f"process is configured for {_configured_mode}, not {mode}")

    wanted = {
        "ENABLE_TENSOR_DATA_REPRESENTATION": str(mode != "v1_numpy"),
        "WORKFLOWS_IMAGE_TENSOR_DEVICE": device,
    }
    for name, value in wanted.items():
        preset = os.environ.get(name)
        if preset is not None and preset.lower() != value.lower():
            raise ValueError(f"{name}={preset!r} is set; mode {mode} needs {value!r}")
        os.environ[name] = value

    import inference.core  # noqa: F401 - installs the Workflows configuration

    _configured_mode = mode


def build_backend(
    mode: str,
    model_id: str,
    confidence: float,
    device: str = "cuda:0",
    max_in_flight: int = 4,
) -> "Backend":
    """Load the TRT model and build the backend of ``mode`` around it.

    Args:
        mode: The mode passed to ``configure_mode``.
        model_id: Roboflow model id, e.g. ``yolov8n-640``.
        confidence: Detection confidence threshold, the same in every mode.
        device: CUDA device for the model; frames must be on it too.
        max_in_flight: Pipeline bound; used by ``v2_pipeline`` only.

    Returns:
        The backend, warmed by nothing; the caller warms it up.

    Raises:
        RuntimeError: When ``configure_mode(mode)`` was not called, or the
            loaded model is not the TRT class on ``device``.
    """
    if _configured_mode != mode:
        raise RuntimeError(f"call configure_mode({mode!r}) before build_backend")

    model = load_trt_model(model_id, device=device)
    backend = _build_for_model(
        mode,
        model=model,
        model_id=model_id,
        confidence=confidence,
        max_in_flight=max_in_flight,
    )

    return backend


def load_trt_model(model_id: str, *, device: str) -> Any:
    """Load ``model_id`` as a TensorRT engine on ``device``; no fallback.

    Args:
        model_id: Roboflow model id.
        device: CUDA device, e.g. ``cuda:0``.

    Returns:
        The loaded ``YOLOv8ForObjectDetectionTRT`` model.

    Raises:
        RuntimeError: When AutoModel returns another class or device.
    """
    import torch

    from inference_models import AutoModel

    requested_device = torch.device(device)
    model = AutoModel.from_pretrained(
        model_id,
        backend="trt",
        device=requested_device,
        allow_untrusted_packages=True,
    )
    actual_class = type(model).__name__
    actual_device = getattr(model, "_device", None)
    if actual_class != TRT_MODEL_CLASS or actual_device != requested_device:
        raise RuntimeError(
            f"{model_id}: wanted {TRT_MODEL_CLASS} on {requested_device}, "
            f"got {actual_class} on {actual_device}"
        )

    return model


def describe_model(model: Any) -> Dict[str, Any]:
    """Facts about the loaded model, read from the model, not the request.

    Reads private attributes of the TRT model class (``_device``,
    ``_trt_config``, ``_engine``, ``_input_name``, ``_output_names``).

    Args:
        model: The loaded model.

    Returns:
        Class, module, device, TRT batch config, engine tensor dtypes and
        shapes, and library versions.
    """
    import torch

    description = {
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "device": str(getattr(model, "_device", None)),
        "class_count": len(model.class_names),
    }
    trt_config = getattr(model, "_trt_config", None)
    if trt_config is not None:
        description["trt_config"] = dict(vars(trt_config))
        # Set by ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND at load; same in all modes.
        description["trt_cuda_graphs"] = model._trt_cuda_graph_cache is not None
    engine = getattr(model, "_engine", None)
    if engine is not None:
        import tensorrt

        names = [model._input_name, *model._output_names]
        description["engine_tensors"] = {
            name: {
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
            }
            for name in names
        }
        description["tensorrt"] = tensorrt.__version__
    description["torch"] = torch.__version__
    description["cuda_device_name"] = (
        torch.cuda.get_device_name(model._device)
        if getattr(model, "_device", None) is not None and model._device.type == "cuda"
        else None
    )

    return description


class Backend:
    """What every mode offers the runner.

    ``process`` returns a ready result. ``submit`` exists for ``v2_pipeline``
    only and returns a future that is done when the result is ready.
    """

    mode: str
    device: Any
    facts: Dict[str, Any]

    def process(self, frame: Any, *, image_id: str) -> Any:
        """Run the workflow on one frame and return its ready result.

        Args:
            frame: CUDA CHW RGB uint8 tensor whose pixels are complete. Not
                modified, and not referenced after this call returns.
            image_id: Identity of the frame, e.g. ``s3-f120``.

        Returns:
            The engine's result for this frame.
        """
        raise NotImplementedError

    def summarize(self, result: Any) -> Dict[str, Any]:
        """Small JSON-friendly view of one result.

        Args:
            result: A result of ``process`` or of a ``submit`` future.

        Returns:
            ``detections`` (count), plus ``host_conversion_ms`` for V1 or the
            ``V2_TIMINGS`` block times for V2.
        """
        raise NotImplementedError

    def predictions(self, result: Any) -> Any:
        """The ``predictions`` output: ``sv.Detections`` (v1_numpy) or native
        ``Detections`` (other modes)."""
        raise NotImplementedError

    def annotated(self, result: Any) -> Any:
        """The ``annotated`` output: HWC BGR numpy (v1_numpy) or a CUDA CHW RGB
        tensor (other modes)."""
        raise NotImplementedError

    def warm_up(self, *, size_hw: tuple = (1080, 1920), runs: int = 3) -> float:
        """Run ``process`` on blank frames, before any source starts.

        A blank frame has no detections, so the first real frames still pay
        for the first label sprites.

        Args:
            size_hw: Frame height and width.
            runs: Number of blank frames.

        Returns:
            Seconds the warmup took.
        """
        import torch

        started = perf_counter()
        for run in range(runs):
            blank = torch.zeros((3, *size_hw), dtype=torch.uint8, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            self.process(blank, image_id=f"warmup-{run}")
        seconds = perf_counter() - started

        return seconds

    def close(self) -> None:
        """Release what the backend opened."""


class V1Backend(Backend):
    """The V1 ExecutionEngine with stock blocks, numpy or tensor representation."""

    def __init__(
        self, mode: str, *, model: Any, model_id: str, confidence: float
    ) -> None:
        import torch
        from roboflow_workflows.core_steps.common.entities import StepExecutionMode
        from roboflow_workflows.execution_engine.core import ExecutionEngine

        from inference.core.interfaces.workflows_models_provider import (
            ModelManagerModelsProvider,
        )
        from inference.core.managers.base import ModelManager

        self.mode = mode
        self._torch = torch
        self._confidence = confidence
        self.device = model._device
        model_manager = ModelManager(
            model_registry=None, models={model_id: _stock_adapter(model)}
        )
        self._engine = ExecutionEngine.init(
            workflow_definition=v1_workflow(model_id),
            init_parameters={
                "workflows_core.model_manager": ModelManagerModelsProvider(
                    model_manager
                ),
                "workflows_core.api_key": None,
                "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
            },
        )
        self.facts = {
            "mode": mode,
            "engine": "V1 ExecutionEngine",
            "model": describe_model(model),
            "blocks": _v1_block_modules(self._engine),
            "model_access": (
                "ModelManagerModelsProvider -> ModelManager -> stock "
                "InferenceModelsObjectDetectionAdapter wrapping the loaded model"
            ),
            "input": (
                "CUDA frame -> RGB->BGR + HWC on GPU -> one device-to-host copy "
                "-> numpy, inside process()"
                if mode == "v1_numpy"
                else "CUDA frame used as WorkflowImageData.tensor_image"
            ),
            "readiness": (
                "results are host numpy after engine.run"
                if mode == "v1_numpy"
                else "event recorded on the current stream after engine.run, "
                "then host wait on that event"
            ),
            "post_process": post_process_parameters(),
        }

    def process(self, frame: Any, *, image_id: str) -> Any:
        from roboflow_workflows.execution_engine.entities.base import (
            ImageParentMetadata,
            WorkflowImageData,
        )

        parent = ImageParentMetadata(parent_id=image_id)
        host_conversion_ms = None
        if self.mode == "v1_numpy":
            started = perf_counter()
            pixels = _host_bgr(frame)
            host_conversion_ms = (perf_counter() - started) * 1000.0
            image = WorkflowImageData(parent_metadata=parent, numpy_image=pixels)
        else:
            image = WorkflowImageData(parent_metadata=parent, tensor_image=frame)

        (result,) = self._engine.run(
            runtime_parameters={"image": image, "confidence": self._confidence}
        )
        if self.mode == "v1_tensor" and self.device.type == "cuda":
            done = self._torch.cuda.Event()
            done.record(self._torch.cuda.current_stream(self.device))
            done.synchronize()
        # Not an engine output: the share of process() spent on the plain path's
        # own input conversion.
        result[HOST_CONVERSION_MS] = host_conversion_ms

        return result

    def summarize(self, result: Any) -> Dict[str, Any]:
        summary = {
            "detections": len(self.predictions(result)),
            HOST_CONVERSION_MS: result[HOST_CONVERSION_MS],
        }

        return summary

    def predictions(self, result: Any) -> Any:
        return result["predictions"]

    def annotated(self, result: Any) -> Any:
        image = result["annotated"]
        pixels = image.numpy_image if self.mode == "v1_numpy" else image.tensor_image

        return pixels


class V2Backend(Backend):
    """The V2 engine with ``gpu_blocks``: ``session.run`` or ``session.pipeline``."""

    def __init__(
        self,
        mode: str,
        *,
        model: Any,
        confidence: float,
        max_in_flight: int,
    ) -> None:
        import gpu_blocks
        from roboflow_workflows.execution_engine.v2.compilation import (
            compile_workflow,
        )
        from roboflow_workflows.execution_engine.v2.pipelining import (
            PipelineOptions,
        )
        from roboflow_workflows.execution_engine.v2.plan import CompileOptions

        self.mode = mode
        self._confidence = confidence
        self.device = model._device
        pipelined = mode == "v2_pipeline"
        plan = compile_workflow(
            V2_WORKFLOW,
            catalogue=gpu_blocks.create_catalogue(),
            options=CompileOptions(
                block_execution="phases" if pipelined else "run",
                mutation_conflicts="error",
            ),
        )
        self._session = plan.create_session({"detection_model": model})
        self._pipeline = None
        if pipelined:
            self._pipeline = self._session.pipeline(
                options=PipelineOptions(max_in_flight=max_in_flight)
            )
            self._pipeline.__enter__()
        self.facts = {
            "mode": mode,
            "engine": "V2 " + ("session.pipeline" if pipelined else "session.run"),
            "max_in_flight": max_in_flight if pipelined else None,
            "model": describe_model(model),
            "blocks": {
                block.type: f"{block.__module__}.{block.__name__}"
                for block in gpu_blocks.BLOCKS
            },
            "model_access": "the loaded model passed as session resource",
            "input": "CUDA frame used as ImageData.tensor_image",
            "readiness": (
                "each block returns ready: TRT phases synchronize their streams; "
                "painters wait on an event recorded on their own stream"
            ),
            "post_process": post_process_parameters(),
        }

    def submit(self, frame: Any, *, image_id: str) -> Future:
        """Submit one frame to the pipeline (``v2_pipeline`` only).

        Args:
            frame: As for ``process``; must stay alive until the future is done.
            image_id: Identity of the frame.

        Returns:
            The pipeline's future; done when the result is ready.

        Raises:
            RuntimeError: In ``v2_serial`` mode.
        """
        if self._pipeline is None:
            raise RuntimeError("submit needs mode v2_pipeline")

        future = self._pipeline.submit(self._inputs(frame, image_id=image_id))

        return future

    def process(self, frame: Any, *, image_id: str) -> Any:
        if self._pipeline is not None:
            result = self.submit(frame, image_id=image_id).result()
            return result

        result = self._session.run(self._inputs(frame, image_id=image_id))

        return result

    def summarize(self, result: Any) -> Dict[str, Any]:
        summary = {"detections": len(self.predictions(result))}
        for name in V2_TIMINGS:
            summary[name] = _v2_output(result, name)

        return summary

    def predictions(self, result: Any) -> Any:
        predictions = _v2_output(result, "predictions")

        return predictions

    def annotated(self, result: Any) -> Any:
        pixels = _v2_output(result, "annotated").tensor_image

        return pixels

    def close(self) -> None:
        if self._pipeline is not None:
            pipeline, self._pipeline = self._pipeline, None
            pipeline.__exit__(None, None, None)

    def _inputs(self, frame: Any, *, image_id: str) -> Dict[str, Any]:
        from roboflow_workflows.execution_engine.v2.blocks.image_data import (
            ImageData,
        )

        inputs = {
            "image": ImageData.from_tensor(frame, image_id=image_id),
            "confidence": self._confidence,
        }

        return inputs


def post_process_parameters() -> Dict[str, Any]:
    """NMS values for V1, equal to the model defaults the V2 detector uses.

    Returns:
        ``iou_threshold``, ``max_detections`` and ``class_agnostic_nms``.
    """
    from inference_models.configuration import (
        INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS as CLASS_AGNOSTIC,
    )
    from inference_models.configuration import (
        INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD as IOU_THRESHOLD,
    )
    from inference_models.configuration import (
        INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS as MAX_DETECTIONS,
    )

    parameters = {
        "iou_threshold": IOU_THRESHOLD,
        "max_detections": MAX_DETECTIONS,
        "class_agnostic_nms": CLASS_AGNOSTIC,
    }

    return parameters


def v1_workflow(model_id: str) -> Dict[str, Any]:
    """The V1 definition: stock detector, boxes on a copy, labels in place.

    Args:
        model_id: Model the detector step asks the models provider for.

    Returns:
        A V1 workflow definition with outputs ``annotated`` and ``predictions``.
    """
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "confidence"},
        ],
        "steps": [
            {
                "type": V1_DETECTOR,
                "name": "detector",
                "images": "$inputs.image",
                "model_id": model_id,
                "confidence": "$inputs.confidence",
                **post_process_parameters(),
            },
            {
                "type": V1_BOXES,
                "name": "boxes",
                "image": "$inputs.image",
                "predictions": "$steps.detector.predictions",
            },
            {
                "type": V1_LABELS,
                "name": "labels",
                "image": "$steps.boxes.image",
                "predictions": "$steps.detector.predictions",
                "text": "Class and Confidence",
                "copy_image": False,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "annotated",
                "selector": "$steps.labels.image",
            },
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.detector.predictions",
            },
        ],
    }

    return definition


V2_WORKFLOW = {
    "version": "2.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "image", "kind": ["image"]},
        {"type": "WorkflowParameter", "name": "confidence", "kind": ["float"]},
    ],
    "steps": [
        {
            "type": "live_detection/object_detector@v1",
            "name": "detector",
            "image": "$inputs.image",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "thor_detection/box_visualization@v1",
            "name": "boxes",
            "image": "$inputs.image",
            "predictions": "$steps.detector.predictions",
        },
        {
            "type": "thor_detection/label_visualization@v1",
            "name": "labels",
            "image": "$steps.boxes.image",
            "predictions": "$steps.detector.predictions",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "annotated", "selector": "$steps.labels.image"},
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.detector.predictions",
        },
        {"type": "JsonField", "name": "pre_ms", "selector": "$steps.detector.pre_ms"},
        {
            "type": "JsonField",
            "name": "model_ms",
            "selector": "$steps.detector.model_ms",
        },
        {
            "type": "JsonField",
            "name": "post_ms",
            "selector": "$steps.detector.post_ms",
        },
        {"type": "JsonField", "name": "boxes_ms", "selector": "$steps.boxes.draw_ms"},
        {
            "type": "JsonField",
            "name": "labels_ms",
            "selector": "$steps.labels.draw_ms",
        },
    ],
}


def _build_for_model(
    mode: str,
    *,
    model: Any,
    model_id: str,
    confidence: float,
    max_in_flight: int,
) -> Backend:
    # Build either engine around the already-loaded model.
    if mode.startswith("v1_"):
        backend = V1Backend(mode, model=model, model_id=model_id, confidence=confidence)
        return backend

    backend = V2Backend(
        mode, model=model, confidence=confidence, max_in_flight=max_in_flight
    )

    return backend


def _stock_adapter(model: Any) -> Any:
    """The server's object detection adapter around an already loaded model.

    Mirrors ``InferenceModelsObjectDetectionAdapter.__init__`` line by line,
    with ``AutoModel.from_pretrained`` replaced by ``model``. Every other
    method is the stock one.
    """
    from inference.core.models import inference_models_adapters as adapters
    from inference.core.models.base import Model

    adapter = adapters.InferenceModelsObjectDetectionAdapter.__new__(
        adapters.InferenceModelsObjectDetectionAdapter
    )
    Model.__init__(adapter)
    adapter.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
    adapter.api_key = None
    adapter.task_type = "object-detection"
    adapter._model = model
    fixed_input_hw = adapters._fixed_input_hw_from_backend(model)
    if fixed_input_hw:
        adapter.img_size_h, adapter.img_size_w = fixed_input_hw
    adapter._preprocess_supports_independent_stage_execution = (
        adapters._supports_independent_stage_execution(model.pre_process)
    )
    adapter.class_names = list(model.class_names)

    return adapter


def _host_bgr(frame: Any) -> Any:
    # What a plain V1 host holds: an HWC BGR numpy frame. Reordering on the
    # GPU first leaves one contiguous device-to-host copy.
    bgr_hwc = frame.flip(0).permute(1, 2, 0).contiguous()
    pixels = bgr_hwc.cpu().numpy()

    return pixels


def _v1_block_modules(engine: Any) -> Dict[str, str]:
    # Private engine attributes; read once for the facts, never for behavior.
    steps = engine._engine._compiled_workflow.steps
    modules = {
        name: f"{type(step.step).__module__}.{type(step.step).__name__}"
        for name, step in steps.items()
    }

    return modules


def _v2_output(result: Any, name: str) -> Any:
    (entry,) = result.selections[name].values()
    value = result.outputs.data[entry]

    return value
