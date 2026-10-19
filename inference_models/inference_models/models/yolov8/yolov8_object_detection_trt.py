import logging
import threading
from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, ObjectDetectionModel, PreProcessingOverrides
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
    INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.models.common.cuda import (
    use_cuda_context,
    use_primary_cuda_context,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    TRTConfig,
    parse_class_names_file,
    parse_inference_config,
    parse_trt_config,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.trt import (
    TRTCudaGraphCache,
    establish_trt_cuda_graph_cache,
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
)
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    OptimizationStage,
)
from inference_models.models.optimization.fallback_warnings import (
    FallbackWarningTracker,
)
from inference_models.models.optimization.ids import BASE_IMPLEMENTATION_ID
from inference_models.models.yolov8.optimization import (
    PostprocessRequest,
    create_postprocess_registry,
    postprocess_runtime_components,
    select_postprocessor,
    validate_postprocess_implementation,
)
from inference_models.weights_providers.entities import RecommendedParameters

logger = logging.getLogger(__name__)

try:
    try:
        import tensorrt as trt
    except ImportError:
        import tensorrt_lean as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLOv8 model with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLOv8 model with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support.",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLOv8ForObjectDetectionTRT(
    ObjectDetectionModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache] = None,
        default_trt_cuda_graph_cache_size: int = 8,
        recommended_parameters: Optional[RecommendedParameters] = None,
        postprocess_implementation: str = BASE_IMPLEMENTATION_ID,
        **kwargs,
    ) -> "YOLOv8ForObjectDetectionTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        validate_postprocess_implementation(postprocess_implementation)
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "trt_config.json",
                "engine.plan",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
            },
            implicit_resize_mode_substitutions={
                ResizeMode.FIT_LONGER_EDGE: (
                    ResizeMode.LETTERBOX,
                    127,
                    "YOLOv8 Object Detection model running with TRT backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        if inference_config.post_processing.type != "nms":
            raise CorruptedModelPackageError(
                message="Expected NMS to be the post-processing",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_trt_model(
                model_path=model_package_content["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        inputs, outputs = get_trt_engine_inputs_and_outputs(engine=engine)
        if len(inputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model input, found: {len(inputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if len(outputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model output, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_name=outputs[0],
            class_names=class_names,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
            recommended_parameters=recommended_parameters,
            postprocess_implementation=postprocess_implementation,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_name: str,
        class_names: List[str],
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        recommended_parameters=None,
        postprocess_implementation: str = BASE_IMPLEMENTATION_ID,
    ):
        validate_postprocess_implementation(postprocess_implementation)
        self._engine = engine
        self._input_name = input_name
        self._output_names = [output_name]
        self._class_names = class_names
        self._inference_config = inference_config
        self._trt_config = trt_config
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._lock = threading.Lock()
        self._inference_stream = torch.cuda.Stream(device=self._device)
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters
        self._postprocess_implementation = postprocess_implementation
        self._postprocess_registry = create_postprocess_registry()
        # Subclasses (YOLOv9, YOLOv11, YOLOv12) share this class but not its
        # qualification, so they resolve the segmented stage to base.
        self._postprocess_runtime_components = postprocess_runtime_components(
            model_qualified=type(self) is YOLOv8ForObjectDetectionTRT
        )
        self._postprocess_state = threading.local()
        self._postprocess_fallback_warnings = FallbackWarningTracker()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def postprocess_selection(self) -> Optional[Mapping[str, Any]]:
        """Return this thread's last successful postprocess selection.

        Returns:
            Read-only ``requested_id``, ``effective_id`` and ``fallback_reason``;
            segmented calls add ``routes``, ``scratch_bytes``,
            ``scratch_limit_bytes``, ``kernels`` and ``calibration``. None
            before the first call or after a failed call.
        """
        selection = getattr(self._postprocess_state, "selection", None)

        return selection

    @property
    def postprocess_ready_event(self) -> Optional[torch.cuda.Event]:
        """Return the completion event of this thread's last successful call.

        Returns:
            The event ``post_process`` already waited on, or None before the
            first call or after a failed call.
        """
        event = getattr(self._postprocess_state, "ready_event", None)

        return event

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        with torch.cuda.stream(self._pre_process_stream):
            pre_processed_images, pre_processing_meta = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                pre_processing_overrides=pre_processing_overrides,
            )
        self._pre_process_stream.synchronize()
        return pre_processed_images, pre_processing_meta

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                return infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                    stream=self._inference_stream,
                    trt_cuda_graph_cache=cache,
                )[0]

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        iou_threshold: float = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD,
        max_detections: int = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS,
        class_agnostic_nms: bool = INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS,
        **kwargs,
    ) -> List[Detections]:
        # A failed call must not leave an earlier call's facts behind.
        self._postprocess_state.selection = None
        self._postprocess_state.ready_event = None

        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE,
        )
        confidence = confidence_filter.get_threshold(self.class_names)
        # Decoding runs on this thread's model-owned stream. The caller may
        # still be producing model_results on its current stream, so the
        # stage waits on an event recorded there instead of the host.
        caller_stream = torch.cuda.current_stream(device=self._device)
        stream = self._post_process_stream
        separate_streams = stream != caller_stream
        producer_event = None
        if separate_streams and model_results.is_cuda:
            producer_event = torch.cuda.Event()
            producer_event.record(caller_stream)
        request = PostprocessRequest(
            model_results=model_results,
            pre_processing_meta=pre_processing_meta,
            threshold=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_agnostic_nms=class_agnostic_nms,
            fused=self._inference_config.post_processing.fused,
            ready_event=producer_event,
        )
        context = ExecutionContext(
            device_kind="gpu",
            device=str(stream.device),
            current_stream=stream,
            runtime_components=self._postprocess_runtime_components,
        )

        selection = select_postprocessor(
            registry=self._postprocess_registry,
            requested_id=self._postprocess_implementation,
            request=request,
            context=context,
        )
        self._log_postprocess_fallback(
            requested_id=selection.requested_id,
            effective_id=selection.effective_id,
            reason=selection.fallback_reason,
        )
        result = selection.implementation.postprocess(request, context)
        detections = result.wait()
        if separate_streams:
            # Outputs were allocated on the model stream; keep their memory
            # from reuse there until the caller's later work on them finishes.
            _record_detections_on_stream(detections, stream=caller_stream)

        self._postprocess_state.selection = MappingProxyType(
            {**selection.to_dict(), **result.details}
        )
        self._postprocess_state.ready_event = result.ready_event

        return detections

    def _log_postprocess_fallback(
        self, *, requested_id: str, effective_id: str, reason: Optional[str]
    ) -> None:
        if reason is None:
            return

        if self._postprocess_fallback_warnings.claim(
            stage=OptimizationStage.POSTPROCESS,
            requested_id=requested_id,
            effective_id=effective_id,
            reason=reason,
        ):
            logger.warning(
                "YOLOv8 TensorRT postprocess %r runs %r: %s",
                requested_id,
                effective_id,
                reason,
            )

    @property
    def _pre_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "pre_process_stream"):
            self._thread_local_storage.pre_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.pre_process_stream

    @property
    def _post_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "post_process_stream"):
            self._thread_local_storage.post_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.post_process_stream


def _record_detections_on_stream(
    detections: List[Detections], *, stream: torch.cuda.Stream
) -> None:
    for detection in detections:
        for tensor in (detection.xyxy, detection.class_id, detection.confidence):
            if tensor.is_cuda:
                tensor.record_stream(stream)
