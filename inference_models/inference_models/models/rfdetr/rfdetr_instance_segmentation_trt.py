import threading
from dataclasses import replace
from typing import Any, List, Mapping, Optional, Set, Tuple, Union

import numpy as np
import torch
from inference_models import (
    InstanceDetections,
    InstanceSegmentationMaskFormat,
    InstanceSegmentationModel,
    PreProcessingOverrides,
)
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_RFDETR_DEFAULT_MAX_DETECTIONS,
    INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED,
    get_rfdetr_pipeline_depth,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.base.async_handoff import (
    get_deferred_postprocess_done_event,
    get_trt_outputs_consumed_event,
)

# Hoisted to module scope to avoid per-call `from ... import` inside the hot
# forward_async path. Re-import inside the function added ~13µs/frame in the
# instrumented run on Jetson Orin. Import here is a no-op on every call.
from inference_models.models.base.instance_segmentation import _DirectInferenceFuture
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
from inference_models.models.common.trt import (
    TRTCudaGraphCache,
    establish_trt_cuda_graph_cache,
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
)
from inference_models.models.optimization.contracts import ExecutionContext
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.optimization.runtime_components import (
    get_runtime_components,
)
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import (
    post_process_instance_segmentation_results,
    post_process_instance_segmentation_results_to_grid,
    post_process_instance_segmentation_results_to_rle_masks,
)
from inference_models.models.rfdetr.optimization.contracts import PreprocessRequest
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
    _normalize_execution_plan_argument,
)
from inference_models.models.rfdetr.optimization.preprocessor_selection import (
    PreprocessorSelector,
)
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)
from inference_models.models.rfdetr.optimization.segmentation_preprocessing import (
    _build_segmentation_preprocessor_registry,
    _resolve_segmentation_plan,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    try:
        import tensorrt as trt
    except ImportError:
        import tensorrt_lean as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import RFDetr model with TRT backend - this error means that some additional dependencies "
        f"are not installed in the environment.  If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support. "
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running model RFDETR with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class RFDetrForInstanceSegmentationTRT(
    InstanceSegmentationModel[
        torch.Tensor,
        PreProcessingMetadata,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache] = None,
        default_trt_cuda_graph_cache_size: int = 8,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        execution_plan: Optional[Union[RFDetrExecutionPlan, Mapping[str, Any]]] = None,
        **kwargs,
    ) -> "RFDetrForInstanceSegmentationTRT":
        """Load a segmentation TensorRT model with selectable preprocessing.

        Args:
            model_name_or_path: Local model package directory.
            device: CUDA inference device.
            engine_host_code_allowed: Permit execution of engine host code.
            trt_cuda_graph_cache: Optional caller-owned graph cache.
            default_trt_cuda_graph_cache_size: Capacity of the default graph cache.
            rf_detr_max_input_resolution: Optional maximum network resolution.
            recommended_parameters: Model-specific recommended parameters.
            execution_plan: Typed RF-DETR plan or canonical serialized mapping.
                Only preprocessing is selectable; other stages retain base behavior.
                An explicit plan overrides environment selections.
            **kwargs: Compatible loader arguments, including the deprecated
                rfdetr_execution_plan alias (cannot be combined with a plan).

        Returns:
            Loaded segmentation model.

        Raises:
            ModelRuntimeError: If the device or selected implementation is unsupported.
            ValueError: If the plan selects an unsupported non-preprocessing stage.
        """
        execution_plan = _normalize_execution_plan_argument(
            execution_plan=execution_plan,
            kwargs=kwargs,
        )
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
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
                    ResizeMode.STRETCH_TO,
                    None,
                    "RFDetr Instance Segmentation model running with TRT backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "RFDetr models. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
            max_allowed_input_size=rf_detr_max_input_resolution,
        )
        classes_re_mapping = None
        if inference_config.class_names_operations:
            class_names, classes_re_mapping = prepare_class_remapping(
                class_names=class_names,
                class_names_operations=inference_config.class_names_operations,
                device=device,
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
        if len(outputs) != 3:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 3 model outputs, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=outputs,
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
            recommended_parameters=recommended_parameters,
            execution_plan=execution_plan,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_names: List[str],
        class_names: List[str],
        classes_re_mapping: Optional[ClassesReMapping],
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        recommended_parameters=None,
        execution_plan: Optional[Union[RFDetrExecutionPlan, Mapping[str, Any]]] = None,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = output_names
        self._inference_config = inference_config
        self._class_names = class_names
        self._classes_re_mapping = classes_re_mapping
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if device.type == "cuda" and device.index is None
            else device
        )
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_config = trt_config
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._lock = threading.Lock()
        self._inference_stream = torch.cuda.Stream(device=self._device)
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters
        self._stream_pipeline_enabled = get_rfdetr_pipeline_depth() > 1
        if self._stream_pipeline_enabled:
            self._pre_process_cuda_stream = torch.cuda.Stream(device=self._device)
            self._post_process_cuda_stream = torch.cuda.Stream(device=self._device)
        self._rfdetr_execution_plan = _resolve_segmentation_plan(
            execution_plan,
            legacy_enabled=INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED,
        )
        self._preprocess_readiness = PreprocessReadinessTracker()
        self._preprocess_context = ExecutionContext(
            device_kind="gpu",
            device=str(self._device),
            current_stream=None,
            compute_capability=torch.cuda.get_device_capability(self._device),
            runtime_components=get_runtime_components(),
        )
        registry = _build_segmentation_preprocessor_registry(device=self._device)
        self._preprocessor_selector = PreprocessorSelector(
            registry=registry,
            context=self._preprocess_context,
            image_pre_processing=inference_config.image_pre_processing,
            network_input=inference_config.network_input,
        )
        selection = self._preprocessor_selector.resolve_model(
            requested_id=self._rfdetr_execution_plan.preprocessor_id,
            allow_fallback=self._rfdetr_execution_plan.allow_compatibility_fallback,
        )
        self._preprocessor = selection.implementation
        self._preprocessor_model_fallback_reason = selection.fallback_reason

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def supported_mask_formats(self) -> Set[InstanceSegmentationMaskFormat]:
        return {"dense", "rle", "grid"}

    @property
    def supports_stream_pipeline(self) -> bool:
        return self._trt_cuda_graph_cache is not None

    @property
    def preprocessor_implementation_id(self) -> str:
        """Return the model-selected preprocessor ID."""
        return self._preprocessor.metadata.implementation_id

    @property
    def preprocessor_implementation_metadata(self):
        """Return typed metadata for the model-selected preprocessor."""
        return self._preprocessor.metadata

    @property
    def last_preprocessor_selection(self):
        """Return this thread's last completed preprocessing route, or None."""
        return getattr(self._thread_local_storage, "preprocessor_selection", None)

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        independent_stage_execution: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        """Preprocess through the selected implementation and preserve readiness.

        Args:
            images: Single image or image batch.
            input_color_format: Caller channel order.
            image_size: Optional reference-path resize override, in width/height order.
            pre_processing_overrides: Request transformation overrides.
            independent_stage_execution: Synchronize output before return when True.
                Otherwise pass this exact tensor to this model's forward method.
            **kwargs: Additional inference arguments accepted for compatibility.

        Returns:
            Per-call FP32 input tensor and image transformation metadata.

        Raises:
            ModelRuntimeError: If selection or a recognized runtime failure cannot
                fall back under the requested execution plan.
        """
        stream = self._pre_process_stream
        # Caller-owned CUDA inputs must be ready on the caller's current stream.
        # Wait once before either implementation reads them; record their lifetime.
        stream.wait_stream(torch.cuda.current_stream(self._device))
        items = images if isinstance(images, list) else [images]
        for image in items:
            if isinstance(image, torch.Tensor) and image.device == stream.device:
                image.record_stream(stream)
        request = PreprocessRequest(
            images=images,
            input_color_format=input_color_format,
            image_size_wh=image_size,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            pre_processing_overrides=pre_processing_overrides,
        )
        context = replace(self._preprocess_context, current_stream=stream)
        selector = self._preprocessor_selector
        plan = self._rfdetr_execution_plan
        selection = selector.resolve_request(
            implementation=self._preprocessor,
            request=request,
            context=context,
            allow_fallback=plan.allow_compatibility_fallback,
        )
        allow_runtime = (
            plan.allow_compatibility_fallback and plan.allow_runtime_failure_fallback
        )
        try:
            selection = selector.resolve_runtime_fallback(
                selection=selection,
                request=request,
                context=context,
                allow_fallback=allow_runtime,
            )
            attempted = set()
            while True:
                attempted.add(selection.effective_id)
                try:
                    result = selection.implementation.preprocess(
                        request=request, context=context
                    )
                    break
                except RecoverableStageExecutionError:
                    # Drain any partial submissions before fallback reuses storage.
                    stream.synchronize()
                    if not allow_runtime:
                        raise

                    fallback = selector.resolve_runtime_fallback(
                        selection=selection,
                        request=request,
                        context=context,
                        allow_fallback=True,
                    )
                    if fallback.effective_id in attempted:
                        raise

                    selection = fallback
        except RecoverableStageExecutionError as error:
            raise ModelRuntimeError(
                message=str(error),
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            ) from error

        fallback_reason = (
            selection.fallback_reason or self._preprocessor_model_fallback_reason
        )
        tensor = result.tensor
        tensor.record_stream(stream)
        event = result.ready_event
        if event is None:
            event = torch.cuda.Event()
            event.record(stream)
        if independent_stage_execution:
            event.synchronize()
        else:
            self._preprocess_readiness.record(
                tensor,
                ready_event=event,
                input_kind=result.input_kind,
                implementation_id=result.implementation_id,
                fallback_reason=fallback_reason,
            )
        self._thread_local_storage.preprocessor_selection = {
            "requested_id": plan.preprocessor_id,
            "effective_id": result.implementation_id,
            "fallback_reason": fallback_reason,
            "input_kind": result.input_kind,
        }
        if self._stream_pipeline_enabled:
            tensor._pre_processing_meta = result.metadata

        return tensor, result.metadata

    def _await_preprocessing(self, tensor):
        # Called under the engine lock by both forward entry points.
        readiness = self._preprocess_readiness.consume(tensor)
        if readiness is not None and readiness.ready_event is not None:
            self._inference_stream.wait_event(readiness.ready_event)
        tensor.record_stream(self._inference_stream)

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        with self._lock:
            self._await_preprocessing(pre_processed_images)
            with use_cuda_context(context=self._cuda_context):
                detections, labels, masks = infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                    stream=self._inference_stream,
                    trt_cuda_graph_cache=cache,
                )
                return detections, labels, masks

    def forward_async(
        self,
        pre_processed_images: torch.Tensor,
        pre_processing_meta,
        **kwargs,
    ):
        """Submit CUDA-graph inference without waiting for completion."""
        if self._trt_cuda_graph_cache is None:
            return super().forward_async(
                pre_processed_images, pre_processing_meta, **kwargs
            )

        with self._lock:
            self._await_preprocessing(pre_processed_images)
            with use_cuda_context(context=self._cuda_context):
                raw = infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                    stream=self._inference_stream,
                    trt_cuda_graph_cache=self._trt_cuda_graph_cache,
                    synchronize=False,
                )
        graph_state = getattr(raw[0], "_trt_graph_state", None)
        if graph_state is None:
            # Dynamic TensorRT execution does not expose graph-owned output
            # buffers, so the future must wait for inference completion before
            # handing outputs to postprocess.
            self._inference_stream.synchronize()
            return _DirectInferenceFuture(self, raw, pre_processing_meta, None, kwargs)
        produce_event = getattr(raw[0], "_trt_produce_event", None)
        if kwargs.get("reuse_trt_graph_outputs", False):
            # The stream pipeline schedules postprocess before launching the
            # next graph replay. That ordering lets postprocess read TensorRT's
            # graph output buffers directly and avoids the DtoD clone below.
            future_kwargs = dict(kwargs)
            future_kwargs["defer_postprocess_sync"] = True
            return _DirectInferenceFuture(
                self, raw, pre_processing_meta, produce_event, future_kwargs
            )

        stream = graph_state.cuda_stream

        tls = self._thread_local_storage
        clone_sets = getattr(tls, "clone_sets", None)
        if clone_sets is None:
            raw0, raw1, raw2 = raw
            clone_sets = [
                (
                    torch.empty_like(raw0),
                    torch.empty_like(raw1),
                    torch.empty_like(raw2),
                )
                for _ in range(3)
            ]
            tls.clone_sets = clone_sets
            tls.clone_idx = 0
        idx = tls.clone_idx
        clones = clone_sets[idx]
        tls.clone_idx = (idx + 1) % len(clone_sets)

        prev_stream = torch.cuda.current_stream(self._device)
        torch.cuda.set_stream(stream)
        try:
            raw0, raw1, raw2 = raw
            # Non-pipelined async callers may launch the next graph replay
            # before this future is consumed. Clone into a small ring so the
            # next TensorRT run can safely reuse its graph output buffers.
            clones[0].copy_(raw0, non_blocking=True)
            clones[1].copy_(raw1, non_blocking=True)
            clones[2].copy_(raw2, non_blocking=True)
            produce_event = torch.cuda.Event()
            produce_event.record(stream)
            consumer_done = graph_state.consumer_done_event
            if consumer_done is None:
                consumer_done = torch.cuda.Event()
                graph_state.consumer_done_event = consumer_done
            consumer_done.record(stream)
        finally:
            torch.cuda.set_stream(prev_stream)

        clones[0]._trt_produce_event = produce_event  # type: ignore[attr-defined]
        future_kwargs = dict(kwargs)
        future_kwargs["defer_postprocess_sync"] = True
        return _DirectInferenceFuture(
            self, clones, pre_processing_meta, produce_event, future_kwargs
        )

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        mask_format: InstanceSegmentationMaskFormat = "dense",
        max_detections: Optional[int] = INFERENCE_MODELS_RFDETR_DEFAULT_MAX_DETECTIONS,
        masks_resolution_factor: float = 1.0,
        **kwargs,
    ) -> List[InstanceDetections]:
        """Post-process detections into the requested mask representation.

        Args:
            model_results: Raw backend detections and native mask outputs.
            pre_processing_meta: Per-image preprocessing geometry.
            confidence: Detection confidence threshold or model-default selector.
            mask_format: Dense masks, RLE masks, or compact native-score ``grid`` masks.
            max_detections: Maximum retained detections per image.
            masks_resolution_factor: Dense/RLE interpolation factor; grid output requires 1.0.
            **kwargs: Additional backend options.

        Returns:
            Instance detections; grid masks retain native scores while boxes
            and outer size fields describe the original image.

        Raises:
            ModelInputError: For unsupported formats or a grid resolution factor
                other than 1.0.
        """
        if mask_format not in self.supported_mask_formats:
            raise ModelInputError(
                message=f"RFDetr Instance Segmentation models support the following mask "
                f"formats: {self.supported_mask_formats}. Requested format: {mask_format} "
                f"is not supported. If you see this error while running on Roboflow platform, "
                f"contact support or raise an issue at https://github.com/roboflow/inference/issues. "
                f"When running locally - please verify your integration to make sure that appropriate "
                f"value of `mask_format` parameter is set.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        if mask_format == "grid" and masks_resolution_factor != 1.0:
            raise ModelInputError(
                message="mask_format='grid' retains native scores; masks_resolution_factor "
                "must be 1.0. Apply the resolution factor during mask materialization.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )

        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        produce_event = getattr(model_results[0], "_trt_produce_event", None)
        graph_state = getattr(model_results[0], "_trt_graph_state", None)
        with torch.cuda.stream(self._post_process_stream):
            if produce_event is not None:
                self._post_process_stream.wait_event(produce_event)
            for result_element in model_results:
                result_element.record_stream(self._post_process_stream)
            bboxes, logits, masks = model_results
            if mask_format == "dense":
                results = post_process_instance_segmentation_results(
                    bboxes=bboxes,
                    logits=logits,
                    masks=masks,
                    pre_processing_meta=pre_processing_meta,
                    threshold=confidence_filter.get_threshold(self.class_names),
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                    max_detections=max_detections,
                    masks_resolution_factor=masks_resolution_factor,
                )
            elif mask_format == "grid":
                results = post_process_instance_segmentation_results_to_grid(
                    bboxes=bboxes,
                    logits=logits,
                    masks=masks,
                    pre_processing_meta=pre_processing_meta,
                    threshold=confidence_filter.get_threshold(self.class_names),
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                    max_detections=max_detections,
                )
            else:
                results = post_process_instance_segmentation_results_to_rle_masks(
                    bboxes=bboxes,
                    logits=logits,
                    masks=masks,
                    pre_processing_meta=pre_processing_meta,
                    threshold=confidence_filter.get_threshold(self.class_names),
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                    max_detections=max_detections,
                    defer_postprocess_sync=kwargs.get("defer_postprocess_sync", False),
                    masks_resolution_factor=masks_resolution_factor,
                )
            if graph_state is not None:
                output_consumed_events = [
                    get_trt_outputs_consumed_event(result) for result in results
                ]
                if output_consumed_events and all(
                    event is not None for event in output_consumed_events
                ):
                    graph_state.consumer_done_event = output_consumed_events[-1]
                else:
                    consumer_done = graph_state.consumer_done_event
                    if consumer_done is None:
                        consumer_done = torch.cuda.Event()
                        graph_state.consumer_done_event = consumer_done
                    consumer_done.record(self._post_process_stream)
        should_sync = True
        if kwargs.get("defer_postprocess_sync", False):
            should_sync = not all(
                get_deferred_postprocess_done_event(result) is not None
                for result in results
            )
        if should_sync:
            self._post_process_stream.synchronize()
        return results

    @property
    def _pre_process_stream(self) -> torch.cuda.Stream:
        if self._stream_pipeline_enabled:
            return self._pre_process_cuda_stream
        if not hasattr(self._thread_local_storage, "pre_process_stream"):
            self._thread_local_storage.pre_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.pre_process_stream

    @property
    def _post_process_stream(self) -> torch.cuda.Stream:
        if self._stream_pipeline_enabled:
            return self._post_process_cuda_stream
        if not hasattr(self._thread_local_storage, "post_process_stream"):
            self._thread_local_storage.post_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.post_process_stream
