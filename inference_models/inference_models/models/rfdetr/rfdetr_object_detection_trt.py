import threading
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import numpy as np
import torch

from inference_models import Detections, ObjectDetectionModel, PreProcessingOverrides
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.logger import LOGGER
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
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    OptimizationMetadata,
    OptimizationStage,
)
from inference_models.models.optimization.fallback_warnings import (
    FallbackWarningTracker,
)
from inference_models.models.optimization.ids import BASE_IMPLEMENTATION_ID
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.optimization.catalog import (
    build_rfdetr_implementation_registry,
)
from inference_models.models.rfdetr.optimization.contracts import (
    Postprocessor,
    PostprocessRequest,
    PreprocessRequest,
)
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.readiness import (
    PreprocessReadinessTracker,
)
from inference_models.models.rfdetr.optimization.selection import (
    resolve_postprocessor_for_request,
    resolve_preprocessor_for_model,
    resolve_preprocessor_for_request,
)
from inference_models.models.rfdetr.pre_processing import (
    resolve_rfdetr_preprocessor_max_workers,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running RFDETR model with TRT backend on GPU requires pycuda installation, which is brought with "
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
        message="Running RFDETR with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support.",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class RFDetrForObjectDetectionTRT(
    (
        ObjectDetectionModel[
            torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
        ]
    )
):
    """Run RF-DETR object detection through TensorRT with selectable path stages."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache] = None,
        default_trt_cuda_graph_cache_size: int = 8,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        rfdetr_preprocessor_max_workers: Optional[int] = None,
        rfdetr_execution_plan: Optional[RFDetrExecutionPlan] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        independent_stage_execution: bool = False,
        **kwargs,
    ) -> "RFDetrForObjectDetectionTRT":
        """Load an RF-DETR TensorRT model package.

        Args:
            model_name_or_path: Local model package directory.
            device: CUDA device used for inference.
            engine_host_code_allowed: Whether TensorRT may execute engine host code.
            trt_cuda_graph_cache: Optional caller-managed CUDA graph cache.
            default_trt_cuda_graph_cache_size: Default automatic graph-cache capacity.
            rf_detr_max_input_resolution: Optional maximum accepted input resolution.
            rfdetr_preprocessor_max_workers: Explicit threaded preprocessing worker
                limit. When omitted, the corresponding environment value is used.
            rfdetr_execution_plan: Explicit composed execution plan. When omitted,
                RF-DETR implementation environment variables are used.
            recommended_parameters: Optional model-specific recommended parameters.
            independent_stage_execution: Whether public preprocessing must return a
                ready tensor that does not rely on model readiness state in forward.
            **kwargs: Additional loader arguments accepted for API compatibility.

        Returns:
            Loaded RF-DETR TensorRT model.

        Raises:
            ModelRuntimeError: If the target or implementation selection is invalid.
            CorruptedModelPackageError: If required package contents are inconsistent.
        """
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
                    "RFDetr Object Detection model running with TRT backend was trained with "
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
        if len(outputs) != 2:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 2 model outputs, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if "dets" not in outputs or "labels" not in outputs:
            raise CorruptedModelPackageError(
                message=f"Expected model outputs to be named `output0` and `output1`, but found: {outputs}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=["dets", "labels"],
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
            rfdetr_preprocessor_max_workers=rfdetr_preprocessor_max_workers,
            rfdetr_execution_plan=rfdetr_execution_plan,
            independent_stage_execution=independent_stage_execution,
            recommended_parameters=recommended_parameters,
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
        rfdetr_preprocessor_max_workers: Optional[int] = None,
        rfdetr_execution_plan: Optional[RFDetrExecutionPlan] = None,
        recommended_parameters=None,
        independent_stage_execution: bool = False,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = output_names
        self._inference_config = inference_config
        self._class_names = class_names
        self._classes_re_mapping = classes_re_mapping
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_config = trt_config
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._independent_stage_execution = independent_stage_execution
        self._rfdetr_preprocessor_max_workers = resolve_rfdetr_preprocessor_max_workers(
            max_workers=rfdetr_preprocessor_max_workers
        )
        requested_plan = RFDetrExecutionPlan.resolve(
            execution_plan=rfdetr_execution_plan,
        )
        self._implementation_registry = build_rfdetr_implementation_registry(
            device=self._device,
            preprocessor_max_workers=self._rfdetr_preprocessor_max_workers,
        )
        resolution_context = self._execution_stage_context(current_stream=None)
        preprocessor_selection = resolve_preprocessor_for_model(
            registry=self._implementation_registry,
            requested_id=requested_plan.preprocessor_id,
            context=resolution_context,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            allow_fallback=requested_plan.allow_compatibility_fallback,
        )
        self._preprocessor = preprocessor_selection.implementation
        self._preprocessor_model_selection = preprocessor_selection.to_dict()
        if preprocessor_selection.used_fallback:
            LOGGER.warning(
                "RF-DETR preprocessor fallback requested=%s effective=%s reason=%s",
                preprocessor_selection.requested_id,
                preprocessor_selection.effective_id,
                preprocessor_selection.fallback_reason,
            )
        self._postprocessor = cast(
            Postprocessor,
            self._implementation_registry.resolve(
                stage=OptimizationStage.POSTPROCESS,
                requested_id=requested_plan.postprocessor_id,
                context=resolution_context,
            ),
        )
        self._rfdetr_execution_plan = RFDetrExecutionPlan(
            preprocessor_id=self._preprocessor.metadata.implementation_id,
            buffer_strategy_id=requested_plan.buffer_strategy_id,
            scheduler_id=requested_plan.scheduler_id,
            postprocessor_id=self._postprocessor.metadata.implementation_id,
            engine_plugin_id=requested_plan.engine_plugin_id,
            allow_compatibility_fallback=(requested_plan.allow_compatibility_fallback),
        )
        self._lock = threading.Lock()
        self._request_fallback_warnings = FallbackWarningTracker()
        self._inference_stream = torch.cuda.Stream(device=self._device)
        self._preprocess_readiness = PreprocessReadinessTracker()
        if self.preprocessor_implementation_id != BASE_IMPLEMENTATION_ID:
            LOGGER.info(
                "Selected RF-DETR preprocessor implementation=%s",
                self.preprocessor_implementation_id,
            )
        if self.postprocessor_implementation_id != BASE_IMPLEMENTATION_ID:
            LOGGER.info(
                "Selected RF-DETR postprocessor implementation=%s",
                self.postprocessor_implementation_id,
            )
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def preprocessor_implementation_id(self) -> str:
        """Return the actually selected preprocessing implementation ID."""
        return self._preprocessor.metadata.implementation_id

    @property
    def preprocessor_implementation_metadata(self) -> OptimizationMetadata:
        """Return typed metadata for the selected preprocessor."""
        return self._preprocessor.metadata

    @property
    def postprocessor_implementation_id(self) -> str:
        """Return the actually selected postprocessing implementation ID."""
        return self._postprocessor.metadata.implementation_id

    @property
    def postprocessor_implementation_metadata(self) -> OptimizationMetadata:
        """Return typed metadata for the selected postprocessor."""
        return self._postprocessor.metadata

    @property
    def rfdetr_execution_plan(self) -> RFDetrExecutionPlan:
        """Return the resolved composed execution plan."""
        return self._rfdetr_execution_plan

    @property
    def optimization_runtime_metadata(self) -> Dict[str, Any]:
        """Return machine-readable selected implementation metadata."""
        metadata = {
            "execution_plan": self.rfdetr_execution_plan.to_dict(),
            "independent_stage_execution": self._independent_stage_execution,
            "preprocessor": self.preprocessor_implementation_metadata.to_dict(),
            "postprocessor": self.postprocessor_implementation_metadata.to_dict(),
            "model_selection": {
                "preprocessor": dict(self._preprocessor_model_selection),
            },
        }
        last_execution = {}
        for stage in ("preprocessor", "postprocessor"):
            selection = getattr(
                self._thread_local_storage,
                f"last_{stage}_selection",
                None,
            )
            if selection is not None:
                last_execution[stage] = dict(selection)
        if last_execution:
            metadata["last_execution"] = last_execution

        return metadata

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        """Preprocess inference inputs using the resolved execution plan.

        Args:
            images: Single image or image batch represented by arrays or tensors.
            input_color_format: Optional caller-supplied color format.
            pre_processing_overrides: Optional request preprocessing overrides.
            **kwargs: Additional request arguments accepted for API compatibility.

        Returns:
            Preprocessed tensor and per-image transformation metadata. In independent
            stage mode the tensor is ready before return; otherwise its readiness is
            tracked for this model instance's ``forward()`` method.

        Raises:
            ModelRuntimeError: If the selected implementation is incompatible.
        """
        stream = self._pre_process_stream
        request = PreprocessRequest(
            images=images,
            input_color_format=input_color_format,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            pre_processing_overrides=pre_processing_overrides,
        )
        context = self._execution_stage_context(
            current_stream=stream,
            resolved_axes=self._resolved_preprocess_axes(images),
        )
        selection = resolve_preprocessor_for_request(
            registry=self._implementation_registry,
            implementation=self._preprocessor,
            request=request,
            context=context,
            allow_fallback=self._rfdetr_execution_plan.allow_compatibility_fallback,
        )
        self._thread_local_storage.last_preprocessor_selection = selection.to_dict()
        if selection.used_fallback and self._request_fallback_warnings.claim(
            stage=OptimizationStage.PREPROCESS,
            requested_id=selection.requested_id,
            effective_id=selection.effective_id,
            reason=selection.fallback_reason,
        ):
            LOGGER.warning(
                "RF-DETR request preprocessor fallback requested=%s effective=%s "
                "reason=%s",
                selection.requested_id,
                selection.effective_id,
                selection.fallback_reason,
            )
        result = selection.implementation.preprocess(
            request=request,
            context=context,
        )
        if selection.fallback_reason is not None:
            result = replace(result, fallback_reason=selection.fallback_reason)
        if result.ready_event is None:
            stream.synchronize()
        elif self._independent_stage_execution:
            result.ready_event.synchronize()

        if not self._independent_stage_execution:
            self._preprocess_readiness.record(
                result.tensor,
                ready_event=result.ready_event,
                input_kind=result.input_kind,
                implementation_id=result.implementation_id,
                fallback_reason=result.fallback_reason,
            )

        return result.tensor, result.metadata

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the protected TensorRT model forward pass.

        Args:
            pre_processed_images: Ready tensor, or the exact tensor returned by this
                model's tracked asynchronous preprocessing path.
            disable_cuda_graphs: Whether to bypass the configured graph cache.
            **kwargs: Additional request arguments accepted for API compatibility.

        Returns:
            TensorRT detection boxes and logits.
        """
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                if not self._independent_stage_execution:
                    readiness = self._preprocess_readiness.consume(pre_processed_images)
                    if readiness is not None and readiness.ready_event is not None:
                        self._inference_stream.wait_event(readiness.ready_event)
                detections, labels = infer_from_trt_engine(
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
                return detections, labels

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        **kwargs,
    ) -> List[Detections]:
        """Postprocess TensorRT outputs using the resolved execution plan.

        Args:
            model_results: TensorRT detection boxes and logits.
            pre_processing_meta: Per-image preprocessing transformations.
            confidence: Global or class-specific confidence threshold selection.
            **kwargs: Additional request arguments accepted for API compatibility.

        Returns:
            Per-image object detections.

        Raises:
            ModelRuntimeError: If the selected implementation is incompatible.
        """
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        threshold = confidence_filter.get_threshold(self.class_names)
        stream = self._post_process_stream
        with torch.cuda.stream(stream):
            for result_element in model_results:
                result_element.record_stream(stream)
            bboxes, logits = model_results
            request = PostprocessRequest(
                bboxes=bboxes,
                logits=logits,
                pre_processing_meta=pre_processing_meta,
                threshold=threshold,
                num_classes=len(self.class_names),
                classes_re_mapping=self._classes_re_mapping,
            )
            context = self._execution_stage_context(
                current_stream=stream,
                resolved_axes={
                    "batch": int(logits.shape[0]),
                    "queries": int(logits.shape[1]),
                    "classes": int(logits.shape[2]),
                },
            )
            selection = resolve_postprocessor_for_request(
                registry=self._implementation_registry,
                implementation=self._postprocessor,
                request=request,
                context=context,
                allow_fallback=(
                    self._rfdetr_execution_plan.allow_compatibility_fallback
                ),
            )
            self._thread_local_storage.last_postprocessor_selection = (
                selection.to_dict()
            )
            if selection.used_fallback and self._request_fallback_warnings.claim(
                stage=OptimizationStage.POSTPROCESS,
                requested_id=selection.requested_id,
                effective_id=selection.effective_id,
                reason=selection.fallback_reason,
            ):
                LOGGER.warning(
                    "RF-DETR request postprocessor fallback requested=%s "
                    "effective=%s reason=%s",
                    selection.requested_id,
                    selection.effective_id,
                    selection.fallback_reason,
                )
            results = selection.implementation.postprocess(
                request=request,
                context=context,
            )
        stream.synchronize()

        return results

    def _execution_stage_context(
        self,
        *,
        current_stream: Optional[torch.cuda.Stream],
        resolved_axes: Optional[Mapping[str, Any]] = None,
    ) -> ExecutionContext:
        device_index = self._device.index or 0
        context = ExecutionContext(
            device_kind="gpu",
            device=str(self._device),
            device_name=torch.cuda.get_device_name(device_index),
            machine_type="runtime",
            scenario="runtime",
            resolved_axes=resolved_axes or {},
            current_stream=current_stream,
            compute_capability=torch.cuda.get_device_capability(device_index),
        )

        return context

    @staticmethod
    def _resolved_preprocess_axes(
        images: Union[
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
        ],
    ) -> Dict[str, Any]:
        if isinstance(images, list):
            batch_size = len(images)
            first = images[0] if images else None
        elif isinstance(images, (torch.Tensor, np.ndarray)) and images.ndim == 4:
            batch_size = int(images.shape[0])
            first = images[0] if batch_size else None
        else:
            batch_size = 1
            first = images
        shape = tuple(first.shape) if first is not None else ()
        axes = {
            "batch": batch_size,
            "input_type": type(first).__name__ if first is not None else "empty",
            "source_shape": shape,
        }

        return axes

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
