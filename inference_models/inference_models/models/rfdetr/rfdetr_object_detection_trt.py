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
    load_trt_model,
)
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    OptimizationMetadata,
    OptimizationStage,
)
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.optimization.fallback_warnings import (
    FallbackWarningTracker,
)
from inference_models.models.optimization.ids import BASE_IMPLEMENTATION_ID
from inference_models.models.optimization.runtime_components import (
    get_runtime_components,
)
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.optimization.catalog import (
    build_rfdetr_implementation_registry,
)
from inference_models.models.rfdetr.optimization.contracts import (
    BufferStrategy,
    EngineAdjacentPlugin,
    EngineExecutionRequest,
    ExecutionScheduler,
    Postprocessor,
    PostprocessRequest,
    PreprocessRequest,
)
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.selection import (
    resolve_postprocessor_for_request,
    resolve_preprocessor_for_model,
    resolve_preprocessor_for_request,
    resolve_preprocessor_runtime_fallback,
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
            trt_execution_context = engine.create_execution_context()
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
            trt_execution_context=trt_execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
            rfdetr_preprocessor_max_workers=rfdetr_preprocessor_max_workers,
            rfdetr_execution_plan=rfdetr_execution_plan,
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
        trt_execution_context: trt.IExecutionContext,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        rfdetr_preprocessor_max_workers: Optional[int] = None,
        rfdetr_execution_plan: Optional[RFDetrExecutionPlan] = None,
        recommended_parameters=None,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = output_names
        self._inference_config = inference_config
        self._class_names = class_names
        self._classes_re_mapping = classes_re_mapping
        self._device = device
        self._cuda_context = cuda_context
        self._trt_execution_context = trt_execution_context
        self._trt_config = trt_config
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
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
        postprocessor_selection = self._implementation_registry.resolve_selection(
            stage=OptimizationStage.POSTPROCESS,
            requested_id=requested_plan.postprocessor_id,
            context=resolution_context,
            allow_fallback=requested_plan.allow_compatibility_fallback,
        )
        self._postprocessor = cast(
            Postprocessor,
            postprocessor_selection.implementation,
        )
        buffer_strategy_selection = self._implementation_registry.resolve_selection(
            stage=OptimizationStage.BUFFER_STRATEGY,
            requested_id=requested_plan.buffer_strategy_id,
            context=resolution_context,
            allow_fallback=requested_plan.allow_compatibility_fallback,
        )
        self._buffer_strategy = cast(
            BufferStrategy,
            buffer_strategy_selection.implementation,
        )
        scheduler_selection = self._implementation_registry.resolve_selection(
            stage=OptimizationStage.SCHEDULER,
            requested_id=requested_plan.scheduler_id,
            context=resolution_context,
            allow_fallback=requested_plan.allow_compatibility_fallback,
        )
        self._scheduler = cast(
            ExecutionScheduler,
            scheduler_selection.implementation,
        )
        engine_plugin_selection = self._implementation_registry.resolve_selection(
            stage=OptimizationStage.ENGINE_PLUGIN,
            requested_id=requested_plan.engine_plugin_id,
            context=resolution_context,
            allow_fallback=requested_plan.allow_compatibility_fallback,
        )
        self._engine_plugin = cast(
            EngineAdjacentPlugin,
            engine_plugin_selection.implementation,
        )
        self._rfdetr_execution_plan = RFDetrExecutionPlan(
            preprocessor_id=self._preprocessor.metadata.implementation_id,
            buffer_strategy_id=self._buffer_strategy.metadata.implementation_id,
            scheduler_id=self._scheduler.metadata.implementation_id,
            postprocessor_id=self._postprocessor.metadata.implementation_id,
            engine_plugin_id=self._engine_plugin.metadata.implementation_id,
            allow_compatibility_fallback=(requested_plan.allow_compatibility_fallback),
            allow_runtime_failure_fallback=(
                requested_plan.allow_runtime_failure_fallback
            ),
        )
        model_selections = {
            "preprocessor": preprocessor_selection,
            "buffer_strategy": buffer_strategy_selection,
            "scheduler": scheduler_selection,
            "postprocessor": postprocessor_selection,
            "engine_plugin": engine_plugin_selection,
        }
        self._model_selections = {
            stage: selection.to_dict() for stage, selection in model_selections.items()
        }
        for stage, selection in model_selections.items():
            if selection.used_fallback:
                LOGGER.warning(
                    "RF-DETR %s fallback requested=%s effective=%s reason=%s",
                    stage,
                    selection.requested_id,
                    selection.effective_id,
                    selection.fallback_reason,
                )
        self._request_fallback_warnings = FallbackWarningTracker()
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
    def buffer_strategy_implementation_id(self) -> str:
        """Return the actually selected buffer-strategy implementation ID."""
        return self._buffer_strategy.metadata.implementation_id

    @property
    def buffer_strategy_implementation_metadata(self) -> OptimizationMetadata:
        """Return typed metadata for the selected buffer strategy."""
        return self._buffer_strategy.metadata

    @property
    def scheduler_implementation_id(self) -> str:
        """Return the actually selected scheduler implementation ID."""
        return self._scheduler.metadata.implementation_id

    @property
    def scheduler_implementation_metadata(self) -> OptimizationMetadata:
        """Return typed metadata for the selected scheduler."""
        return self._scheduler.metadata

    @property
    def postprocessor_implementation_id(self) -> str:
        """Return the actually selected postprocessing implementation ID."""
        return self._postprocessor.metadata.implementation_id

    @property
    def postprocessor_implementation_metadata(self) -> OptimizationMetadata:
        """Return typed metadata for the selected postprocessor."""
        return self._postprocessor.metadata

    @property
    def engine_plugin_implementation_id(self) -> str:
        """Return the actually selected engine-plugin implementation ID."""
        return self._engine_plugin.metadata.implementation_id

    @property
    def engine_plugin_implementation_metadata(self) -> OptimizationMetadata:
        """Return typed metadata for the selected engine plugin."""
        return self._engine_plugin.metadata

    @property
    def rfdetr_execution_plan(self) -> RFDetrExecutionPlan:
        """Return the resolved composed execution plan."""
        return self._rfdetr_execution_plan

    @property
    def optimization_runtime_metadata(self) -> Dict[str, Any]:
        """Return machine-readable selected implementation metadata."""
        metadata = {
            "execution_plan": self.rfdetr_execution_plan.to_dict(),
            "preprocessor": self.preprocessor_implementation_metadata.to_dict(),
            "buffer_strategy": (self.buffer_strategy_implementation_metadata.to_dict()),
            "scheduler": self.scheduler_implementation_metadata.to_dict(),
            "postprocessor": self.postprocessor_implementation_metadata.to_dict(),
            "engine_plugin": self.engine_plugin_implementation_metadata.to_dict(),
            "model_selection": {
                stage: dict(selection)
                for stage, selection in self._model_selections.items()
            },
        }
        last_execution = {}
        for stage in (
            "preprocessor",
            "buffer_strategy",
            "scheduler",
            "postprocessor",
            "engine_plugin",
        ):
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

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[Detections]:
        """Run composed inference with an asynchronous preprocessing handoff.

        Args:
            images: Single image or image batch represented by arrays or tensors.
            **kwargs: Inference arguments forwarded through all stages.

        Returns:
            Per-image object detections.
        """
        kwargs.pop("independent_stage_execution", None)
        pre_processed_images, pre_processing_meta = self.pre_process(
            images=images,
            independent_stage_execution=False,
            **kwargs,
        )
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, pre_processing_meta, **kwargs)

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        independent_stage_execution: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        """Preprocess inference inputs using the resolved execution plan.

        Args:
            images: Single image or image batch represented by arrays or tensors.
            input_color_format: Optional caller-supplied color format.
            pre_processing_overrides: Optional request preprocessing overrides.
            independent_stage_execution: Whether preprocessing must finish before
                returning. Composed ``infer()`` sets this to ``False`` and transfers
                readiness to ``forward()`` through the exact returned tensor.
            **kwargs: Additional request arguments accepted for API compatibility.

        Returns:
            Preprocessed tensor and per-image transformation metadata. By default the
            tensor is ready before return; otherwise its readiness is tracked for this
            model instance's ``forward()`` method.

        Raises:
            ModelRuntimeError: If the selected implementation is incompatible.
        """
        stream = self._scheduler.preprocess_stream()
        request = PreprocessRequest(
            images=images,
            input_color_format=input_color_format,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            pre_processing_overrides=pre_processing_overrides,
        )
        context = self._execution_stage_context(current_stream=stream)
        selection = resolve_preprocessor_for_request(
            registry=self._implementation_registry,
            implementation=self._preprocessor,
            request=request,
            context=context,
            allow_fallback=self._rfdetr_execution_plan.allow_compatibility_fallback,
        )

        def _as_model_runtime_error(
            error: RecoverableStageExecutionError,
        ) -> ModelRuntimeError:
            message = str(error.args[0]) if error.args else str(error)

            return ModelRuntimeError(message=message, help_url=error.help_url)

        try:
            selection = resolve_preprocessor_runtime_fallback(
                registry=self._implementation_registry,
                selection=selection,
                request=request,
                context=context,
                allow_fallback=(
                    self._rfdetr_execution_plan.allow_runtime_failure_fallback
                ),
            )
            try:
                result = selection.implementation.preprocess(
                    request=request,
                    context=context,
                )
            except RecoverableStageExecutionError:
                if not self._rfdetr_execution_plan.allow_runtime_failure_fallback:
                    raise
                fallback_selection = resolve_preprocessor_runtime_fallback(
                    registry=self._implementation_registry,
                    selection=selection,
                    request=request,
                    context=context,
                    allow_fallback=True,
                )
                if fallback_selection.implementation is selection.implementation:
                    raise
                selection = fallback_selection
                result = selection.implementation.preprocess(
                    request=request,
                    context=context,
                )
        except RecoverableStageExecutionError as error:
            raise _as_model_runtime_error(error) from error
        self._record_last_execution(
            stage="preprocessor",
            selection=selection.to_dict(),
        )
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
        if selection.fallback_reason is not None:
            result = replace(result, fallback_reason=selection.fallback_reason)
        engine_input_buffer = self._buffer_strategy.prepare_engine_input(
            result=result,
            context=context,
        )
        self._record_static_stage_execution(stage="buffer_strategy")
        self._record_static_stage_execution(stage="scheduler")
        pre_processed_images = self._scheduler.finalize_preprocess(
            engine_input_buffer,
            context=context,
            independent_stage_execution=independent_stage_execution,
        )

        return pre_processed_images, result.metadata

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

        def execute_engine(
            stream: torch.cuda.Stream,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            with use_cuda_context(context=self._cuda_context):
                context = self._execution_stage_context(current_stream=stream)
                request = EngineExecutionRequest(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    trt_execution_context=self._trt_execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    output_names=self._output_names,
                    trt_cuda_graph_cache=cache,
                )
                self._record_static_stage_execution(stage="engine_plugin")
                model_results = self._engine_plugin.execute(
                    request=request,
                    context=context,
                )

                return model_results

        self._record_static_stage_execution(stage="scheduler")
        model_results = self._scheduler.execute_engine(
            pre_processed_images,
            operation=execute_engine,
        )

        return model_results

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

        def execute_postprocess(stream: torch.cuda.Stream) -> List[Detections]:
            bboxes, logits = model_results
            request = PostprocessRequest(
                bboxes=bboxes,
                logits=logits,
                pre_processing_meta=pre_processing_meta,
                threshold=threshold,
                num_classes=len(self.class_names),
                classes_re_mapping=self._classes_re_mapping,
            )
            context = self._execution_stage_context(current_stream=stream)
            selection = resolve_postprocessor_for_request(
                registry=self._implementation_registry,
                implementation=self._postprocessor,
                request=request,
                context=context,
                allow_fallback=(
                    self._rfdetr_execution_plan.allow_compatibility_fallback
                ),
            )
            self._record_last_execution(
                stage="postprocessor",
                selection=selection.to_dict(),
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

            return results

        self._record_static_stage_execution(stage="scheduler")
        results = self._scheduler.execute_postprocess(
            model_results,
            operation=execute_postprocess,
        )

        return results

    def _execution_stage_context(
        self,
        *,
        current_stream: Optional[torch.cuda.Stream],
    ) -> ExecutionContext:
        context = ExecutionContext(
            device_kind="gpu",
            device=str(self._device),
            current_stream=current_stream,
            compute_capability=torch.cuda.get_device_capability(
                self._device.index or 0
            ),
            runtime_components=get_runtime_components(),
        )

        return context

    def _record_static_stage_execution(self, *, stage: str) -> None:
        self._record_last_execution(
            stage=stage,
            selection=self._model_selections[stage],
        )

    def _record_last_execution(
        self,
        *,
        stage: str,
        selection: Mapping[str, Optional[str]],
    ) -> None:
        setattr(
            self._thread_local_storage,
            f"last_{stage}_selection",
            dict(selection),
        )
