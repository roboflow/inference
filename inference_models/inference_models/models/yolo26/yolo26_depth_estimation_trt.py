import threading
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch

from inference_models import ColorFormat
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.logger import LOGGER
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.base.depth_estimation import DepthEstimationModel
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
    parse_inference_config,
    parse_trt_config,
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
from inference_models.models.optimization.registry import ImplementationSelection
from inference_models.models.optimization.runtime_components import (
    get_runtime_components,
)
from inference_models.models.yolo26.optimization.execution_plan import (
    YOLO26DepthExecutionPlan,
)
from inference_models.models.yolo26.optimization.postprocessors import (
    BaseYOLO26DepthPostprocessor,
    ExactFusedTritonAAYOLO26DepthPostprocessor,
    ExactTritonAAYOLO26DepthPostprocessor,
    TritonAAYOLO26DepthPostprocessor,
    build_yolo26_depth_implementation_registry,
)
from inference_models.models.yolo26.optimization.preprocessors import (
    BaseYOLO26DepthPreprocessor,
    OpenCVFixedMap5xPinnedFusedConvertYOLO26DepthPreprocessor,
    TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor,
    TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor,
)
from inference_models.models.yolo26.optimization.schedulers import (
    BaseYOLO26DepthExecutionScheduler,
    CUDAEventHandoffYOLO26DepthExecutionScheduler,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLO26 model with TRT backend on GPU requires pycuda installation, which is brought with "
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
        message="Running YOLO26 model with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support.",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLO26ForDepthEstimationTRT(
    DepthEstimationModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
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
        execution_plan: Optional[YOLO26DepthExecutionPlan] = None,
        preprocessor_implementation_id: Optional[str] = None,
        scheduler_implementation_id: Optional[str] = None,
        postprocessor_implementation_id: Optional[str] = None,
        allow_compatibility_fallback: bool = True,
        **kwargs,
    ) -> "YOLO26ForDepthEstimationTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "inference_config.json",
                "trt_config.json",
                "engine.plan",
            ],
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
                    "YOLO26 Depth Estimation model running with TRT backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
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
        resolved_execution_plan = YOLO26DepthExecutionPlan.resolve(
            execution_plan=execution_plan,
            preprocessor_id=preprocessor_implementation_id,
            scheduler_id=scheduler_implementation_id,
            postprocessor_id=postprocessor_implementation_id,
            allow_compatibility_fallback=allow_compatibility_fallback,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_name=outputs[0],
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
            recommended_parameters=recommended_parameters,
            execution_plan=resolved_execution_plan,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_name: str,
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        recommended_parameters: Optional[RecommendedParameters] = None,
        execution_plan: Optional[YOLO26DepthExecutionPlan] = None,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = [output_name]
        self._inference_config = inference_config
        self._trt_config = trt_config
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters
        self._execution_plan = YOLO26DepthExecutionPlan.resolve(
            execution_plan=execution_plan,
        )
        self._optimization_context = ExecutionContext(
            device_kind="gpu",
            device=str(self._device),
            compute_capability=torch.cuda.get_device_capability(self._device),
            runtime_components=get_runtime_components(),
        )
        self._implementation_registry = build_yolo26_depth_implementation_registry(
            device=self._device,
        )
        self._preprocessor_selection = cast(
            ImplementationSelection[
                Union[
                    BaseYOLO26DepthPreprocessor,
                    OpenCVFixedMap5xPinnedFusedConvertYOLO26DepthPreprocessor,
                    TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor,
                    TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor,
                ]
            ],
            self._implementation_registry.resolve_selection(
                stage=OptimizationStage.PREPROCESS,
                requested_id=self._execution_plan.preprocessor_id,
                context=self._optimization_context,
                allow_fallback=self._execution_plan.allow_compatibility_fallback,
            ),
        )
        self._scheduler_selection = cast(
            ImplementationSelection[
                Union[
                    BaseYOLO26DepthExecutionScheduler,
                    CUDAEventHandoffYOLO26DepthExecutionScheduler,
                ]
            ],
            self._implementation_registry.resolve_selection(
                stage=OptimizationStage.SCHEDULER,
                requested_id=self._execution_plan.scheduler_id,
                context=self._optimization_context,
                allow_fallback=self._execution_plan.allow_compatibility_fallback,
            ),
        )
        self._postprocessor_selection = cast(
            ImplementationSelection[
                Union[
                    BaseYOLO26DepthPostprocessor,
                    ExactFusedTritonAAYOLO26DepthPostprocessor,
                    ExactTritonAAYOLO26DepthPostprocessor,
                    TritonAAYOLO26DepthPostprocessor,
                ]
            ],
            self._implementation_registry.resolve_selection(
                stage=OptimizationStage.POSTPROCESS,
                requested_id=self._execution_plan.postprocessor_id,
                context=self._optimization_context,
                allow_fallback=self._execution_plan.allow_compatibility_fallback,
            ),
        )
        LOGGER.info(
            "YOLO26 depth preprocessor selection: %s",
            self._preprocessor_selection.to_dict(),
        )
        LOGGER.info(
            "YOLO26 depth scheduler selection: %s",
            self._scheduler_selection.to_dict(),
        )
        LOGGER.info(
            "YOLO26 depth postprocessor selection: %s",
            self._postprocessor_selection.to_dict(),
        )

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[torch.Tensor]:
        """Run composed inference with scheduler-managed preprocessing readiness."""
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
        pre_process_stream = (
            self._scheduler_selection.implementation.preprocess_stream()
        )
        effective_id = self._preprocessor_selection.effective_id
        with torch.cuda.nvtx.range(
            f"yolo26-depth.preprocess[phase=submit,effective={effective_id}]"
        ):
            with torch.cuda.stream(pre_process_stream):
                context = ExecutionContext(
                    device_kind=self._optimization_context.device_kind,
                    device=self._optimization_context.device,
                    current_stream=pre_process_stream,
                    compute_capability=self._optimization_context.compute_capability,
                    runtime_components=self._optimization_context.runtime_components,
                )
                pre_processed_images, pre_processing_meta = (
                    self._preprocessor_selection.implementation.preprocess(
                        images=images,
                        image_pre_processing=(
                            self._inference_config.image_pre_processing
                        ),
                        network_input=self._inference_config.network_input,
                        target_device=self._device,
                        input_color_format=input_color_format,
                        pre_processing_overrides=pre_processing_overrides,
                        context=context,
                    )
                )
        pre_processed_images = (
            self._scheduler_selection.implementation.finalize_preprocess(
                pre_processed_images,
                context=context,
                independent_stage_execution=independent_stage_execution,
            )
        )

        return pre_processed_images, pre_processing_meta

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None

        def execute_engine(stream: torch.cuda.Stream) -> torch.Tensor:
            with use_cuda_context(context=self._cuda_context):
                return infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                    stream=stream,
                    trt_cuda_graph_cache=cache,
                )[0]

        return self._scheduler_selection.implementation.execute_engine(
            pre_processed_images,
            operation=execute_engine,
        )

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        **kwargs,
    ) -> List[torch.Tensor]:
        post_process_stream = self._post_process_stream
        effective_id = self._postprocessor_selection.effective_id
        with torch.cuda.nvtx.range(
            f"yolo26-depth.postprocess[phase=submit,effective={effective_id}]"
        ):
            with torch.cuda.stream(post_process_stream):
                model_results.record_stream(post_process_stream)
                context = ExecutionContext(
                    device_kind=self._optimization_context.device_kind,
                    device=self._optimization_context.device,
                    current_stream=post_process_stream,
                    compute_capability=self._optimization_context.compute_capability,
                    runtime_components=self._optimization_context.runtime_components,
                )
                results = self._postprocessor_selection.implementation.postprocess(
                    model_results=model_results,
                    pre_processing_meta=pre_processing_meta,
                    context=context,
                )
        with torch.cuda.nvtx.range(
            f"yolo26-depth.postprocess[phase=synchronize,effective={effective_id}]"
        ):
            post_process_stream.synchronize()
        return results

    @property
    def execution_plan_metadata(self) -> Dict[str, Any]:
        """Return requested and effective inference-path implementation IDs.

        Returns:
            JSON-compatible execution-plan and postprocessor selection metadata.
        """
        metadata = {
            "requested_plan": self._execution_plan.to_dict(),
            "preprocessor": self._preprocessor_selection.to_dict(),
            "preprocessor_metadata": (
                self._preprocessor_selection.implementation.metadata.to_dict()
            ),
            "scheduler": self._scheduler_selection.to_dict(),
            "scheduler_metadata": (
                self._scheduler_selection.implementation.metadata.to_dict()
            ),
            "postprocessor": self._postprocessor_selection.to_dict(),
            "postprocessor_metadata": (
                self._postprocessor_selection.implementation.metadata.to_dict()
            ),
        }

        return metadata

    @property
    def _post_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "post_process_stream"):
            self._thread_local_storage.post_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.post_process_stream
