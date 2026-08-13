from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch

from inference_models import ColorFormat
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.developer_tools import align_device_with_onnx_session
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.logger import LOGGER
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.base.depth_estimation import DepthEstimationModel
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import set_onnx_execution_provider_defaults
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import (
    post_process_depth_estimation_map,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.common.streams import get_cuda_stream, use_cuda_stream
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    OptimizationMetadata,
    OptimizationStage,
)
from inference_models.models.optimization.ids import BASE_IMPLEMENTATION_ID
from inference_models.models.yolo26.optimization.catalog import (
    build_yolo26_depth_onnx_scheduler_registry,
)
from inference_models.models.yolo26.optimization.contracts import (
    OnnxExecutionScheduler,
)
from inference_models.models.yolo26.optimization.execution_plan import (
    YOLO26DepthOnnxExecutionPlan,
)
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLO26 Depth Estimation model with ONNX backend requires `onnxruntime` installation, which is brought with "
        "`onnx-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLO26ForDepthEstimationOnnx(
    DepthEstimationModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        yolo26_depth_onnx_execution_plan: Optional[
            Union[YOLO26DepthOnnxExecutionPlan, Dict[str, Any]]
        ] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "YOLO26ForDepthEstimationOnnx":
        requested_plan = YOLO26DepthOnnxExecutionPlan.resolve(
            execution_plan=yolo26_depth_onnx_execution_plan,
        )
        if onnx_execution_providers is None:
            onnx_execution_providers = get_selected_onnx_execution_providers()
        if not onnx_execution_providers:
            raise EnvironmentConfigurationError(
                message=f"Could not initialize model - selected backend is ONNX which requires execution provider to "
                f"be specified - explicitly in `from_pretrained(...)` method or via env variable "
                f"`ONNXRUNTIME_EXECUTION_PROVIDERS`. If you run model locally - adjust your setup, otherwise "
                f"contact the platform support.",
                help_url="https://inference-models.roboflow.com/errors/runtime-environment/#environmentconfigurationerror",
            )
        onnx_execution_providers = set_onnx_execution_provider_defaults(
            providers=onnx_execution_providers,
            model_package_path=model_name_or_path,
            device=device,
            default_onnx_trt_options=default_onnx_trt_options,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "inference_config.json",
                "weights.onnx",
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
                    "YOLO26 Depth Estimation model running with ONNX backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `letterbox` "
                    "resize mode with gray edges will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        session = onnxruntime.InferenceSession(
            path_or_bytes=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        device = align_device_with_onnx_session(session=session, device=device)
        input_batch_size = session.get_inputs()[0].shape[0]
        if isinstance(input_batch_size, str):
            input_batch_size = None
        input_name = session.get_inputs()[0].name
        return cls(
            session=session,
            input_name=input_name,
            inference_config=inference_config,
            device=device,
            input_batch_size=input_batch_size,
            yolo26_depth_onnx_execution_plan=requested_plan,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        inference_config: InferenceConfig,
        device: torch.device,
        input_batch_size: Optional[int],
        yolo26_depth_onnx_execution_plan: Optional[
            Union[YOLO26DepthOnnxExecutionPlan, Dict[str, Any]]
        ] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._device = device
        self._input_batch_size = input_batch_size
        self._session_thread_lock = Lock()
        self._candidate_request_lock = Lock()
        requested_plan = YOLO26DepthOnnxExecutionPlan.resolve(
            execution_plan=yolo26_depth_onnx_execution_plan,
        )
        self._validate_supported_plan_stages(requested_plan)
        self._implementation_registry = build_yolo26_depth_onnx_scheduler_registry(
            session=self._session,
            input_name=self._input_name,
            input_batch_size=self._input_batch_size,
            device=self._device,
        )
        scheduler_selection = self._implementation_registry.resolve_selection(
            stage=OptimizationStage.SCHEDULER,
            requested_id=requested_plan.scheduler_id,
            context=self._execution_context(),
            allow_fallback=requested_plan.allow_compatibility_fallback,
        )
        self._scheduler = cast(
            OnnxExecutionScheduler,
            scheduler_selection.implementation,
        )
        self._yolo26_depth_onnx_execution_plan = YOLO26DepthOnnxExecutionPlan(
            scheduler_id=self._scheduler.metadata.implementation_id,
            allow_compatibility_fallback=requested_plan.allow_compatibility_fallback,
        )
        self._scheduler_selection = scheduler_selection.to_dict()
        if scheduler_selection.used_fallback:
            LOGGER.warning(
                "YOLO26 depth ONNX scheduler fallback requested=%s effective=%s "
                "reason=%s",
                scheduler_selection.requested_id,
                scheduler_selection.effective_id,
                scheduler_selection.fallback_reason,
            )
        elif self.scheduler_implementation_id != BASE_IMPLEMENTATION_ID:
            LOGGER.info(
                "Selected YOLO26 depth ONNX scheduler implementation=%s",
                self.scheduler_implementation_id,
            )
        self.recommended_parameters = recommended_parameters

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[torch.Tensor]:
        """Run the selected scheduler and preserve independent returned tensors."""
        if self._scheduler.metadata.supports_concurrency:
            return super().infer(images=images, **kwargs)
        with self._candidate_request_lock:
            results = super().infer(images=images, **kwargs)
            return [result.clone() for result in results]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        pre_process_stream = self._pre_process_stream
        with use_cuda_stream(pre_process_stream):
            pre_processed_images, pre_processing_meta = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                image_size_wh=image_size,
                pre_processing_overrides=pre_processing_overrides,
            )
        if pre_process_stream is not None:
            pre_process_stream.synchronize()
        return pre_processed_images, pre_processing_meta

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with self._session_thread_lock:
            return self._scheduler.execute(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        **kwargs,
    ) -> List[torch.Tensor]:
        post_process_stream = self._post_process_stream
        with use_cuda_stream(post_process_stream):
            if post_process_stream is not None:
                model_results.record_stream(post_process_stream)
            results = post_process_depth_estimation_map(
                model_results=model_results,
                pre_processing_meta=pre_processing_meta,
                device=self._device,
            )
        if post_process_stream is not None:
            post_process_stream.synchronize()
        return results

    @property
    def _pre_process_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="pre-processing")

    @property
    def _post_process_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="post-processing")

    @property
    def _inference_stream(self) -> Optional[torch.cuda.Stream]:
        return get_cuda_stream(device=self._device, purpose="inference")

    @property
    def scheduler_implementation_id(self) -> str:
        """Return the scheduler that will actually execute."""
        return self._scheduler.metadata.implementation_id

    @property
    def scheduler_implementation_metadata(self) -> OptimizationMetadata:
        """Return typed metadata for the selected scheduler."""
        return self._scheduler.metadata

    @property
    def yolo26_depth_onnx_execution_plan(self) -> YOLO26DepthOnnxExecutionPlan:
        """Return the effective composed execution plan."""
        return self._yolo26_depth_onnx_execution_plan

    @property
    def optimization_runtime_metadata(self) -> Dict[str, Any]:
        """Expose requested and effective IDs for profiling provenance."""
        return {
            "execution_plan": self.yolo26_depth_onnx_execution_plan.to_dict(),
            "scheduler": self.scheduler_implementation_metadata.to_dict(),
            "model_selection": {
                "scheduler": dict(self._scheduler_selection),
            },
        }

    def _execution_context(self) -> ExecutionContext:
        device_kind = "gpu" if self._device.type == "cuda" else "cpu"
        compute_capability = None
        if self._device.type == "cuda":
            compute_capability = torch.cuda.get_device_capability(self._device)
        return ExecutionContext(
            device_kind=device_kind,
            device=str(self._device),
            compute_capability=compute_capability,
            runtime_components={
                "onnxruntime": True,
                "torch": True,
            },
        )

    @staticmethod
    def _validate_supported_plan_stages(
        execution_plan: YOLO26DepthOnnxExecutionPlan,
    ) -> None:
        unsupported = {
            stage: implementation_id
            for stage, implementation_id in {
                "preprocessor": execution_plan.preprocessor_id,
                "buffer_strategy": execution_plan.buffer_strategy_id,
                "postprocessor": execution_plan.postprocessor_id,
                "engine_plugin": execution_plan.engine_plugin_id,
            }.items()
            if implementation_id != BASE_IMPLEMENTATION_ID
        }
        if unsupported:
            raise ModelRuntimeError(
                message=(
                    "YOLO26 depth ONNX currently exposes alternatives only for the "
                    f"scheduler stage; unsupported selections: {unsupported!r}."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )
