from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import ColorFormat, SemanticSegmentationModel
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.developer_tools import align_device_with_onnx_session
from inference_models.entities import Confidence
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
)
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import (
    create_onnx_inference_session,
    get_onnx_static_input_spatial_size,
    run_onnx_session_with_batch_size_limit,
    set_onnx_execution_provider_defaults,
)
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    align_training_input_size_with_model,
    ensure_input_size_within_limit,
    parse_class_names_file,
)
from inference_models.models.common.roboflow.semantic_segmentation import (
    resolve_background_class_id,
    validate_class_names,
)
from inference_models.models.common.streams import get_cuda_stream, use_cuda_stream
from inference_models.models.rfdetr.pre_processing import pre_process_network_input
from inference_models.models.rfdetr.semantic_segmentation_common import (
    parse_semantic_segmentation_inference_config,
    post_process_semantic_segmentation_results,
)
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running RFDetr Semantic Segmentation model with ONNX backend requires `onnxruntime` installation, "
        "which is brought with `onnx-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class RFDetrForSemanticSegmentationOnnx(
    SemanticSegmentationModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "RFDetrForSemanticSegmentationOnnx":
        """Load an RF-DETR semantic segmentation model package for the ONNX backend.

        Args:
            model_name_or_path: Directory of the model package.
            onnx_execution_providers: ONNX Runtime execution providers; the
                environment's selection when ``None``.
            default_onnx_trt_options: Apply the default TensorRT provider options.
            device: Device to run the model on.
            rf_detr_max_input_resolution: Largest input size the environment accepts.
            recommended_parameters: Thresholds recommended for the model.
            **kwargs: Ignored loader arguments.

        Returns:
            The loaded model.

        Raises:
            EnvironmentConfigurationError: If no execution provider is selected.
        """
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
            default_onnx_coreml_options=True,
        )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.onnx",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        validate_class_names(class_names)
        background_class_id = resolve_background_class_id(class_names)
        inference_config = parse_semantic_segmentation_inference_config(
            model_package_content["inference_config.json"],
            backend_name="ONNX",
            max_allowed_input_size=rf_detr_max_input_resolution,
        )

        session = create_onnx_inference_session(
            model_path=model_package_content["weights.onnx"],
            providers=onnx_execution_providers,
        )
        model_input_size = get_onnx_static_input_spatial_size(session=session)
        if model_input_size is not None:
            inference_config = align_training_input_size_with_model(
                inference_config,
                model_input_height=model_input_size[0],
                model_input_width=model_input_size[1],
            )
            # The declared size was checked before building the session; the model's own size is what runs.
            ensure_input_size_within_limit(
                inference_config=inference_config,
                max_allowed_input_size=rf_detr_max_input_resolution,
            )
        device = align_device_with_onnx_session(session=session, device=device)
        input_batch_size = session.get_inputs()[0].shape[0]
        if isinstance(input_batch_size, str):
            input_batch_size = None
        return cls(
            session=session,
            input_name=session.get_inputs()[0].name,
            inference_config=inference_config,
            class_names=class_names,
            background_class_id=background_class_id,
            device=device,
            input_batch_size=input_batch_size,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        session: onnxruntime.InferenceSession,
        input_name: str,
        inference_config: InferenceConfig,
        class_names: List[str],
        background_class_id: int,
        device: torch.device,
        input_batch_size: Optional[int],
        recommended_parameters: Optional[RecommendedParameters] = None,
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._background_class_id = background_class_id
        self._device = device
        self._min_batch_size = input_batch_size
        self._max_batch_size = (
            input_batch_size
            if input_batch_size is not None
            else inference_config.forward_pass.max_dynamic_batch_size
        )
        self._session_thread_lock = Lock()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

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
            (logits,) = run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._min_batch_size,
                max_batch_size=self._max_batch_size,
                stream=self._inference_stream,
            )

        return logits

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        **kwargs,
    ) -> List[SemanticSegmentationResult]:
        post_process_stream = self._post_process_stream
        with use_cuda_stream(post_process_stream):
            if post_process_stream is not None:
                model_results.record_stream(post_process_stream)
            results = post_process_semantic_segmentation_results(
                model_results,
                pre_processing_meta,
                class_names=self._class_names,
                background_class_id=self._background_class_id,
                device=self._device,
                confidence=confidence,
                recommended_parameters=self.recommended_parameters,
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
