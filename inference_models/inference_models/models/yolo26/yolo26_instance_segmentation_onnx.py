from threading import Lock
from typing import List, Optional, Set, Tuple, Union

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
    INFERENCE_MODELS_YOLO26_DEFAULT_CONFIDENCE,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    EnvironmentConfigurationError,
    MissingDependencyError,
    ModelInputError,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.onnx import (
    run_onnx_session_with_batch_size_limit,
    set_onnx_execution_provider_defaults,
)
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    post_process_nms_fused_model_output,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.models.yolo26.common import prepare_dense_masks, prepare_rle_masks
from inference_models.utils.onnx_introspection import (
    get_selected_onnx_execution_providers,
)
from inference_models.weights_providers.entities import RecommendedParameters

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running YOLO26 model with ONNX backend requires pycuda installation, which is brought with "
        "`onnx-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class YOLO26ForInstanceSegmentationOnnx(
    InstanceSegmentationModel[
        torch.Tensor, PreProcessingMetadata, Tuple[torch.Tensor, torch.Tensor]
    ]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "YOLO26ForInstanceSegmentationOnnx":
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
                "class_names.txt",
                "inference_config.json",
                "weights.onnx",
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
                    "YOLO26 Instance Segmentation model running with ONNX backend was trained with "
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
        input_batch_size = session.get_inputs()[0].shape[0]
        if isinstance(input_batch_size, str):
            input_batch_size = None
        input_name = session.get_inputs()[0].name
        return cls(
            session=session,
            input_name=input_name,
            class_names=class_names,
            inference_config=inference_config,
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
        device: torch.device,
        input_batch_size: Optional[int],
        recommended_parameters=None,
    ):
        self._session = session
        self._input_name = input_name
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device
        self._input_batch_size = input_batch_size
        self._session_thread_lock = Lock()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def supported_mask_formats(self) -> Set[InstanceSegmentationMaskFormat]:
        return {"dense", "rle"}

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
            pre_processing_overrides=pre_processing_overrides,
        )

    def forward(
        self, pre_processed_images: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._session_thread_lock:
            instances, protos = run_onnx_session_with_batch_size_limit(
                session=self._session,
                inputs={self._input_name: pre_processed_images},
                min_batch_size=self._input_batch_size,
                max_batch_size=self._input_batch_size,
            )
            return instances, protos

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        mask_format: InstanceSegmentationMaskFormat = "dense",
        **kwargs,
    ) -> List[InstanceDetections]:
        if mask_format not in self.supported_mask_formats:
            raise ModelInputError(
                message=f"YOLO26 Instance Segmentation models support the following mask "
                f"formats: {self.supported_mask_formats}. Requested format: {mask_format} "
                f"is not supported. If you see this error while running on Roboflow platform, "
                f"contact support or raise an issue at https://github.com/roboflow/inference/issues. "
                f"When running locally - please verify your integration to make sure that appropriate "
                f"value of `mask_format` parameter is set.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_YOLO26_DEFAULT_CONFIDENCE,
        )
        confidence = confidence_filter.get_threshold(self.class_names)
        instances, protos = model_results
        filtered_results = post_process_nms_fused_model_output(
            output=instances, conf_thresh=confidence
        )
        if mask_format == "dense":
            return prepare_dense_masks(
                filtered_results=filtered_results,
                protos=protos,
                pre_processing_meta=pre_processing_meta,
            )
        return prepare_rle_masks(
            filtered_results=filtered_results,
            protos=protos,
            pre_processing_meta=pre_processing_meta,
        )
