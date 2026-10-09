from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import ColorFormat, SemanticSegmentationModel
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import Confidence
from inference_models.errors import InvalidModelInitParameterError
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    parse_class_names_file,
)
from inference_models.models.common.roboflow.semantic_segmentation import (
    resolve_background_class_id,
    validate_class_names,
)
from inference_models.models.rfdetr.common import parse_model_type
from inference_models.models.rfdetr.pre_processing import pre_process_network_input
from inference_models.models.rfdetr.rfdetr_base_pytorch import (
    RFDETRSem2XLargeConfig,
    RFDETRSemLargeConfig,
    RFDETRSemMediumConfig,
    RFDETRSemNanoConfig,
    RFDETRSemSmallConfig,
    RFDETRSemXLargeConfig,
    build_model,
)
from inference_models.models.rfdetr.semantic_segmentation_common import (
    parse_semantic_segmentation_inference_config,
    post_process_semantic_segmentation_results,
)
from inference_models.models.rfdetr.semantic_segmentation_model import (
    RFDetrSemanticSegmentationNetwork,
)
from inference_models.weights_providers.entities import RecommendedParameters

CONFIG_FOR_MODEL_TYPE = {
    "rfdetr-sem-nano": RFDETRSemNanoConfig,
    "rfdetr-sem-small": RFDETRSemSmallConfig,
    "rfdetr-sem-medium": RFDETRSemMediumConfig,
    "rfdetr-sem-large": RFDETRSemLargeConfig,
    "rfdetr-sem-xlarge": RFDETRSemXLargeConfig,
    "rfdetr-sem-2xlarge": RFDETRSem2XLargeConfig,
}


class RFDetrForSemanticSegmentationTorch(
    SemanticSegmentationModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "RFDetrForSemanticSegmentationTorch":
        """Load an RF-DETR semantic segmentation model package for the Torch backend.

        Args:
            model_name_or_path: Directory of the model package.
            device: Device to run the model on.
            rf_detr_max_input_resolution: Largest input size the environment accepts.
            recommended_parameters: Thresholds recommended for the model.
            **kwargs: Ignored loader arguments.

        Returns:
            The loaded model.

        Raises:
            InvalidModelInitParameterError: If the package names an unknown model type.
        """
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "model_type.json",
                "weights.pth",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        validate_class_names(class_names)
        background_class_id = resolve_background_class_id(class_names)
        inference_config = parse_semantic_segmentation_inference_config(
            model_package_content["inference_config.json"],
            backend_name="Torch",
            max_allowed_input_size=rf_detr_max_input_resolution,
        )

        model_type = parse_model_type(
            config_path=model_package_content["model_type.json"]
        )
        if model_type not in CONFIG_FOR_MODEL_TYPE:
            raise InvalidModelInitParameterError(
                message=f"Model package describes model_type as '{model_type}' which is not supported. "
                f"Supported model types: {list(CONFIG_FOR_MODEL_TYPE.keys())}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
            )

        weights_dict = torch.load(
            model_package_content["weights.pth"],
            map_location=device,
            weights_only=False,
        )["model"]
        model_config = CONFIG_FOR_MODEL_TYPE[model_type](device=device)
        model_config.num_classes = weights_dict["semantic_embed.bias"].shape[0]
        model_config.resolution = (
            inference_config.network_input.training_input_size.height
        )
        model = build_model(config=model_config)
        model.load_state_dict(weights_dict)
        model = model.eval().to(device)
        return cls(
            model=model,
            inference_config=inference_config,
            class_names=class_names,
            background_class_id=background_class_id,
            device=device,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        model: RFDetrSemanticSegmentationNetwork,
        inference_config: InferenceConfig,
        class_names: List[str],
        background_class_id: int,
        device: torch.device,
        recommended_parameters: Optional[RecommendedParameters] = None,
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
        self._background_class_id = background_class_id
        self._device = device
        self._lock = Lock()
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
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
            pre_processing_overrides=pre_processing_overrides,
        )

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with self._lock, torch.inference_mode():
            logits = self._model(pre_processed_images)

        return logits

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        **kwargs,
    ) -> List[SemanticSegmentationResult]:
        results = post_process_semantic_segmentation_results(
            model_results,
            pre_processing_meta,
            class_names=self._class_names,
            background_class_id=self._background_class_id,
            device=self._device,
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
        )
        return results
