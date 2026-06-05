from threading import Lock
from typing import List, Optional, Tuple, Union

import segmentation_models_pytorch as smp
import torch

from inference_models import ColorFormat, SemanticSegmentationModel
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_DEEP_LAB_V3_PLUS_DEFAULT_CONFIDENCE,
)
from inference_models.entities import Confidence
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import PreprocessingMetadata
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
    resolve_background_class_id,
)
from inference_models.models.common.roboflow.post_processing import (
    post_process_semantic_segmentation_logits,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_models.weights_providers.entities import RecommendedParameters


class DeepLabV3PlusForSemanticSegmentationTorch(
    SemanticSegmentationModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "DeepLabV3PlusForSemanticSegmentationTorch":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.pt",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        background_class_id = resolve_background_class_id(class_names)
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
                ResizeMode.FIT_LONGER_EDGE,
            },
        )
        if inference_config.model_initialization is None:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameters not provided in inference config.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        num_classes = inference_config.model_initialization.get("classes")
        in_channels = inference_config.model_initialization.get("in_channels")
        encoder_name = inference_config.model_initialization.get("encoder_name")
        if not isinstance(num_classes, int) or num_classes < 1:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `num_classes` not provided or in invalid format.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if not isinstance(in_channels, int) or in_channels not in {1, 3}:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `in_channels` not provided or in invalid format.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if not isinstance(encoder_name, str):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `encoder_name` not provided or in invalid format.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        model = (
            smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=num_classes,
            )
            .to(device)
            .eval()
        )
        state_dict = torch.load(
            model_package_content["weights.pt"],
            weights_only=True,
            map_location=device,
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return cls(
            model=model.eval(),
            inference_config=inference_config,
            class_names=class_names,
            background_class_id=background_class_id,
            device=device,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        model: smp.DeepLabV3Plus,
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
        images: Union[torch.Tensor, List[torch.Tensor]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, PreprocessingMetadata]:
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
            return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        **kwargs,
    ) -> List[SemanticSegmentationResult]:
        return post_process_semantic_segmentation_logits(
            model_results=model_results,
            pre_processing_meta=pre_processing_meta,
            class_names=self._class_names,
            background_class_id=self._background_class_id,
            device=self._device,
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEEP_LAB_V3_PLUS_DEFAULT_CONFIDENCE,
        )
