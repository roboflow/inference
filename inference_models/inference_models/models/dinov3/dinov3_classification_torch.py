from typing import List, Optional, Tuple, Union

import numpy as np
import timm
import torch
from torch import nn

from inference_models import (
    ClassificationModel,
    ClassificationPrediction,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_DINOV3_DEFAULT_CONFIDENCE,
)
from inference_models.entities import ColorFormat
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)


class DinoV3Model(nn.Module):
    """DINOv3 model for classification using timm's EVA ViT backbone."""

    def __init__(
        self, num_classes: int, model_name: str = "vit_small_patch16_dinov3.lvd1689m"
    ):
        """
        Args:
            num_classes: Number of classes to classify
            model_name: Name of the backbone model from timm
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        # DinoV3 is implemented as a parameterization of EVA ViT in timm
        self.backbone: timm.models.Eva = timm.create_model(
            self.model_name, pretrained=False
        )
        self.backbone = self.backbone.eval()
        self.linear_layer = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using the CLS token (position 0)."""
        return self.backbone.forward_features(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(self.forward_embedding(x))


class DinoV3ForClassificationTorch(ClassificationModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "DinoV3ForClassificationTorch":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.pth",
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
        )

        if inference_config.model_initialization is None:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameters not provided in inference config.",
                help_url="https://todo",
            )
        num_classes = inference_config.model_initialization.get("num_classes")
        model_name = inference_config.model_initialization.get("model_name")
        if not isinstance(num_classes, int):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `num_classes` not provided or in invalid format.",
                help_url="https://todo",
            )
        if not isinstance(model_name, str):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `model_name` not provided or in invalid format.",
                help_url="https://todo",
            )

        if (
            not inference_config.post_processing
            or inference_config.post_processing.type != "softmax"
        ):
            raise CorruptedModelPackageError(
                message="Expected Softmax to be the post-processing",
                help_url="https://todo",
            )

        # Create model and load weights
        model = DinoV3Model(num_classes=num_classes, model_name=model_name)
        state_dict = torch.load(
            model_package_content["weights.pth"],
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model = model.to(device).eval()

        return cls(
            model=model,
            inference_config=inference_config,
            class_names=class_names,
            device=device,
        )

    def __init__(
        self,
        model: DinoV3Model,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
        )[0]

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.inference_mode():
            return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        **kwargs,
    ) -> ClassificationPrediction:
        if (
            self._inference_config.post_processing
            and self._inference_config.post_processing.fused
        ):
            confidence = model_results
        else:
            confidence = torch.nn.functional.softmax(model_results, dim=-1)
        return ClassificationPrediction(
            class_id=confidence.argmax(dim=-1),
            confidence=confidence,
        )


class DinoV3ForMultiLabelClassificationTorch(
    MultiLabelClassificationModel[torch.Tensor, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "DinoV3ForMultiLabelClassificationTorch":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "weights.pth",
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
        )

        if inference_config.model_initialization is None:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameters not provided in inference config.",
                help_url="https://todo",
            )
        num_classes = inference_config.model_initialization.get("num_classes")
        model_name = inference_config.model_initialization.get("model_name")
        if not isinstance(num_classes, int):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `num_classes` not provided or in invalid format.",
                help_url="https://todo",
            )
        if not isinstance(model_name, str):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `model_name` not provided or in invalid format.",
                help_url="https://todo",
            )

        if (
            inference_config.post_processing
            and inference_config.post_processing.type != "sigmoid"
        ):
            raise CorruptedModelPackageError(
                message="Expected Sigmoid to be the post-processing",
                help_url="https://todo",
            )

        # Create model and load weights
        model = DinoV3Model(num_classes=num_classes, model_name=model_name)
        state_dict = torch.load(
            model_package_content["weights.pth"],
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model = model.to(device).eval()

        return cls(
            model=model,
            inference_config=inference_config,
            class_names=class_names,
            device=device,
        )

    def __init__(
        self,
        model: DinoV3Model,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
        )[0]

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.inference_mode():
            return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        confidence: float = INFERENCE_MODELS_DINOV3_DEFAULT_CONFIDENCE,
        **kwargs,
    ) -> List[MultiLabelClassificationPrediction]:
        if (
            self._inference_config.post_processing
            and self._inference_config.post_processing.fused
        ):
            model_results = model_results
        else:
            model_results = torch.nn.functional.sigmoid(model_results)
        results = []
        for batch_element_confidence in model_results:
            predicted_classes = torch.argwhere(
                batch_element_confidence >= confidence
            ).squeeze(dim=-1)
            results.append(
                MultiLabelClassificationPrediction(
                    class_ids=predicted_classes,
                    confidence=batch_element_confidence,
                )
            )
        return results
