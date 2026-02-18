from threading import Lock
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
    INFERENCE_MODELS_RESNET_DEFAULT_CONFIDENCE,
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


class ResNetClassifier(nn.Module):

    def __init__(self, backbone: nn.Module, softmax_fused: bool):
        super().__init__()
        self._backbone = backbone
        self._softmax_fused = softmax_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = self._backbone(x)
        if not self._softmax_fused:
            results = torch.nn.functional.softmax(results, dim=-1)
        return results


class ResNetForClassificationTorch(ClassificationModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "ResNetForClassificationTorch":
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
            implicit_resize_mode_substitutions={
                ResizeMode.FIT_LONGER_EDGE: (
                    ResizeMode.STRETCH_TO,
                    None,
                    "ResNetForClassification model running with Torch backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
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
        if inference_config.post_processing.type != "softmax":
            raise CorruptedModelPackageError(
                message="Expected softmax to be the post-processing",
                help_url="https://todo",
            )
        backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
        ).to(device)
        state_dict = torch.load(
            model_package_content["weights.pth"],
            weights_only=True,
            map_location=device,
        )
        backbone.load_state_dict(state_dict)
        model = ResNetClassifier(
            backbone=backbone,
            softmax_fused=inference_config.post_processing.fused,
        ).to(device)
        return cls(
            model=model.eval(),
            inference_config=inference_config,
            class_names=class_names,
            device=device,
        )

    def __init__(
        self,
        model: ResNetClassifier,
        inference_config: InferenceConfig,
        class_names: List[str],
        device: torch.device,
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
        self._device = device
        self._lock = Lock()

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
        with self._lock, torch.inference_mode():
            return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        **kwargs,
    ) -> ClassificationPrediction:
        return ClassificationPrediction(
            class_id=model_results.argmax(dim=-1),
            confidence=model_results,
        )


class ResNetMultiLabelClassifier(nn.Module):

    def __init__(self, backbone: nn.Module, sigmoid_fused: bool):
        super().__init__()
        self._backbone = backbone
        self._sigmoid_fused = sigmoid_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = self._backbone(x)
        if not self._sigmoid_fused:
            results = torch.nn.functional.sigmoid(results)
        return results


class ResNetForMultiLabelClassificationTorch(
    MultiLabelClassificationModel[torch.Tensor, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "ResNetForMultiLabelClassificationTorch":
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
            implicit_resize_mode_substitutions={
                ResizeMode.FIT_LONGER_EDGE: (
                    ResizeMode.STRETCH_TO,
                    None,
                    "ResNetForMultiLabelClassification model running with Torch backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during weights export. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
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
        if inference_config.post_processing.type != "sigmoid":
            raise CorruptedModelPackageError(
                message="Expected sigmoid to be the post-processing",
                help_url="https://todo",
            )
        backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
        ).to(device)
        state_dict = torch.load(
            model_package_content["weights.pth"],
            weights_only=True,
            map_location=device,
        )
        backbone.load_state_dict(state_dict)
        model = ResNetMultiLabelClassifier(
            backbone=backbone,
            sigmoid_fused=inference_config.post_processing.fused,
        ).to(device)
        return cls(
            model=model.eval(),
            inference_config=inference_config,
            class_names=class_names,
            device=device,
        )

    def __init__(
        self,
        model: ResNetMultiLabelClassifier,
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
        confidence: float = INFERENCE_MODELS_RESNET_DEFAULT_CONFIDENCE,
        **kwargs,
    ) -> List[MultiLabelClassificationPrediction]:
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
