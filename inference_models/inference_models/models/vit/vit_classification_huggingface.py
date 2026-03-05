import os
from threading import Lock
from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import ViTModel

from inference_models import (
    ClassificationModel,
    ClassificationPrediction,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_VIT_CLASSIFIER_DEFAULT_CONFIDENCE,
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


class VITClassifier(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        softmax_fused: bool,
    ):
        super().__init__()
        self._backbone = backbone
        self._classifier = classifier
        self._softmax_fused = softmax_fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self._backbone(pixel_values=pixel_values)
        logits = self._classifier(outputs.last_hidden_state[:, 0])
        if not self._softmax_fused:
            logits = torch.nn.functional.softmax(logits, dim=-1)
        return logits


class VITForClassificationHF(ClassificationModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "VITForClassificationHF":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "classifier_layer_weights.pth",
                "inference_config.json",
                "vit/config.json",
                "vit/model.safetensors",
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
                    "VIT Classification model running with HF backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during training. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        if inference_config.model_initialization is None:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameters not provided in inference config.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        num_classes = inference_config.model_initialization.get("num_classes")
        if not isinstance(num_classes, int):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `num_classes` not provided or in invalid format.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if inference_config.post_processing.type != "softmax":
            raise CorruptedModelPackageError(
                message="Expected Softmax to be the post-processing",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        backbone = ViTModel.from_pretrained(os.path.join(model_name_or_path, "vit")).to(
            device
        )
        classifier = nn.Linear(backbone.config.hidden_size, num_classes).to(device)
        classifier_state_dict = torch.load(
            model_package_content["classifier_layer_weights.pth"],
            weights_only=True,
            map_location=device,
        )
        classifier.load_state_dict(classifier_state_dict)
        model = (
            VITClassifier(
                backbone=backbone,
                classifier=classifier,
                softmax_fused=inference_config.post_processing.fused,
            )
            .to(device)
            .eval()
        )
        return cls(
            model=model,
            inference_config=inference_config,
            class_names=class_names,
            device=device,
        )

    def __init__(
        self,
        model: VITClassifier,
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
        **kwargs,
    ) -> torch.Tensor:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
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


class VITMultiLabelClassifier(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        sigmoid_fused: bool,
    ):
        super().__init__()
        self._backbone = backbone
        self._classifier = classifier
        self._sigmoid_fused = sigmoid_fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self._backbone(pixel_values=pixel_values)
        logits = self._classifier(outputs.last_hidden_state[:, 0])
        if not self._sigmoid_fused:
            logits = torch.nn.functional.sigmoid(logits)
        return logits


class VITForMultiLabelClassificationHF(
    MultiLabelClassificationModel[torch.Tensor, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "VITForMultiLabelClassificationHF":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "classifier_layer_weights.pth",
                "inference_config.json",
                "vit/config.json",
                "vit/model.safetensors",
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
                    "VIT Multi-Label Classification model running with HF backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "models with input dimensions fixed during training. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
        )
        if inference_config.model_initialization is None:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameters not provided in inference config.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        num_classes = inference_config.model_initialization.get("num_classes")
        if not isinstance(num_classes, int):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `num_classes` not provided or in invalid format.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if inference_config.post_processing.type != "sigmoid":
            raise CorruptedModelPackageError(
                message="Expected sigmoid to be the post-processing",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        backbone = ViTModel.from_pretrained(os.path.join(model_name_or_path, "vit")).to(
            device
        )
        classifier = nn.Linear(backbone.config.hidden_size, num_classes).to(device)
        classifier_state_dict = torch.load(
            model_package_content["classifier_layer_weights.pth"],
            weights_only=True,
            map_location=device,
        )
        classifier.load_state_dict(classifier_state_dict)
        model = (
            VITMultiLabelClassifier(
                backbone=backbone,
                classifier=classifier,
                sigmoid_fused=inference_config.post_processing.fused,
            )
            .to(device)
            .eval()
        )
        return cls(
            model=model,
            inference_config=inference_config,
            class_names=class_names,
            device=device,
        )

    def __init__(
        self,
        model: VITMultiLabelClassifier,
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
        **kwargs,
    ) -> torch.Tensor:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
        )[0]

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.inference_mode():
            return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        confidence: float = INFERENCE_MODELS_VIT_CLASSIFIER_DEFAULT_CONFIDENCE,
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
