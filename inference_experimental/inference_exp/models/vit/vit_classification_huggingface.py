import os
from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import (
    ClassificationModel,
    ClassificationPrediction,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import (
    CorruptedModelPackageError,
    EnvironmentConfigurationError,
    MissingDependencyError,
)
from inference_exp.models.base.types import PreprocessedInputs
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.onnx import (
    run_session_with_batch_size_limit,
    set_execution_provider_defaults,
)
from inference_exp.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from inference_exp.utils.onnx_introspection import get_selected_onnx_execution_providers
from torch import nn
from transformers import ViTModel

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import VIT model with ONNX backend - this error means that some additional dependencies "
        f"are not installed in the environment. If you run the `inference-exp` library directly in your Python "
        f"program, make sure the following extras of the package are installed: \n"
        f"\t* `onnx-cpu` - when you wish to use library with CPU support only\n"
        f"\t* `onnx-cu12` - for running on GPU with Cuda 12 installed\n"
        f"\t* `onnx-cu118` - for running on GPU with Cuda 11.8 installed\n"
        f"\t* `onnx-jp6-cu126` - for running on Jetson with Jetpack 6\n"
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


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
        self.softmax_fused = softmax_fused

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self._backbone(pixel_values=pixel_values)
        logits = self._classifier(outputs.last_hidden_state[:, 0])
        if self._softmax_fused:
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
                ResizeMode.FIT_LONGER_EDGE,
            },
        )
        if inference_config.model_initialization is None:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameters not provided in inference config.",
                help_url="https://todo",
            )
        num_classes = inference_config.model_initialization.get("num_classes")
        if not isinstance(num_classes, int):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `num_classes` not provided or in invalid format.",
                help_url="https://todo",
            )
        if inference_config.post_processing.type != "softmax":
            raise CorruptedModelPackageError(
                message="Expected Softmax to be the post-processing",
                help_url="https://todo",
            )
        backbone = ViTModel.from_pretrained(os.path.join(model_name_or_path, "vit"))
        classifier = nn.Linear(backbone.config.hidden_size, num_classes)
        classifier_state_dict = torch.load(
            model_package_content["classifier_layer_weights.pth"],
            weights_only=True,
            map_location=device,
        )
        classifier.load_state_dict(classifier_state_dict)
        model = VITClassifier(
            backbone=backbone,
            classifier=classifier,
            softmax_fused=inference_config.post_processing.fused,
        ).eval()
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
        **kwargs,
    ) -> ClassificationPrediction:
        if self._inference_config.post_processing.fused:
            confidence = model_results
        else:
            confidence = torch.nn.functional.softmax(model_results, dim=-1)
        return ClassificationPrediction(
            class_id=confidence.argmax(dim=-1),
            confidence=confidence,
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
        if self._sigmoid_fused:
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
                ResizeMode.FIT_LONGER_EDGE,
            },
        )
        if inference_config.model_initialization is None:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameters not provided in inference config.",
                help_url="https://todo",
            )
        num_classes = inference_config.model_initialization.get("num_classes")
        if not isinstance(num_classes, int):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `num_classes` not provided or in invalid format.",
                help_url="https://todo",
            )
        if inference_config.post_processing.type != "sigmoid":
            raise CorruptedModelPackageError(
                message="Expected sigmoid to be the post-processing",
                help_url="https://todo",
            )
        backbone = ViTModel.from_pretrained(os.path.join(model_name_or_path, "vit"))
        classifier = nn.Linear(backbone.config.hidden_size, num_classes)
        classifier_state_dict = torch.load(
            model_package_content["classifier_layer_weights.pth"],
            weights_only=True,
            map_location=device,
        )
        classifier.load_state_dict(classifier_state_dict)
        model = VITMultiLabelClassifier(
            backbone=backbone,
            classifier=classifier,
            sigmoid_fused=inference_config.post_processing.fused,
        ).eval()
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
        **kwargs,
    ) -> MultiLabelClassificationPrediction:
        if self._inference_config.post_processing.fused:
            confidence = model_results
        else:
            confidence = torch.nn.functional.sigmoid(model_results)
        return MultiLabelClassificationPrediction(
            class_ids=confidence.argmax(dim=-1),
            confidence=confidence,
        )
