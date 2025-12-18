from typing import List, Optional, Tuple, Union

import numpy as np
import timm
import torch
from inference_exp import (
    ClassificationModel,
    ClassificationPrediction,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_class_names_file,
    parse_inference_config,
)
from inference_exp.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)
from torch import nn

# Mapping from model names in inference_config to timm model configs
# Format: "config_name": (timm_model_name, patch_size)
# We use DINOv2 with registers from timm since DINOv3 weights are compatible
# with the architecture when using proper key remapping
DINOV3_MODEL_CONFIG = {
    "vit_small_patch16_dinov3": ("vit_small_patch14_reg4_dinov2", 16),
    "vit_base_patch16_dinov3": ("vit_base_patch14_reg4_dinov2", 16),
    "vit_large_patch16_dinov3": ("vit_large_patch14_reg4_dinov2", 16),
    "vit_giant_patch16_dinov3": ("vit_giant_patch14_reg4_dinov2", 16),
}


def _remap_state_dict_keys(state_dict: dict) -> Tuple[dict, dict]:
    """Remap state dict keys from DINOv3 format to timm DINOv2 format.

    DINOv3 weights use:
    - backbone.blocks.X.gamma_1 -> blocks.X.ls1.gamma
    - backbone.blocks.X.gamma_2 -> blocks.X.ls2.gamma

    Returns:
        Tuple of (backbone_state_dict, linear_layer_state_dict)
    """
    backbone_state_dict = {}
    linear_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("backbone."):
            new_key = key[len("backbone.") :]
            # Remap layer scale keys
            new_key = new_key.replace(".gamma_1", ".ls1.gamma")
            new_key = new_key.replace(".gamma_2", ".ls2.gamma")
            backbone_state_dict[new_key] = value
        elif key.startswith("linear_layer."):
            new_key = key[len("linear_layer.") :]
            linear_state_dict[new_key] = value

    return backbone_state_dict, linear_state_dict


def _create_dinov3_backbone(model_name: str, img_size: int = 224) -> nn.Module:
    """Create DINOv3-compatible backbone using timm's DINOv2 with registers.

    The DINOv3 architecture is compatible with timm's DINOv2 with registers
    when using patch_size override and qkv_bias=False.
    """
    # Remove dataset suffix like ".lvd1689m" or ".sat493m"
    base_name = model_name.split(".")[0]
    if base_name not in DINOV3_MODEL_CONFIG:
        raise CorruptedModelPackageError(
            message=f"Unknown DINOv3 model name: {model_name}. "
            f"Supported models: {list(DINOV3_MODEL_CONFIG.keys())}",
            help_url="https://todo",
        )
    timm_model_name, patch_size = DINOV3_MODEL_CONFIG[base_name]

    backbone = timm.create_model(
        timm_model_name,
        pretrained=False,
        qkv_bias=False,
        patch_size=patch_size,
        num_classes=0,  # Remove classification head
        img_size=img_size,
    )
    return backbone


class DinoV3Classifier(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        linear_layer: nn.Module,
        softmax_fused: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.linear_layer = linear_layer
        self._softmax_fused = softmax_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.linear_layer(features)
        if not self._softmax_fused:
            logits = torch.nn.functional.softmax(logits, dim=-1)
        return logits


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

        # Get image size from config
        img_size = inference_config.network_input.training_input_size.width

        # Create backbone and linear layer
        backbone = _create_dinov3_backbone(model_name, img_size=img_size)
        embed_dim = backbone.embed_dim
        linear_layer = nn.Linear(embed_dim, num_classes)

        # Load state dict with key remapping
        state_dict = torch.load(
            model_package_content["weights.pth"],
            map_location=device,
            weights_only=True,
        )
        backbone_state_dict, linear_state_dict = _remap_state_dict_keys(state_dict)

        # Load backbone weights (strict=False to skip pos_embed which is generated)
        backbone.load_state_dict(backbone_state_dict, strict=False)
        linear_layer.load_state_dict(linear_state_dict)

        model = DinoV3Classifier(
            backbone=backbone,
            linear_layer=linear_layer,
            softmax_fused=inference_config.post_processing.fused,
        )
        model = model.to(device).eval()
        return cls(
            model=model,
            inference_config=inference_config,
            class_names=class_names,
            device=device,
        )

    def __init__(
        self,
        model: DinoV3Classifier,
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
        return ClassificationPrediction(
            class_id=model_results.argmax(dim=-1),
            confidence=model_results,
        )


class DinoV3MultiLabelClassifier(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        linear_layer: nn.Module,
        sigmoid_fused: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.linear_layer = linear_layer
        self._sigmoid_fused = sigmoid_fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.linear_layer(features)
        if not self._sigmoid_fused:
            logits = torch.nn.functional.sigmoid(logits)
        return logits


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

        # Get image size from config
        img_size = inference_config.network_input.training_input_size.width

        # Create backbone and linear layer
        backbone = _create_dinov3_backbone(model_name, img_size=img_size)
        embed_dim = backbone.embed_dim
        linear_layer = nn.Linear(embed_dim, num_classes)

        # Load state dict with key remapping
        state_dict = torch.load(
            model_package_content["weights.pth"],
            map_location=device,
            weights_only=True,
        )
        backbone_state_dict, linear_state_dict = _remap_state_dict_keys(state_dict)

        # Load backbone weights (strict=False to skip pos_embed which is generated)
        backbone.load_state_dict(backbone_state_dict, strict=False)
        linear_layer.load_state_dict(linear_state_dict)

        model = DinoV3MultiLabelClassifier(
            backbone=backbone,
            linear_layer=linear_layer,
            sigmoid_fused=inference_config.post_processing.fused,
        )
        model = model.to(device).eval()
        return cls(
            model=model,
            inference_config=inference_config,
            class_names=class_names,
            device=device,
        )

    def __init__(
        self,
        model: DinoV3MultiLabelClassifier,
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
        confidence: float = 0.5,
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
