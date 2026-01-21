from typing import List, Optional, Tuple, Union

import segmentation_models_pytorch as smp
import torch
from torchvision.transforms import functional

from inference_models import ColorFormat, SemanticSegmentationModel
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import CorruptedModelPackageError
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
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)


class DeepLabV3PlusForSemanticSegmentationTorch(
    SemanticSegmentationModel[torch.Tensor, PreProcessingMetadata, torch.Tensor]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
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
        try:
            background_class_id = [c.lower() for c in class_names].index("background")
        except ValueError:
            background_class_id = -1
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
        num_classes = inference_config.model_initialization.get("classes")
        in_channels = inference_config.model_initialization.get("in_channels")
        encoder_name = inference_config.model_initialization.get("encoder_name")
        if not isinstance(num_classes, int) or num_classes < 1:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `num_classes` not provided or in invalid format.",
                help_url="https://todo",
            )
        if not isinstance(in_channels, int) or in_channels not in {1, 3}:
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `in_channels` not provided or in invalid format.",
                help_url="https://todo",
            )
        if not isinstance(encoder_name, str):
            raise CorruptedModelPackageError(
                message="Expected model initialization parameter `encoder_name` not provided or in invalid format.",
                help_url="https://todo",
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
        )

    def __init__(
        self,
        model: smp.DeepLabV3Plus,
        inference_config: InferenceConfig,
        class_names: List[str],
        background_class_id: int,
        device: torch.device,
    ):
        self._model = model
        self._inference_config = inference_config
        self._class_names = class_names
        self._background_class_id = background_class_id
        self._device = device

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, PreprocessingMetadata]:
        return pre_process_network_input(
            images=images,
            image_pre_processing=self._inference_config.image_pre_processing,
            network_input=self._inference_config.network_input,
            target_device=self._device,
            input_color_format=input_color_format,
            image_size_wh=image_size,
        )

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.inference_mode():
            return self._model(pre_processed_images)

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: float = 0.5,
        **kwargs,
    ) -> List[SemanticSegmentationResult]:
        results = []
        for image_results, image_metadata in zip(model_results, pre_processing_meta):
            inference_size = image_metadata.inference_size
            mask_h_scale = model_results.shape[2] / inference_size.height
            mask_w_scale = model_results.shape[3] / inference_size.width
            mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = (
                round(mask_h_scale * image_metadata.pad_top),
                round(mask_h_scale * image_metadata.pad_bottom),
                round(mask_w_scale * image_metadata.pad_left),
                round(mask_w_scale * image_metadata.pad_right),
            )
            _, mh, mw = image_results.shape
            if (
                mask_pad_top < 0
                or mask_pad_bottom < 0
                or mask_pad_left < 0
                or mask_pad_right < 0
            ):
                image_results = torch.nn.functional.pad(
                    image_results,
                    (
                        abs(min(mask_pad_left, 0)),
                        abs(min(mask_pad_right, 0)),
                        abs(min(mask_pad_top, 0)),
                        abs(min(mask_pad_bottom, 0)),
                    ),
                    "constant",
                    self._background_class_id,
                )
                padded_mask_offset_top = max(mask_pad_top, 0)
                padded_mask_offset_bottom = max(mask_pad_bottom, 0)
                padded_mask_offset_left = max(mask_pad_left, 0)
                padded_mask_offset_right = max(mask_pad_right, 0)
                image_results = image_results[
                    :,
                    padded_mask_offset_top : image_results.shape[1]
                    - padded_mask_offset_bottom,
                    padded_mask_offset_left : image_results.shape[1]
                    - padded_mask_offset_right,
                ]
            else:
                image_results = image_results[
                    :,
                    mask_pad_top : mh - mask_pad_bottom,
                    mask_pad_left : mw - mask_pad_right,
                ]
            if (
                image_results.shape[1]
                != image_metadata.size_after_pre_processing.height
                or image_results.shape[2]
                != image_metadata.size_after_pre_processing.width
            ):
                image_results = functional.resize(
                    image_results,
                    [
                        image_metadata.size_after_pre_processing.height,
                        image_metadata.size_after_pre_processing.width,
                    ],
                    interpolation=functional.InterpolationMode.BILINEAR,
                )
            image_results = torch.nn.functional.softmax(image_results, dim=0)
            image_confidence, image_class_ids = torch.max(image_results, dim=0)
            below_threshold = image_confidence < confidence
            image_confidence[below_threshold] = 0.0
            image_class_ids[below_threshold] = self._background_class_id
            if (
                image_metadata.static_crop_offset.offset_x > 0
                or image_metadata.static_crop_offset.offset_y > 0
            ):
                original_size_confidence_canvas = torch.zeros(
                    (
                        image_metadata.original_size.height,
                        image_metadata.original_size.width,
                    ),
                    device=self._device,
                    dtype=image_confidence.dtype,
                )
                original_size_confidence_canvas[
                    image_metadata.static_crop_offset.offset_y : image_metadata.static_crop_offset.offset_y
                    + image_confidence.shape[0],
                    image_metadata.static_crop_offset.offset_x : image_metadata.static_crop_offset.offset_x
                    + image_confidence.shape[1],
                ] = image_confidence
                original_size_confidence_class_id_canvas = (
                    torch.ones(
                        (
                            image_metadata.original_size.height,
                            image_metadata.original_size.width,
                        ),
                        device=self._device,
                        dtype=image_class_ids.dtype,
                    )
                    * self._background_class_id
                )
                original_size_confidence_class_id_canvas[
                    image_metadata.static_crop_offset.offset_y : image_metadata.static_crop_offset.offset_y
                    + image_class_ids.shape[0],
                    image_metadata.static_crop_offset.offset_x : image_metadata.static_crop_offset.offset_x
                    + image_class_ids.shape[1],
                ] = image_class_ids
                image_class_ids = original_size_confidence_class_id_canvas
                image_confidence = original_size_confidence_canvas
            results.append(
                SemanticSegmentationResult(
                    segmentation_map=image_class_ids,
                    confidence=image_confidence,
                )
            )
        return results
