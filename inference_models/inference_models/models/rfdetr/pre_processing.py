from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
    TrainingInputSize,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input as _pre_process_network_input,
)


def _needs_nonsquare_two_step_resize(network_input: NetworkInputDefinition) -> bool:
    dims = network_input.dataset_version_resize_dimensions
    return (
        dims is not None
        and network_input.resize_mode != ResizeMode.STRETCH_TO
        and dims.width != dims.height
    )


def pre_process_network_input(
    images: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    image_size_wh: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    """RF-DETR wrapper around the shared pre_process_network_input.

    When the model config specifies non-square dataset_version_resize_dimensions
    with a square training_input_size, we first letterbox to the non-square
    intermediate size, then bilinear-interpolate to the square training size.
    This matches the two-step resize the Roboflow training pipeline applies.
    """
    two_step = _needs_nonsquare_two_step_resize(network_input)
    effective_network_input = network_input
    if two_step:
        dims = network_input.dataset_version_resize_dimensions
        effective_network_input = network_input.model_copy(
            update={
                "training_input_size": TrainingInputSize(
                    height=dims.height, width=dims.width
                )
            }
        )

    tensor, metadata = _pre_process_network_input(
        images=images,
        image_pre_processing=image_pre_processing,
        network_input=effective_network_input,
        target_device=target_device,
        input_color_format=input_color_format,
        image_size_wh=image_size_wh,
    )

    if two_step:
        target_h = network_input.training_input_size.height
        target_w = network_input.training_input_size.width
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
        )
        actual_inference_size = ImageDimensions(height=target_h, width=target_w)
        metadata = [
            m._replace(
                nonsquare_intermediate_size=m.inference_size,
                inference_size=actual_inference_size,
            )
            for m in metadata
        ]

    return tensor, metadata
