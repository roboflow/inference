"""RFDETR-specific preprocessing.

Mirrors `rfdetr_internal/detr.py:RFDETR.predict()` exactly so inference-time
inputs are byte-equivalent to the training-time / parity-runner pipeline.

predict() has two branches that diverge at the resize step:

  numpy / PIL inputs (HTTP server case after JPEG decode):
      PIL → torchvision F.resize on PIL (PIL bilinear)
          → F.to_tensor (PIL → contiguous float32 CHW [0, 1])
          → F.normalize(mean, std)
  This branch matches `datasets/transforms.py` (training source-of-truth).

  torch.Tensor inputs (advanced caller, assumed float CHW [0, 1]):
      F.resize on tensor (torchvision tensor bilinear)
          → F.normalize(mean, std)
  This branch skips the PIL / numpy round-trip and uses tensor F.resize,
  matching predict()'s tensor branch. Note the tensor and PIL bilinear
  kernels are not byte-identical; predict() already accepts that drift on
  the tensor path, so we mirror it.

RFDETR training only ever stretches to a square (`square_resize_div_64=True`
on every Roboflow-trained RFDETR), so this preprocessor always stretches and
ignores `dataset_version_resize_dimensions` / `resize_mode` from the network
input definition. No fast-path gate; this chain applies for every RFDETR
inference regardless of the dataset version's preprocessing config.

YOLO and other model families continue to use the shared cv2-on-uint8 path in
`models.common.roboflow.pre_processing` to match their Ultralytics-derived
training conventions.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from inference_models import PreProcessingOverrides
from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.errors import ModelRuntimeError
from inference_models.logger import LOGGER
from inference_models.models.common.roboflow.model_packages import (
    AnySizePadding,
    ColorMode,
    DivisiblePadding,
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.pre_processing import (
    apply_pre_processing_to_numpy_image,
    apply_pre_processing_to_torch_image,
    make_the_value_divisible,
)


def pre_process_network_input(
    images: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    image_size_wh: Optional[Union[int, Tuple[int, int]]] = None,
    pre_processing_overrides: Optional[PreProcessingOverrides] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    input_color_mode = (
        ColorMode(input_color_format) if input_color_format is not None else None
    )
    if isinstance(image_size_wh, (int, float)):
        image_size_wh = (int(image_size_wh), int(image_size_wh))
    target_w = network_input.training_input_size.width
    target_h = network_input.training_input_size.height
    if image_size_wh is not None and image_size_wh != (target_w, target_h):
        if not network_input.dynamic_spatial_size_supported:
            LOGGER.warning(
                "Requested image size: %s cannot be applied for RFDETR input, as model was trained with "
                "input resolution and does not support inputs of a different shape. `image_size_wh` gets ignored.",
                image_size_wh,
            )
        elif isinstance(network_input.dynamic_spatial_size_mode, DivisiblePadding):
            target_w = make_the_value_divisible(
                x=image_size_wh[0], by=network_input.dynamic_spatial_size_mode.value
            )
            target_h = make_the_value_divisible(
                x=image_size_wh[1], by=network_input.dynamic_spatial_size_mode.value
            )
        elif isinstance(network_input.dynamic_spatial_size_mode, AnySizePadding):
            target_w, target_h = image_size_wh
        else:
            raise ModelRuntimeError(
                message=(
                    "Handler for dynamic spatial mode of type "
                    f"{type(network_input.dynamic_spatial_size_mode)} is not implemented."
                ),
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
    target_size = ImageDimensions(width=target_w, height=target_h)

    image_list = images if isinstance(images, list) else [images]

    tensors: List[torch.Tensor] = []
    metadata: List[PreProcessingMetadata] = []
    for img in image_list:
        if isinstance(img, torch.Tensor) and img.is_floating_point():
            tensor, meta = _pre_process_tensor(
                image=img,
                image_pre_processing=image_pre_processing,
                network_input=network_input,
                target_size=target_size,
                input_color_mode=input_color_mode,
                pre_processing_overrides=pre_processing_overrides,
            )
        elif isinstance(img, (np.ndarray, torch.Tensor)):
            np_img = (
                _tensor_to_hwc_uint8(img) if isinstance(img, torch.Tensor) else _ensure_hwc_uint8(img)
            )
            tensor, meta = _pre_process_numpy(
                image=np_img,
                image_pre_processing=image_pre_processing,
                network_input=network_input,
                target_size=target_size,
                input_color_mode=input_color_mode,
                pre_processing_overrides=pre_processing_overrides,
            )
        else:
            raise TypeError(
                f"Unsupported image input type for RFDETR pre-processing: {type(img)}"
            )
        tensors.append(tensor.to(device=target_device))
        metadata.append(meta)

    batch = torch.stack(tensors).contiguous()
    return batch, metadata


def _pre_process_numpy(
    image: np.ndarray,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_size: ImageDimensions,
    input_color_mode: Optional[ColorMode],
    pre_processing_overrides: Optional[PreProcessingOverrides],
) -> Tuple[torch.Tensor, PreProcessingMetadata]:
    """numpy / uint8-tensor branch: PIL chain matching training source-of-truth."""
    original_size = ImageDimensions(width=image.shape[1], height=image.shape[0])
    image, static_crop_offset = apply_pre_processing_to_numpy_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input_channels=network_input.input_channels,
        input_color_mode=input_color_mode,
        pre_processing_overrides=pre_processing_overrides,
    )
    size_after_pre_processing = ImageDimensions(
        width=image.shape[1], height=image.shape[0]
    )
    if input_color_mode != network_input.color_mode:
        image = image[:, :, ::-1]
    pil = Image.fromarray(np.ascontiguousarray(image))
    resized = TF.resize(pil, (target_size.height, target_size.width))
    tensor = TF.to_tensor(resized)
    tensor = _apply_normalization(tensor, network_input)
    return tensor, _build_metadata(
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        target_size=target_size,
        static_crop_offset=static_crop_offset,
    )


def _pre_process_tensor(
    image: torch.Tensor,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_size: ImageDimensions,
    input_color_mode: Optional[ColorMode],
    pre_processing_overrides: Optional[PreProcessingOverrides],
) -> Tuple[torch.Tensor, PreProcessingMetadata]:
    """Float-tensor branch: tensor F.resize matching predict()'s tensor branch.
    Skips PIL round-trip; assumes input is float CHW (or NCHW with N=1) in
    [0, 1]. Tensor F.resize bilinear is NOT byte-equivalent to PIL F.resize
    bilinear — predict() already accepts that on its tensor path."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[1] not in (1, 3) and image.shape[-1] in (1, 3):
        image = image.permute(0, 3, 1, 2)
    original_size = ImageDimensions(width=image.shape[3], height=image.shape[2])
    image, static_crop_offset = apply_pre_processing_to_torch_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input_channels=network_input.input_channels,
        pre_processing_overrides=pre_processing_overrides,
    )
    size_after_pre_processing = ImageDimensions(
        width=image.shape[3], height=image.shape[2]
    )
    if input_color_mode != network_input.color_mode:
        image = image[:, [2, 1, 0], :, :]
    resized = TF.resize(image, (target_size.height, target_size.width))
    if resized.shape[0] == 1:
        resized = resized.squeeze(0)
    tensor = _apply_normalization(resized, network_input)
    return tensor, _build_metadata(
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        target_size=target_size,
        static_crop_offset=static_crop_offset,
    )


def _apply_normalization(
    tensor: torch.Tensor, network_input: NetworkInputDefinition
) -> torch.Tensor:
    if network_input.normalization:
        mean, std = network_input.normalization
        tensor = TF.normalize(tensor, mean=mean, std=std)
    elif (
        network_input.scaling_factor is not None
        and network_input.scaling_factor != 255
    ):
        tensor = tensor * (255.0 / network_input.scaling_factor)
    return tensor


def _build_metadata(
    original_size: ImageDimensions,
    size_after_pre_processing: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> PreProcessingMetadata:
    return PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=target_size,
        scale_width=target_size.width / size_after_pre_processing.width,
        scale_height=target_size.height / size_after_pre_processing.height,
        static_crop_offset=static_crop_offset,
    )


def _ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _tensor_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return _ensure_hwc_uint8(arr)
