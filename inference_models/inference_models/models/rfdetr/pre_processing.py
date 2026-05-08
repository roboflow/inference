"""RFDETR-specific preprocessing matching the training pipeline.

numpy / PIL inputs:
    [optional cv2 dataset-version resize] → PIL F.resize → F.to_tensor → F.normalize

The cv2 dataset-version resize runs first when `resize_mode != STRETCH_TO`
and `dataset_version_resize_dimensions` is set; the PIL stretch then takes
the result to `training_input_size`.

torch.Tensor inputs (advanced caller, float CHW [0, 1]):
    tensor F.resize → F.normalize
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
    ResizeMode,
    StaticCropOffset,
    TrainingInputSize,
)
from inference_models.models.common.roboflow.pre_processing import (
    apply_pre_processing_to_numpy_image,
    apply_pre_processing_to_torch_image,
    make_the_value_divisible,
    pre_process_numpy_image,
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

    if isinstance(images, list):
        image_list: List[Union[np.ndarray, torch.Tensor]] = list(images)
    elif isinstance(images, torch.Tensor) and images.ndim == 4:
        image_list = list(images.unbind(0))
    elif isinstance(images, np.ndarray) and images.ndim == 4:
        image_list = list(images)
    else:
        image_list = [images]

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
    """numpy / uint8-tensor branch: PIL chain matching training source-of-truth.

    For non-stretch resize modes with non-square `dataset_version_resize_dimensions`
    we first apply the dataset-version resize via the shared cv2-on-uint8 handler
    (matching what Roboflow's exporter applies), then PIL F.resize stretches to
    `training_input_size` (matching training's SquareResize). Otherwise we stretch
    directly in a single PIL F.resize step.
    """
    if _needs_two_step_resize(network_input):
        intermediate_image, meta = _dataset_version_resize_uint8(
            image=image,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            input_color_mode=input_color_mode,
            pre_processing_overrides=pre_processing_overrides,
        )
        meta = meta._replace(
            nonsquare_intermediate_size=meta.inference_size,
            inference_size=target_size,
        )
        pil = Image.fromarray(np.ascontiguousarray(intermediate_image))
    else:
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
        meta = _build_metadata(
            original_size=original_size,
            size_after_pre_processing=size_after_pre_processing,
            target_size=target_size,
            static_crop_offset=static_crop_offset,
        )

    resized = TF.resize(pil, (target_size.height, target_size.width), antialias=True)
    tensor = TF.to_tensor(resized)
    tensor = _apply_normalization(tensor, network_input)
    return tensor, meta


def _needs_two_step_resize(network_input: NetworkInputDefinition) -> bool:
    """True when the dataset-version resize_mode is non-stretch — the resize
    Roboflow's exporter applied at version creation needs to be replayed at
    inference for production-served pixels to match training-time pixels.
    STRETCH_TO with any dims collapses to a single PIL F.resize stretch."""
    dims = network_input.dataset_version_resize_dimensions
    return (
        dims is not None
        and network_input.resize_mode != ResizeMode.STRETCH_TO
    )


def _dataset_version_resize_uint8(
    image: np.ndarray,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    input_color_mode: Optional[ColorMode],
    pre_processing_overrides: Optional[PreProcessingOverrides],
) -> Tuple[np.ndarray, PreProcessingMetadata]:
    """Apply the dataset-version resize via the shared cv2-on-uint8 handler and
    return (uint8 HWC numpy, metadata). The numpy is in `network_input.color_mode`
    channel order. We rebuild a stripped-down NetworkInputDefinition that targets
    the dataset-version dims and disables `scaling_factor` / `normalization` so
    the handler skips the /255 + normalize step and we keep raw pixels for the
    second PIL F.resize."""
    dims = network_input.dataset_version_resize_dimensions
    effective = network_input.model_copy(
        update={
            "training_input_size": TrainingInputSize(
                height=dims.height, width=dims.width
            ),
            "scaling_factor": None,
            "normalization": None,
            "dataset_version_resize_dimensions": None,
        }
    )
    tensor, metadatas = pre_process_numpy_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input=effective,
        target_device=torch.device("cpu"),
        input_color_mode=input_color_mode,
        pre_processing_overrides=pre_processing_overrides,
    )
    # Handler returns NCHW. STRETCH_TO produces uint8 (cv2.resize on uint8); the
    # letterbox / center-crop / letterbox-reflect paths construct a float32 buffer
    # and scatter the cv2-resized uint8 into it (values stay in [0, 255] integers,
    # padding values are 0/127/255). Round-trip via .to(uint8) is exact.
    chw = tensor[0]
    if chw.dtype == torch.uint8:
        arr = chw.permute(1, 2, 0).cpu().numpy()
    elif chw.dtype == torch.float32:
        arr = chw.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    else:
        raise ModelRuntimeError(
            message=(
                f"Unexpected dtype {chw.dtype} from shared dataset-version resize "
                "handler; expected torch.uint8 or torch.float32."
            ),
            help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
        )
    return arr, metadatas[0]


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
    resized = TF.resize(image, (target_size.height, target_size.width), antialias=True)
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
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        return (image * 255.0).clip(0, 255).astype(np.uint8)
    return image.astype(np.uint8)


def _tensor_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return _ensure_hwc_uint8(arr)
