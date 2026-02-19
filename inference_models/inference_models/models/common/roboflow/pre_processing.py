import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL
import torch
from PIL.Image import Image
from skimage import exposure
from torchvision.transforms import Grayscale, functional

from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.errors import ModelInputError, ModelRuntimeError
from inference_models.logger import LOGGER
from inference_models.models.common.roboflow.model_packages import (
    AnySizePadding,
    ColorMode,
    ContrastType,
    DivisiblePadding,
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
    StaticCrop,
    StaticCropOffset,
)


def pre_process_network_input(
    images: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    image_size_wh: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if network_input.input_channels != 3:
        raise ModelRuntimeError(
            message=f"`inference` currently does not support Roboflow pre-processing for model inputs with "
            f"channels numbers different than 1. Let us know if you need this feature.",
            help_url="https://todo",
        )
    input_color_mode = None
    if input_color_format is not None:
        input_color_mode = ColorMode(input_color_format)
    if isinstance(image_size_wh, (int, float)):
        image_size_wh = int(image_size_wh), int(image_size_wh)
    if isinstance(images, np.ndarray):
        return pre_process_numpy_image(
            image=images,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=target_device,
            input_color_mode=input_color_mode,
            image_size_wh=image_size_wh,
        )
    if isinstance(images, torch.Tensor):
        return pre_process_images_tensor(
            images=images,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            input_color_mode=input_color_mode,
            target_device=target_device,
            image_size_wh=image_size_wh,
        )
    if not isinstance(images, list):
        raise ModelInputError(
            message="Pre-processing supports only np.array or torch.Tensor or list of above.",
            help_url="https://todo",
        )
    if not len(images):
        raise ModelInputError(
            message="Detected empty input to the model", help_url="https://todo"
        )
    if network_input.resize_mode is ResizeMode.FIT_LONGER_EDGE:
        raise ModelRuntimeError(
            message="Model input resize type (fit-longer-edge) cannot be applied equally for "
            "all input batch elements arbitrarily - this type of model does not support input batches.",
            help_url="https://todo",
        )
    if isinstance(images[0], np.ndarray):
        return pre_process_numpy_images_list(
            images=images,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            input_color_mode=input_color_mode,
            target_device=target_device,
            image_size_wh=image_size_wh,
        )
    if isinstance(images[0], torch.Tensor):
        return pre_process_images_tensor_list(
            images=images,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            input_color_mode=input_color_mode,
            target_device=target_device,
            image_size_wh=image_size_wh,
        )
    raise ModelInputError(
        message=f"Detected unknown input batch element: {type(images[0])}",
        help_url="https://todo",
    )


@torch.inference_mode()
def pre_process_images_tensor(
    images: torch.Tensor,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_mode: Optional[ColorMode] = None,
    image_size_wh: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if input_color_mode is None:
        input_color_mode = ColorMode.RGB
    target_dimensions = (
        network_input.training_input_size.width,
        network_input.training_input_size.height,
    )
    if image_size_wh is not None and image_size_wh != target_dimensions:
        if not network_input.dynamic_spatial_size_supported:
            LOGGER.warning(
                f"Requested image size: {image_size_wh} cannot be applied for model input, as model was trained with "
                f"input resolution and does not support inputs of a different shape. `image_size_wh` gets ignored."
            )
        elif isinstance(network_input.dynamic_spatial_size_mode, DivisiblePadding):
            target_dimensions = (
                make_the_value_divisible(
                    x=image_size_wh[0], by=network_input.dynamic_spatial_size_mode.value
                ),
                make_the_value_divisible(
                    x=image_size_wh[1], by=network_input.dynamic_spatial_size_mode.value
                ),
            )
        elif isinstance(network_input.dynamic_spatial_size_mode, AnySizePadding):
            target_dimensions = image_size_wh
        else:
            raise ModelRuntimeError(
                message=f"Handler for dynamic spatial mode of type {type(network_input.dynamic_spatial_size_mode)} "
                f"is not implemented.",
                help_url="",
            )
    if images.device != target_device:
        images = images.to(target_device)
    if len(images.shape) == 3:
        images = torch.unsqueeze(images, 0)
    if (
        images.shape[1] != network_input.input_channels
        and images.shape[3] == network_input.input_channels
    ):
        images = images.permute(0, 3, 1, 2)
    original_size = ImageDimensions(width=images.shape[3], height=images.shape[2])
    image, static_crop_offset = apply_pre_processing_to_torch_image(
        image=images,
        image_pre_processing=image_pre_processing,
        network_input_channels=network_input.input_channels,
    )
    if network_input.resize_mode not in NUMPY_IMAGES_PREPARATION_HANDLERS:
        raise ModelRuntimeError(
            message=f"Unsupported model input resize mode: {network_input.resize_mode}",
            help_url="https://todo",
        )
    return TORCH_IMAGES_PREPARATION_HANDLERS[network_input.resize_mode](
        image,
        network_input,
        input_color_mode,
        original_size,
        ImageDimensions(width=target_dimensions[0], height=target_dimensions[1]),
        static_crop_offset,
    )


def apply_pre_processing_to_torch_image(
    image: torch.Tensor,
    image_pre_processing: ImagePreProcessing,
    network_input_channels: int,
) -> Tuple[torch.Tensor, StaticCropOffset]:
    static_crop_offset = StaticCropOffset(
        offset_x=0,
        offset_y=0,
        crop_width=image.shape[3],
        crop_height=image.shape[2],
    )
    if image_pre_processing.static_crop and image_pre_processing.static_crop.enabled:
        image, static_crop_offset = apply_static_crop_to_torch_image(
            image=image,
            config=image_pre_processing.static_crop,
        )
    if image_pre_processing.grayscale and image_pre_processing.grayscale.enabled:
        image = Grayscale(num_output_channels=network_input_channels)(image)
    if image_pre_processing.contrast and image_pre_processing.contrast.enabled:
        if (
            image_pre_processing.contrast.type
            not in CONTRAST_ADJUSTMENT_METHODS_FOR_TORCH
        ):
            raise ModelRuntimeError(
                message=f"Unsupported image contrast adjustment type: {image_pre_processing.contrast.type.value}",
                help_url="https://todo",
            )
        image = CONTRAST_ADJUSTMENT_METHODS_FOR_TORCH[
            image_pre_processing.contrast.type
        ](image)
    return image, static_crop_offset


def apply_static_crop_to_torch_image(
    image: torch.Tensor, config: StaticCrop
) -> Tuple[torch.Tensor, StaticCropOffset]:
    width, height = image.shape[3], image.shape[2]
    x_min = int(config.x_min / 100 * width)
    y_min = int(config.y_min / 100 * height)
    x_max = int(config.x_max / 100 * width)
    y_max = int(config.y_max / 100 * height)
    cropped_tensor = image[:, :, y_min:y_max, x_min:x_max]
    offset = StaticCropOffset(
        offset_x=x_min,
        offset_y=y_min,
        crop_width=cropped_tensor.shape[3],
        crop_height=cropped_tensor.shape[2],
    )
    return cropped_tensor, offset


def apply_adaptive_equalization_to_torch_image(image: torch.Tensor) -> torch.Tensor:
    original_device = image.device
    results = []
    for single_image in image:
        single_image_numpy = np.transpose(single_image.cpu().numpy(), (1, 2, 0))
        image = single_image_numpy.astype(np.float32) / 255
        image_adapted = (
            exposure.equalize_adapthist(image, clip_limit=0.03) * 255
        ).astype(np.uint8)
        results.append(torch.from_numpy(image_adapted).to(original_device))
    return torch.stack(results, dim=0).permute(0, 3, 1, 2)


def apply_contrast_stretching_to_torch_image(image: torch.Tensor) -> torch.Tensor:
    original_device = image.device
    results = []
    for single_image in image:
        single_image_numpy = np.transpose(single_image.cpu().numpy(), (1, 2, 0))
        p2 = np.percentile(single_image_numpy, 2)
        p98 = np.percentile(single_image_numpy, 98)
        rescaled_image = exposure.rescale_intensity(
            single_image_numpy, in_range=(p2, p98)
        )
        results.append(torch.from_numpy(rescaled_image).to(original_device))
    return torch.stack(results, dim=0).permute(0, 3, 1, 2)


def apply_histogram_equalization_to_torch_image(image: torch.Tensor) -> torch.Tensor:
    original_device = image.device
    results = []
    for single_image in image:
        single_image_numpy = np.transpose(single_image.cpu().numpy(), (1, 2, 0))
        single_image_numpy = single_image_numpy.astype(np.float32) / 255
        image_equalized = exposure.equalize_hist(single_image_numpy) * 255
        results.append(torch.from_numpy(image_equalized).to(original_device))
    return torch.stack(results, dim=0).permute(0, 3, 1, 2)


CONTRAST_ADJUSTMENT_METHODS_FOR_TORCH = {
    ContrastType.ADAPTIVE_EQUALIZATION: apply_adaptive_equalization_to_torch_image,
    ContrastType.CONTRAST_STRETCHING: apply_contrast_stretching_to_torch_image,
    ContrastType.HISTOGRAM_EQUALIZATION: apply_histogram_equalization_to_torch_image,
}


def handle_tensor_input_preparation_with_stretch(
    image: torch.Tensor,
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    original_size: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    size_after_pre_processing = ImageDimensions(
        height=image.shape[2], width=image.shape[3]
    )
    if image.device.type == "cuda":
        image = image.float()
    image = torch.nn.functional.interpolate(
        image,
        size=[target_size.height, target_size.width],
        mode="bilinear",
    )
    if input_color_mode != network_input.color_mode:
        image = image[:, [2, 1, 0], :, :]
    if network_input.scaling_factor is not None:
        image = image / network_input.scaling_factor
    if network_input.normalization is not None:
        if not image.is_floating_point():
            image = image.to(dtype=torch.float32)
        image = functional.normalize(
            image,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    metadata = PreProcessingMetadata(
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
    return image.contiguous(), [metadata] * image.shape[0]


def handle_torch_input_preparation_with_letterbox(
    image: torch.Tensor,
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    original_size: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    original_height, original_width = image.shape[2], image.shape[3]
    size_after_pre_processing = ImageDimensions(
        height=original_height, width=original_width
    )
    scale_w = target_size.width / original_width
    scale_h = target_size.height / original_height
    scale = min(scale_w, scale_h)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    pad_top = int((target_size.height - new_height) / 2)
    pad_left = int((target_size.width - new_width) / 2)
    if image.device.type == "cuda":
        image = image.float()
    image = torch.nn.functional.interpolate(
        image,
        [new_height, new_width],
        mode="bilinear",
    )
    if input_color_mode != network_input.color_mode:
        image = image[:, [2, 1, 0], :, :]
    final_batch = torch.full(
        (
            image.shape[0],
            image.shape[1],
            target_size.height,
            target_size.width,
        ),
        network_input.padding_value or 0,
        dtype=torch.float32,
        device=image.device,
    )
    final_batch[
        :, :, pad_top : pad_top + new_height, pad_left : pad_left + new_width
    ] = image
    pad_right = target_size.width - pad_left - new_width
    pad_bottom = target_size.height - pad_top - new_height
    metadata = PreProcessingMetadata(
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=target_size,
        scale_width=scale,
        scale_height=scale,
        static_crop_offset=static_crop_offset,
    )
    if network_input.scaling_factor is not None:
        final_batch = final_batch / network_input.scaling_factor
    if network_input.normalization is not None:
        if not final_batch.is_floating_point():
            final_batch = final_batch.to(dtype=torch.float32)
        final_batch = functional.normalize(
            final_batch,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    return final_batch.contiguous(), [metadata] * final_batch.shape[0]


def handle_torch_input_preparation_with_center_crop(
    image: torch.Tensor,
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    original_size: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if input_color_mode != network_input.color_mode:
        image = image[:, [2, 1, 0], :, :]
    size_after_pre_processing = ImageDimensions(
        height=image.shape[2], width=image.shape[3]
    )
    padding_ltrb = [0, 0, 0, 0]
    if (
        target_size.width > size_after_pre_processing.width
        or target_size.height > size_after_pre_processing.height
    ):
        padding_ltrb = [
            (
                (target_size.width - size_after_pre_processing.width) // 2
                if target_size.width > size_after_pre_processing.width
                else 0
            ),
            (
                (target_size.height - size_after_pre_processing.height) // 2
                if target_size.height > size_after_pre_processing.height
                else 0
            ),
            (
                (target_size.width - size_after_pre_processing.width + 1) // 2
                if target_size.width > size_after_pre_processing.width
                else 0
            ),
            (
                (target_size.height - size_after_pre_processing.height + 1) // 2
                if target_size.height > size_after_pre_processing.height
                else 0
            ),
        ]
        image = functional.pad(image, padding_ltrb, fill=0)
    crop_ltrb = [0, 0, 0, 0]
    if target_size.width != image.shape[3] or target_size.height != image.shape[2]:
        crop_top = int(round((image.shape[2] - target_size.height) / 2.0))
        crop_bottom = image.shape[2] - target_size.height - crop_top
        crop_left = int(round((image.shape[3] - target_size.width) / 2.0))
        crop_right = image.shape[3] - target_size.width - crop_left
        crop_ltrb = [crop_left, crop_top, crop_right, crop_bottom]
        image = functional.crop(
            image, crop_top, crop_left, target_size.height, target_size.width
        )
    if target_size.height > size_after_pre_processing.height:
        reported_padding_top = padding_ltrb[1]
        reported_padding_bottom = padding_ltrb[3]
    else:
        reported_padding_top = -crop_ltrb[1]
        reported_padding_bottom = -crop_ltrb[3]
    if target_size.width > size_after_pre_processing.width:
        reported_padding_left = padding_ltrb[0]
        reported_padding_right = padding_ltrb[2]
    else:
        reported_padding_left = -crop_ltrb[0]
        reported_padding_right = -crop_ltrb[2]
    image_metadata = PreProcessingMetadata(
        pad_left=reported_padding_left,
        pad_top=reported_padding_top,
        pad_right=reported_padding_right,
        pad_bottom=reported_padding_bottom,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=target_size,
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=static_crop_offset,
    )
    if network_input.scaling_factor is not None:
        image = image / network_input.scaling_factor
    if network_input.normalization is not None:
        if not image.is_floating_point():
            image = image.to(dtype=torch.float32)
        image = functional.normalize(
            image,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    return image.contiguous(), [image_metadata] * image.shape[0]


def handle_torch_input_preparation_fitting_longer_edge(
    image: torch.Tensor,
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    original_size: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    original_height, original_width = image.shape[2], image.shape[3]
    size_after_pre_processing = ImageDimensions(
        height=original_height, width=original_width
    )
    scale_ox = target_size.width / size_after_pre_processing.width
    scale_oy = target_size.height / size_after_pre_processing.height
    if scale_ox < scale_oy:
        actual_target_width = target_size.width
        actual_target_height = round(scale_ox * size_after_pre_processing.height)
    else:
        actual_target_width = round(scale_oy * size_after_pre_processing.width)
        actual_target_height = target_size.height
    actual_target_size = ImageDimensions(
        height=actual_target_height,
        width=actual_target_width,
    )
    if image.device.type == "cuda":
        image = image.float()
    image = torch.nn.functional.interpolate(
        image,
        [actual_target_size.height, actual_target_size.width],
        mode="bilinear",
    )
    if input_color_mode != network_input.color_mode:
        image = image[:, [2, 1, 0], :, :]
    image_metadata = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=actual_target_size,
        scale_width=actual_target_size.width / size_after_pre_processing.width,
        scale_height=actual_target_size.height / size_after_pre_processing.height,
        static_crop_offset=static_crop_offset,
    )
    if network_input.scaling_factor is not None:
        image = image / network_input.scaling_factor
    if network_input.normalization is not None:
        if not image.is_floating_point():
            image = image.to(dtype=torch.float32)
        image = functional.normalize(
            image,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    return image.contiguous(), [image_metadata] * image.shape[0]


TORCH_IMAGES_PREPARATION_HANDLERS = {
    ResizeMode.STRETCH_TO: handle_tensor_input_preparation_with_stretch,
    ResizeMode.LETTERBOX: handle_torch_input_preparation_with_letterbox,
    ResizeMode.CENTER_CROP: handle_torch_input_preparation_with_center_crop,
    ResizeMode.FIT_LONGER_EDGE: handle_torch_input_preparation_fitting_longer_edge,
    ResizeMode.LETTERBOX_REFLECT_EDGES: handle_torch_input_preparation_with_letterbox,
}


@torch.inference_mode()
def pre_process_images_tensor_list(
    images: List[torch.Tensor],
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_mode: Optional[ColorMode] = None,
    image_size_wh: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if network_input.resize_mode not in TORCH_LIST_IMAGES_PREPARATION_HANDLERS:
        raise ModelRuntimeError(
            message=f"Unsupported model input resize mode: {network_input.resize_mode}",
            help_url="https://todo",
        )
    if input_color_mode is None:
        input_color_mode = ColorMode.RGB
    target_dimensions = (
        network_input.training_input_size.width,
        network_input.training_input_size.height,
    )
    if image_size_wh is not None and image_size_wh != target_dimensions:
        if not network_input.dynamic_spatial_size_supported:
            LOGGER.warning(
                f"Requested image size: {image_size_wh} cannot be applied for model input, as model was trained with "
                f"input resolution and does not support inputs of a different shape. `image_size_wh` gets ignored."
            )
        elif isinstance(network_input.dynamic_spatial_size_mode, DivisiblePadding):
            target_dimensions = (
                make_the_value_divisible(
                    x=image_size_wh[0], by=network_input.dynamic_spatial_size_mode.value
                ),
                make_the_value_divisible(
                    x=image_size_wh[1], by=network_input.dynamic_spatial_size_mode.value
                ),
            )
        elif isinstance(network_input.dynamic_spatial_size_mode, AnySizePadding):
            target_dimensions = image_size_wh
        else:
            raise ModelRuntimeError(
                message=f"Handler for dynamic spatial mode of type {type(network_input.dynamic_spatial_size_mode)} "
                f"is not implemented.",
                help_url="",
            )
    images, static_crop_offsets, original_sizes = (
        apply_pre_processing_to_list_of_torch_image(
            images=images,
            image_pre_processing=image_pre_processing,
            network_input_channels=network_input.input_channels,
            target_device=target_device,
        )
    )
    return TORCH_LIST_IMAGES_PREPARATION_HANDLERS[network_input.resize_mode](
        images,
        network_input,
        input_color_mode,
        original_sizes,
        ImageDimensions(width=target_dimensions[0], height=target_dimensions[1]),
        static_crop_offsets,
        target_device,
    )


def apply_pre_processing_to_list_of_torch_image(
    images: List[torch.Tensor],
    image_pre_processing: ImagePreProcessing,
    network_input_channels: int,
    target_device: torch.device,
) -> Tuple[List[torch.Tensor], List[StaticCropOffset], List[ImageDimensions]]:
    result_images, result_offsets, original_sizes = [], [], []
    for image in images:
        if len(image.shape) != 3:
            raise ModelInputError(
                message="When providing List[torch.Tensor] as input, model requires tensors to have 3 dimensions.",
                help_url="https://todo",
            )
        image = image.to(target_device)
        if image.shape[0] != 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        original_sizes.append(
            ImageDimensions(height=image.shape[1], width=image.shape[2])
        )
        result_image, result_offset = apply_pre_processing_to_torch_image(
            image=image.unsqueeze(0),
            image_pre_processing=image_pre_processing,
            network_input_channels=network_input_channels,
        )
        result_images.append(result_image)
        result_offsets.append(result_offset)
    return result_images, result_offsets, original_sizes


def handle_tensor_list_input_preparation_with_stretch(
    images: List[torch.Tensor],
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    original_sizes: List[ImageDimensions],
    target_size: ImageDimensions,
    static_crop_offsets: List[StaticCropOffset],
    target_device: torch.device,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    processed = []
    images_metadata = []
    for img, offset, original_size in zip(images, static_crop_offsets, original_sizes):
        size_after_pre_processing = ImageDimensions(
            height=img.shape[2], width=img.shape[3]
        )
        if input_color_mode != network_input.color_mode:
            img = img[:, [2, 1, 0], :, :]
        if img.device.type == "cuda":
            img = img.float()
        img = torch.nn.functional.interpolate(
            img,
            size=[target_size.height, target_size.width],
            mode="bilinear",
        )
        if network_input.scaling_factor is not None:
            img = img / network_input.scaling_factor
        if network_input.normalization is not None:
            if not img.is_floating_point():
                img = img.to(dtype=torch.float32)
            img = functional.normalize(
                img,
                mean=network_input.normalization[0],
                std=network_input.normalization[1],
            )
        processed.append(img.contiguous())
        image_metadata = PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=original_size,
            size_after_pre_processing=size_after_pre_processing,
            inference_size=target_size,
            scale_width=target_size.width / size_after_pre_processing.width,
            scale_height=target_size.height / size_after_pre_processing.height,
            static_crop_offset=offset,
        )
        images_metadata.append(image_metadata)
    return torch.concat(processed, dim=0).contiguous(), images_metadata


def handle_tensor_list_input_preparation_with_letterbox(
    images: List[torch.Tensor],
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    original_sizes: List[ImageDimensions],
    target_size: ImageDimensions,
    static_crop_offsets: List[StaticCropOffset],
    target_device: torch.device,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    num_images = len(images)
    final_batch = torch.full(
        (num_images, 3, target_size.height, target_size.width),
        network_input.padding_value or 0,
        dtype=torch.float32,
        device=target_device,
    )
    original_shapes = torch.tensor(
        [[img.shape[2], img.shape[3]] for img in images], dtype=torch.float32
    )
    scale_w = target_size.width / original_shapes[:, 1]
    scale_h = target_size.height / original_shapes[:, 0]
    scales = torch.minimum(scale_w, scale_h)
    new_ws = (original_shapes[:, 1] * scales).int()
    new_hs = (original_shapes[:, 0] * scales).int()
    pad_tops = ((target_size.height - new_hs) / 2).int()
    pad_lefts = ((target_size.width - new_ws) / 2).int()
    images_metadata = []
    for i in range(num_images):
        img = images[i]
        if len(img.shape) != 4:
            raise ModelInputError(
                message="When providing List[torch.Tensor] as input, model requires tensors to have 3 dimensions.",
                help_url="https://todo",
            )
        original_size = original_sizes[i]
        size_after_pre_processing = ImageDimensions(
            height=img.shape[2], width=img.shape[3]
        )
        if input_color_mode != network_input.color_mode:
            img = img[:, [2, 1, 0], :, :]
        new_h_i, new_w_i = new_hs[i].item(), new_ws[i].item()
        if img.device.type == "cuda":
            img = img.float()
        img = torch.nn.functional.interpolate(
            img,
            size=[new_h_i, new_w_i],
            mode="bilinear",
        )
        pad_top_i, pad_left_i = pad_tops[i].item(), pad_lefts[i].item()
        final_batch[
            i, :, pad_top_i : pad_top_i + new_h_i, pad_left_i : pad_left_i + new_w_i
        ] = img
        pad_right = target_size.width - pad_left_i - new_w_i
        pad_bottom = target_size.height - pad_top_i - new_h_i
        image_metadata = PreProcessingMetadata(
            pad_left=pad_left_i,
            pad_top=pad_top_i,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            original_size=original_size,
            size_after_pre_processing=size_after_pre_processing,
            inference_size=target_size,
            scale_width=scales[i].item(),
            scale_height=scales[i].item(),
            static_crop_offset=static_crop_offsets[i],
        )
        images_metadata.append(image_metadata)
    if network_input.scaling_factor is not None:
        final_batch = final_batch / network_input.scaling_factor
    if network_input.normalization:
        if not final_batch.is_floating_point():
            final_batch = final_batch.to(dtype=torch.float32)
        final_batch = functional.normalize(
            final_batch,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    return final_batch.contiguous(), images_metadata


def handle_tensor_list_input_preparation_with_center_crop(
    images: List[torch.Tensor],
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    original_sizes: List[ImageDimensions],
    target_size: ImageDimensions,
    static_crop_offsets: List[StaticCropOffset],
    target_device: torch.device,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    result_tensors, result_metadata = [], []
    for image, offset, original_size in zip(
        images, static_crop_offsets, original_sizes
    ):
        if len(image.shape) != 4:
            raise ModelInputError(
                message="When providing List[torch.Tensor] as input, model requires tensors to have 3 dimensions.",
                help_url="https://todo",
            )
        image = image.to(target_device)
        if (
            image.shape[1] != network_input.input_channels
            and image.shape[3] == network_input.input_channels
        ):
            image = image.permute(0, 3, 1, 2)
        tensor, metadata = handle_torch_input_preparation_with_center_crop(
            image=image,
            network_input=network_input,
            input_color_mode=input_color_mode,
            original_size=original_size,
            target_size=target_size,
            static_crop_offset=offset,
        )
        result_tensors.append(tensor)
        result_metadata.append(metadata[0])
    return torch.concat(result_tensors, dim=0), result_metadata


TORCH_LIST_IMAGES_PREPARATION_HANDLERS = {
    ResizeMode.STRETCH_TO: handle_tensor_list_input_preparation_with_stretch,
    ResizeMode.LETTERBOX: handle_tensor_list_input_preparation_with_letterbox,
    ResizeMode.CENTER_CROP: handle_tensor_list_input_preparation_with_center_crop,
    ResizeMode.LETTERBOX_REFLECT_EDGES: handle_tensor_list_input_preparation_with_letterbox,
}


def pre_process_numpy_images_list(
    images: List[np.ndarray],
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_mode: Optional[ColorMode] = None,
    image_size_wh: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    result_tensors, result_metadata = [], []
    for image in images:
        tensor, metadata = pre_process_numpy_image(
            image=image,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=target_device,
            input_color_mode=input_color_mode,
            image_size_wh=image_size_wh,
        )
        result_tensors.append(tensor)
        result_metadata.extend(metadata)
    return torch.concat(result_tensors, dim=0).contiguous(), result_metadata


@torch.inference_mode()
def pre_process_numpy_image(
    image: np.ndarray,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_mode: Optional[ColorMode] = None,
    image_size_wh: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if input_color_mode is None:
        input_color_mode = ColorMode.BGR
    target_dimensions = (
        network_input.training_input_size.width,
        network_input.training_input_size.height,
    )
    if image_size_wh is not None and image_size_wh != target_dimensions:
        if not network_input.dynamic_spatial_size_supported:
            LOGGER.warning(
                f"Requested image size: {image_size_wh} cannot be applied for model input, as model was trained with "
                f"input resolution and does not support inputs of a different shape. `image_size_wh` gets ignored."
            )
        elif isinstance(network_input.dynamic_spatial_size_mode, DivisiblePadding):
            target_dimensions = (
                make_the_value_divisible(
                    x=image_size_wh[0], by=network_input.dynamic_spatial_size_mode.value
                ),
                make_the_value_divisible(
                    x=image_size_wh[1], by=network_input.dynamic_spatial_size_mode.value
                ),
            )
        elif isinstance(network_input.dynamic_spatial_size_mode, AnySizePadding):
            target_dimensions = image_size_wh
        else:
            raise ModelRuntimeError(
                message=f"Handler for dynamic spatial mode of type {type(network_input.dynamic_spatial_size_mode)} "
                f"is not implemented.",
                help_url="",
            )
    original_size = ImageDimensions(width=image.shape[1], height=image.shape[0])
    image, static_crop_offset = apply_pre_processing_to_numpy_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input_channels=network_input.input_channels,
        input_color_mode=input_color_mode,
    )
    if network_input.resize_mode not in NUMPY_IMAGES_PREPARATION_HANDLERS:
        raise ModelRuntimeError(
            message=f"Unsupported model input resize mode: {network_input.resize_mode}",
            help_url="https://todo",
        )
    return NUMPY_IMAGES_PREPARATION_HANDLERS[network_input.resize_mode](
        image,
        network_input,
        target_device,
        input_color_mode,
        original_size,
        ImageDimensions(width=target_dimensions[0], height=target_dimensions[1]),
        static_crop_offset,
    )


def apply_pre_processing_to_numpy_image(
    image: np.ndarray,
    image_pre_processing: ImagePreProcessing,
    network_input_channels: int,
    input_color_mode: Optional[ColorMode] = None,
) -> Tuple[np.ndarray, StaticCropOffset]:
    if input_color_mode is None:
        input_color_mode = ColorMode.BGR
    static_crop_offset = StaticCropOffset(
        offset_x=0,
        offset_y=0,
        crop_width=image.shape[1],
        crop_height=image.shape[0],
    )
    if image_pre_processing.static_crop and image_pre_processing.static_crop.enabled:
        image, static_crop_offset = apply_static_crop_to_numpy_image(
            image=image,
            config=image_pre_processing.static_crop,
        )
    if image_pre_processing.grayscale and image_pre_processing.grayscale.enabled:
        mode = (
            cv2.COLOR_BGR2GRAY
            if input_color_mode is ColorMode.BGR
            else cv2.COLOR_RGB2GRAY
        )
        image = cv2.cvtColor(image, mode)
        image = np.stack([image] * network_input_channels, axis=2)
    if image_pre_processing.contrast and image_pre_processing.contrast.enabled:
        if (
            image_pre_processing.contrast.type
            not in CONTRAST_ADJUSTMENT_METHODS_FOR_NUMPY
        ):
            raise ModelRuntimeError(
                message=f"Unsupported image contrast adjustment type: {image_pre_processing.contrast.type.value}",
                help_url="https://todo",
            )
        image = CONTRAST_ADJUSTMENT_METHODS_FOR_NUMPY[
            image_pre_processing.contrast.type
        ](image)
    return image, static_crop_offset


def apply_static_crop_to_numpy_image(
    image: np.ndarray, config: StaticCrop
) -> Tuple[np.ndarray, StaticCropOffset]:
    width, height = image.shape[1], image.shape[0]
    x_min = int(config.x_min / 100 * width)
    y_min = int(config.y_min / 100 * height)
    x_max = int(config.x_max / 100 * width)
    y_max = int(config.y_max / 100 * height)
    result_image = image[y_min:y_max, x_min:x_max]
    return result_image, StaticCropOffset(
        offset_x=x_min,
        offset_y=y_min,
        crop_width=result_image.shape[1],
        crop_height=result_image.shape[0],
    )


def apply_adaptive_equalization_to_numpy_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255
    image_adapted = exposure.equalize_adapthist(image, clip_limit=0.03) * 255
    return image_adapted.astype(np.uint8)


def apply_contrast_stretching_to_numpy_image(image: np.ndarray) -> np.ndarray:
    p2 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)
    return exposure.rescale_intensity(image, in_range=(p2, p98))


def apply_histogram_equalization_to_numpy_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255
    image_equalized = exposure.equalize_hist(image) * 255
    return image_equalized.astype(np.uint8)


CONTRAST_ADJUSTMENT_METHODS_FOR_NUMPY = {
    ContrastType.ADAPTIVE_EQUALIZATION: apply_adaptive_equalization_to_numpy_image,
    ContrastType.CONTRAST_STRETCHING: apply_contrast_stretching_to_numpy_image,
    ContrastType.HISTOGRAM_EQUALIZATION: apply_histogram_equalization_to_numpy_image,
}


def handle_numpy_input_preparation_with_stretch(
    image: np.ndarray,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_mode: ColorMode,
    original_size: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    size_after_pre_processing = ImageDimensions(
        height=image.shape[0], width=image.shape[1]
    )
    resized_image = cv2.resize(image, (target_size.width, target_size.height))
    tensor = torch.from_numpy(resized_image).to(device=target_device)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.permute(0, 3, 1, 2)
    if input_color_mode != network_input.color_mode:
        tensor = tensor[:, [2, 1, 0], :, :]
    if network_input.scaling_factor is not None:
        tensor = tensor / network_input.scaling_factor
    if network_input.normalization:
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        tensor = functional.normalize(
            tensor,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    image_metadata = PreProcessingMetadata(
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
    return tensor.contiguous(), [image_metadata]


def handle_numpy_input_preparation_with_letterbox(
    image: np.ndarray,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_mode: ColorMode,
    original_size: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    padding_value = network_input.padding_value or 0
    original_height, original_width = image.shape[0], image.shape[1]
    size_after_pre_processing = ImageDimensions(
        height=original_height, width=original_width
    )
    scale_w = target_size.width / original_width
    scale_h = target_size.height / original_height
    scale = min(scale_w, scale_h)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    pad_top = int((target_size.height - new_height) / 2)
    pad_left = int((target_size.width - new_width) / 2)
    scaled_image = cv2.resize(image, (new_width, new_height))
    scaled_image_tensor = torch.from_numpy(scaled_image).to(target_device)
    scaled_image_tensor = scaled_image_tensor.permute(2, 0, 1)
    final_batch = torch.full(
        (
            1,
            image.shape[2],
            target_size.height,
            target_size.width,
        ),
        padding_value,
        dtype=torch.float32,
        device=target_device,
    )
    final_batch[
        0, :, pad_top : pad_top + new_height, pad_left : pad_left + new_width
    ] = scaled_image_tensor
    if input_color_mode != network_input.color_mode:
        final_batch = final_batch[:, [2, 1, 0], :, :]
    pad_right = target_size.width - pad_left - new_width
    pad_bottom = target_size.height - pad_top - new_height
    image_metadata = PreProcessingMetadata(
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=target_size,
        scale_width=scale,
        scale_height=scale,
        static_crop_offset=static_crop_offset,
    )
    if network_input.scaling_factor is not None:
        final_batch = final_batch / network_input.scaling_factor
    if network_input.normalization is not None:
        if not final_batch.is_floating_point():
            final_batch = final_batch.to(dtype=torch.float32)
        final_batch = functional.normalize(
            final_batch,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    return final_batch.contiguous(), [image_metadata]


def handle_numpy_input_preparation_with_center_crop(
    image: np.ndarray,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_mode: ColorMode,
    original_size: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    original_height, original_width = image.shape[0], image.shape[1]
    size_after_pre_processing = ImageDimensions(
        height=original_height, width=original_width
    )
    canvas = np.zeros((target_size.height, target_size.width, 3), dtype=np.uint8)
    canvas_ox_padding = max(target_size.width - image.shape[1], 0)
    canvas_padding_left = canvas_ox_padding // 2
    canvas_padding_right = canvas_ox_padding - canvas_padding_left
    canvas_oy_padding = max(target_size.height - image.shape[0], 0)
    canvas_padding_top = canvas_oy_padding // 2
    canvas_padding_bottom = canvas_oy_padding - canvas_padding_top
    original_image_ox_padding = max(image.shape[1] - target_size.width, 0)
    original_image_padding_left = original_image_ox_padding // 2
    original_image_padding_right = (
        original_image_ox_padding - original_image_padding_left
    )
    original_image_oy_padding = max(image.shape[0] - target_size.height, 0)
    original_image_padding_top = original_image_oy_padding // 2
    original_image_padding_bottom = (
        original_image_oy_padding - original_image_padding_top
    )
    canvas[
        canvas_padding_top : canvas.shape[0] - canvas_padding_bottom,
        canvas_padding_left : canvas.shape[1] - canvas_padding_right,
    ] = image[
        original_image_padding_top : image.shape[0] - original_image_padding_bottom,
        original_image_padding_left : image.shape[1] - original_image_padding_right,
    ]
    if canvas.shape[0] > image.shape[0]:
        reported_padding_top = canvas_padding_top
        reported_padding_bottom = canvas_padding_bottom
    else:
        reported_padding_top = -original_image_padding_top
        reported_padding_bottom = -original_image_padding_bottom
    if canvas.shape[1] > image.shape[1]:
        reported_padding_left = canvas_padding_left
        reported_padding_right = canvas_padding_right
    else:
        reported_padding_left = -original_image_padding_left
        reported_padding_right = -original_image_padding_right
    image_metadata = PreProcessingMetadata(
        pad_left=reported_padding_left,
        pad_top=reported_padding_top,
        pad_right=reported_padding_right,
        pad_bottom=reported_padding_bottom,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=target_size,
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=static_crop_offset,
    )
    tensor = torch.from_numpy(canvas).to(device=target_device)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.permute(0, 3, 1, 2)
    if input_color_mode != network_input.color_mode:
        tensor = tensor[:, [2, 1, 0], :, :]
    if network_input.scaling_factor is not None:
        tensor = tensor / network_input.scaling_factor
    if network_input.normalization:
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        tensor = functional.normalize(
            tensor,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    return tensor.contiguous(), [image_metadata]


def handle_numpy_input_preparation_fitting_longer_edge(
    image: np.ndarray,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    input_color_mode: ColorMode,
    original_size: ImageDimensions,
    target_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    original_height, original_width = image.shape[0], image.shape[1]
    size_after_pre_processing = ImageDimensions(
        height=original_height, width=original_width
    )
    scale_ox = target_size.width / size_after_pre_processing.width
    scale_oy = target_size.height / size_after_pre_processing.height
    if scale_ox < scale_oy:
        actual_target_width = target_size.width
        actual_target_height = round(scale_ox * size_after_pre_processing.height)
    else:
        actual_target_width = round(scale_oy * size_after_pre_processing.width)
        actual_target_height = target_size.height
    actual_target_size = ImageDimensions(
        height=actual_target_height,
        width=actual_target_width,
    )
    scaled_image = cv2.resize(
        image, (actual_target_size.width, actual_target_size.height)
    )
    image_metadata = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=actual_target_size,
        scale_width=actual_target_size.width / size_after_pre_processing.width,
        scale_height=actual_target_size.height / size_after_pre_processing.height,
        static_crop_offset=static_crop_offset,
    )
    tensor = torch.from_numpy(scaled_image).to(device=target_device)
    tensor = torch.unsqueeze(tensor, 0)
    tensor = tensor.permute(0, 3, 1, 2)
    if input_color_mode != network_input.color_mode:
        tensor = tensor[:, [2, 1, 0], :, :]
    if network_input.scaling_factor is not None:
        tensor = tensor / network_input.scaling_factor
    if network_input.normalization:
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        tensor = functional.normalize(
            tensor,
            mean=network_input.normalization[0],
            std=network_input.normalization[1],
        )
    return tensor.contiguous(), [image_metadata]


NUMPY_IMAGES_PREPARATION_HANDLERS = {
    ResizeMode.STRETCH_TO: handle_numpy_input_preparation_with_stretch,
    ResizeMode.LETTERBOX: handle_numpy_input_preparation_with_letterbox,
    ResizeMode.CENTER_CROP: handle_numpy_input_preparation_with_center_crop,
    ResizeMode.FIT_LONGER_EDGE: handle_numpy_input_preparation_fitting_longer_edge,
    ResizeMode.LETTERBOX_REFLECT_EDGES: handle_numpy_input_preparation_with_letterbox,
}


def extract_input_images_dimensions(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
) -> List[ImageDimensions]:
    if isinstance(images, np.ndarray):
        return [ImageDimensions(height=images.shape[0], width=images.shape[1])]
    if isinstance(images, torch.Tensor):
        if len(images.shape) == 3:
            images = torch.unsqueeze(images, dim=0)
        image_dimensions = []
        for image in images:
            image_dimensions.append(
                ImageDimensions(height=image.shape[1], width=image.shape[2])
            )
        return image_dimensions
    if not isinstance(images, list):
        raise ModelInputError(
            message="Pre-processing supports only np.array or torch.Tensor or list of above.",
            help_url="https://todo",
        )
    if not len(images):
        raise ModelInputError(
            message="Detected empty input to the model", help_url="https://todo"
        )
    if isinstance(images[0], np.ndarray):
        return [ImageDimensions(height=i.shape[0], width=i.shape[1]) for i in images]
    if isinstance(images[0], torch.Tensor):
        image_dimensions = []
        for image in images:
            image_dimensions.append(
                ImageDimensions(height=image.shape[1], width=image.shape[2])
            )
        return image_dimensions
    raise ModelInputError(
        message=f"Detected unknown input batch element: {type(images[0])}",
        help_url="https://todo",
    )


def images_to_pillow(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat] = None,
    model_color_format: ColorFormat = "rgb",
) -> Tuple[List[Image], List[ImageDimensions]]:
    if isinstance(images, np.ndarray):
        input_color_format = input_color_format or "bgr"
        if input_color_format != model_color_format:
            images = images[:, :, ::-1]
        h, w = images.shape[:2]
        return [PIL.Image.fromarray(images)], [ImageDimensions(height=h, width=w)]
    if isinstance(images, torch.Tensor):
        input_color_format = input_color_format or "rgb"
        if len(images.shape) == 3:
            images = torch.unsqueeze(images, dim=0)
        if input_color_format != model_color_format:
            images = images[:, [2, 1, 0], :, :]
        result = []
        dimensions = []
        for image in images:
            np_image = image.permute(1, 2, 0).cpu().numpy()
            result.append(PIL.Image.fromarray(np_image))
            dimensions.append(
                ImageDimensions(height=np_image.shape[0], width=np_image.shape[1])
            )
        return result, dimensions
    if not isinstance(images, list):
        raise ModelInputError(
            message="Pre-processing supports only np.array or torch.Tensor or list of above.",
            help_url="https://todo",
        )
    if not len(images):
        raise ModelInputError(
            message="Detected empty input to the model", help_url="https://todo"
        )
    if isinstance(images[0], np.ndarray):
        input_color_format = input_color_format or "bgr"
        if input_color_format != model_color_format:
            images = [i[:, :, ::-1] for i in images]
        dimensions = [
            ImageDimensions(height=i.shape[0], width=i.shape[1]) for i in images
        ]
        images = [PIL.Image.fromarray(i) for i in images]
        return images, dimensions
    if isinstance(images[0], torch.Tensor):
        result = []
        dimensions = []
        input_color_format = input_color_format or "rgb"
        for image in images:
            if input_color_format != model_color_format:
                image = image[[2, 1, 0], :, :]
            np_image = image.permute(1, 2, 0).cpu().numpy()
            result.append(PIL.Image.fromarray(np_image))
            dimensions.append(
                ImageDimensions(height=np_image.shape[0], width=np_image.shape[1])
            )
        return result, dimensions
    raise ModelInputError(
        message=f"Detected unknown input batch element: {type(images[0])}",
        help_url="https://todo",
    )


def make_the_value_divisible(x: int, by: int) -> int:
    return math.ceil(x / by) * by
