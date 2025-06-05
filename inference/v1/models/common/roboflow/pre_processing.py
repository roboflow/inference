from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL
import torch
from PIL.Image import Image
from torchvision.transforms import functional

from inference.v1.entities import ColorFormat, ImageDimensions
from inference.v1.errors import ModelRuntimeError
from inference.v1.models.common.roboflow.model_packages import (
    PreProcessingConfig,
    PreProcessingMetadata,
    PreProcessingMode,
)


def pre_process_network_input(
    images: torch.Tensor,
    pre_processing_config: PreProcessingConfig,
    expected_network_color_format: ColorFormat,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    rescaling_constant: Optional[float] = 255.0,
    normalization: Optional[Tuple[List[float], List[float]]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if isinstance(images, np.ndarray):
        return pre_process_numpy_image(
            image=images,
            pre_processing_config=pre_processing_config,
            input_color_format=input_color_format,
            expected_network_color_format=expected_network_color_format,
            target_device=target_device,
            rescaling_constant=rescaling_constant,
            normalization=normalization,
        )
    if isinstance(images, torch.Tensor):
        return pre_process_images_tensor(
            images=images,
            pre_processing_config=pre_processing_config,
            input_color_format=input_color_format,
            expected_network_color_format=expected_network_color_format,
            target_device=target_device,
            rescaling_constant=rescaling_constant,
            normalization=normalization,
        )
    if not isinstance(images, list):
        raise ModelRuntimeError(
            "Pre-processing supports only np.array or torch.Tensor or list of above."
        )
    if not len(images):
        raise ModelRuntimeError("Detected empty input to the model")
    if isinstance(images[0], np.ndarray):
        return pre_process_numpy_images_list(
            images=images,
            pre_processing_config=pre_processing_config,
            input_color_format=input_color_format,
            expected_network_color_format=expected_network_color_format,
            target_device=target_device,
            normalization_constant=rescaling_constant,
            normalization=normalization,
        )
    if isinstance(images[0], torch.Tensor):
        return pre_process_images_tensor_list(
            images=images,
            pre_processing_config=pre_processing_config,
            input_color_format=input_color_format,
            expected_network_color_format=expected_network_color_format,
            target_device=target_device,
            rescaling_constant=rescaling_constant,
            normalization=normalization,
        )
    raise ModelRuntimeError(f"Detected unknown input batch element: {type(images[0])}")


@torch.inference_mode()
def pre_process_images_tensor(
    images: torch.Tensor,
    pre_processing_config: PreProcessingConfig,
    expected_network_color_format: ColorFormat,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    rescaling_constant: Optional[float] = 255.0,
    normalization: Optional[Tuple[List[float], List[float]]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if (
        pre_processing_config.mode is PreProcessingMode.NONE
        or pre_processing_config.target_size is None
    ):
        raise ModelRuntimeError(
            "Could not pre-process data before model inference - pre-processing configuration "
            "does not specify input resizing."
        )
    if input_color_format is None:
        input_color_format = "rgb"
    if images.device != target_device:
        images = images.to(target_device)
    if len(images.shape) == 3:
        images = torch.unsqueeze(images, 0)
    if images.shape[1] != 3 and images.shape[3] == 3:
        images = images.permute(0, 3, 1, 2)
    original_size = ImageDimensions(height=images.shape[2], width=images.shape[3])
    if pre_processing_config.mode is PreProcessingMode.STRETCH:
        if images.device.type == "cuda":
            images = images.float()
        images = torch.nn.functional.interpolate(
            images,
            [
                pre_processing_config.target_size.height,
                pre_processing_config.target_size.width,
            ],
            mode="bilinear",
        )
        if input_color_format != expected_network_color_format:
            images = images[:, [2, 1, 0], :, :]
        if rescaling_constant is not None:
            images = images / rescaling_constant
        if normalization:
            images = functional.normalize(
                images, mean=normalization[0], std=normalization[1]
            )
        metadata = PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=original_size,
            inference_size=pre_processing_config.target_size,
            scale_width=pre_processing_config.target_size.width / original_size.width,
            scale_height=pre_processing_config.target_size.height
            / original_size.height,
        )
        return images.contiguous(), [metadata] * images.shape[0]
    if pre_processing_config.mode is not PreProcessingMode.LETTERBOX:
        raise ModelRuntimeError(
            f"Cannot find implementation for pre-processing operation: {pre_processing_config.mode}"
        )
    original_height, original_width = images.shape[2], images.shape[3]
    scale_w = pre_processing_config.target_size.width / original_width
    scale_h = pre_processing_config.target_size.height / original_height
    scale = min(scale_w, scale_h)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    pad_top = int((pre_processing_config.target_size.height - new_height) / 2)
    pad_left = int((pre_processing_config.target_size.width - new_width) / 2)
    if images.device.type == "cuda":
        images = images.float()
    images = torch.nn.functional.interpolate(
        images,
        [new_height, new_width],
        mode="bilinear",
    )
    if input_color_format != expected_network_color_format:
        images = images[:, [2, 1, 0], :, :]
    final_batch = torch.full(
        (
            images.shape[0],
            images.shape[1],
            pre_processing_config.target_size.height,
            pre_processing_config.target_size.width,
        ),
        pre_processing_config.padding_value,
        dtype=torch.float32,
        device=target_device,
    )
    final_batch[
        :, :, pad_top : pad_top + new_height, pad_left : pad_left + new_width
    ] = images
    pad_right = pre_processing_config.target_size.width - pad_left - new_width
    pad_bottom = pre_processing_config.target_size.height - pad_top - new_height
    metadata = PreProcessingMetadata(
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        original_size=original_size,
        inference_size=pre_processing_config.target_size,
        scale_width=scale,
        scale_height=scale,
    )
    if rescaling_constant is not None:
        final_batch = final_batch / rescaling_constant
    if normalization:
        final_batch = functional.normalize(
            final_batch, mean=normalization[0], std=normalization[1]
        )
    return final_batch.contiguous(), [metadata] * final_batch.shape[0]


@torch.inference_mode()
def pre_process_images_tensor_list(
    images: List[torch.Tensor],
    pre_processing_config: PreProcessingConfig,
    expected_network_color_format: ColorFormat,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    rescaling_constant: Optional[float] = 255.0,
    normalization: Optional[Tuple[List[float], List[float]]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if (
        pre_processing_config.mode is PreProcessingMode.NONE
        or pre_processing_config.target_size is None
    ):
        raise ModelRuntimeError(
            "Could not pre-process data before model inference - pre-processing configuration "
            "does not specify input resizing."
        )
    if input_color_format is None:
        input_color_format = "rgb"
    target_height = pre_processing_config.target_size.height
    target_width = pre_processing_config.target_size.width
    if pre_processing_config.mode is PreProcessingMode.STRETCH:
        processed = []
        images_metadata = []
        for img in images:
            if len(img.shape) != 3:
                raise ModelRuntimeError(
                    "When providing List[torch.Tensor] as input, model requires tensors to have 3 dimensions."
                )
            if img.device != target_device:
                img = img.to(target_device)
            if img.shape[0] != 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            original_size = ImageDimensions(height=img.shape[1], width=img.shape[2])
            if input_color_format != expected_network_color_format:
                img = img[[2, 1, 0], :, :]
            img = img.unsqueeze(0)
            img = functional.resize(
                img,
                [target_height, target_width],
                interpolation=functional.InterpolationMode.BILINEAR,
            )
            if rescaling_constant is not None:
                img = img / rescaling_constant
            if normalization:
                img = functional.normalize(
                    img, mean=normalization[0], std=normalization[1]
                )
            processed.append(img.contiguous())
            image_metadata = PreProcessingMetadata(
                pad_left=0,
                pad_top=0,
                pad_right=0,
                pad_bottom=0,
                original_size=original_size,
                inference_size=pre_processing_config.target_size,
                scale_width=pre_processing_config.target_size.width
                / original_size.width,
                scale_height=pre_processing_config.target_size.height
                / original_size.height,
            )
            images_metadata.append(image_metadata)
        return torch.concat(processed, dim=0).contiguous(), images_metadata
    if pre_processing_config.mode is PreProcessingMode.LETTERBOX:
        target_h, target_w = (
            pre_processing_config.target_size.height,
            pre_processing_config.target_size.width,
        )
        num_images = len(images)
        final_batch = torch.full(
            (num_images, 3, target_h, target_w),
            pre_processing_config.padding_value or 0,
            dtype=torch.float32,
            device=target_device,
        )
        original_shapes = torch.tensor(
            [[img.shape[1], img.shape[2]] for img in images], dtype=torch.float32
        )
        scale_w = target_w / original_shapes[:, 1]
        scale_h = target_h / original_shapes[:, 0]
        scales = torch.minimum(scale_w, scale_h)
        new_ws = (original_shapes[:, 1] * scales).int()
        new_hs = (original_shapes[:, 0] * scales).int()
        pad_tops = ((target_h - new_hs) / 2).int()
        pad_lefts = ((target_w - new_ws) / 2).int()
        images_metadata = []
        for i in range(num_images):
            image_hwc = images[i].to(target_device)  # Ensure on correct device
            if image_hwc.dtype != torch.float32:
                image_hwc = image_hwc.float()
            if input_color_format != expected_network_color_format:
                image_hwc = image_hwc[..., [2, 1, 0]]
            if image_hwc.shape[0] != 3 and image_hwc.shape[-1] == 3:
                image_hwc = image_hwc.permute(2, 0, 1)
            original_size = ImageDimensions(
                height=image_hwc.shape[1], width=image_hwc.shape[2]
            )
            new_h_i, new_w_i = new_hs[i].item(), new_ws[i].item()
            resized_chw = functional.resize(
                image_hwc,
                [new_h_i, new_w_i],
                interpolation=functional.InterpolationMode.BILINEAR,
            )
            pad_top_i, pad_left_i = pad_tops[i].item(), pad_lefts[i].item()
            final_batch[
                i, :, pad_top_i : pad_top_i + new_h_i, pad_left_i : pad_left_i + new_w_i
            ] = resized_chw
            pad_right = pre_processing_config.target_size.width - pad_left_i - new_w_i
            pad_bottom = pre_processing_config.target_size.height - pad_top_i - new_h_i
            image_metadata = PreProcessingMetadata(
                pad_left=pad_left_i,
                pad_top=pad_top_i,
                pad_right=pad_right,
                pad_bottom=pad_bottom,
                original_size=original_size,
                inference_size=pre_processing_config.target_size,
                scale_width=scales[i].item(),
                scale_height=scales[i].item(),
            )
            images_metadata.append(image_metadata)
        if rescaling_constant is not None:
            final_batch = final_batch / rescaling_constant
        if normalization:
            final_batch = functional.normalize(
                final_batch, mean=normalization[0], std=normalization[1]
            )
        return final_batch.contiguous(), images_metadata
    raise ModelRuntimeError(
        f"Unsupported pre-processing mode: {pre_processing_config.mode}"
    )


def pre_process_numpy_images_list(
    images: List[np.ndarray],
    pre_processing_config: PreProcessingConfig,
    expected_network_color_format: ColorFormat,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    normalization_constant: float = 255.0,
    normalization: Optional[Tuple[List[float], List[float]]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    result_tensors, result_metadata = [], []
    for image in images:
        tensor, metadata = pre_process_numpy_image(
            image=image,
            pre_processing_config=pre_processing_config,
            expected_network_color_format=expected_network_color_format,
            target_device=target_device,
            input_color_format=input_color_format,
            rescaling_constant=normalization_constant,
            normalization=normalization,
        )
        result_tensors.append(tensor)
        result_metadata.extend(metadata)
    return torch.concat(result_tensors, dim=0).contiguous(), result_metadata


@torch.inference_mode()
def pre_process_numpy_image(
    image: np.ndarray,
    pre_processing_config: PreProcessingConfig,
    expected_network_color_format: ColorFormat,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
    rescaling_constant: Optional[float] = 255.0,
    normalization: Optional[Tuple[List[float], List[float]]] = None,
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    if (
        pre_processing_config.mode is PreProcessingMode.NONE
        or pre_processing_config.target_size is None
    ):
        raise ModelRuntimeError(
            "Could not pre-process data before model inference - pre-processing configuration "
            "does not specify input resizing."
        )
    if input_color_format is None:
        input_color_format = "bgr"
    original_size = ImageDimensions(height=image.shape[0], width=image.shape[1])
    if pre_processing_config.mode is PreProcessingMode.STRETCH:
        resized_image = cv2.resize(
            image,
            (
                pre_processing_config.target_size.width,
                pre_processing_config.target_size.height,
            ),
        )
        tensor = torch.from_numpy(resized_image).to(device=target_device)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = tensor.permute(0, 3, 1, 2)
        if input_color_format != expected_network_color_format:
            tensor = tensor[:, [2, 1, 0], :, :]
        if rescaling_constant is not None:
            tensor = tensor / rescaling_constant
        if normalization:
            tensor = functional.normalize(
                tensor, mean=normalization[0], std=normalization[1]
            )
        image_metadata = PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=original_size,
            inference_size=pre_processing_config.target_size,
            scale_width=pre_processing_config.target_size.width / original_size.width,
            scale_height=pre_processing_config.target_size.height
            / original_size.height,
        )
        return tensor.contiguous(), [image_metadata]
    if pre_processing_config.mode is PreProcessingMode.LETTERBOX:
        original_height, original_width = image.shape[0], image.shape[1]
        scale_w = pre_processing_config.target_size.width / original_width
        scale_h = pre_processing_config.target_size.height / original_height
        scale = min(scale_w, scale_h)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        pad_top = int((pre_processing_config.target_size.height - new_height) / 2)
        pad_left = int((pre_processing_config.target_size.width - new_width) / 2)
        scaled_image = cv2.resize(image, (new_width, new_height))
        scaled_image_tensor = torch.from_numpy(scaled_image).to(target_device)
        scaled_image_tensor = scaled_image_tensor.permute(2, 0, 1)
        final_batch = torch.full(
            (
                1,
                image.shape[2],
                pre_processing_config.target_size.height,
                pre_processing_config.target_size.width,
            ),
            pre_processing_config.padding_value,
            dtype=torch.float32,
            device=target_device,
        )
        final_batch[
            0, :, pad_top : pad_top + new_height, pad_left : pad_left + new_width
        ] = scaled_image_tensor
        if input_color_format != expected_network_color_format:
            final_batch = final_batch[:, [2, 1, 0], :, :]
        pad_right = pre_processing_config.target_size.width - pad_left - new_width
        pad_bottom = pre_processing_config.target_size.height - pad_top - new_height
        image_metadata = PreProcessingMetadata(
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            original_size=original_size,
            inference_size=pre_processing_config.target_size,
            scale_width=scale,
            scale_height=scale,
        )
        if rescaling_constant is not None:
            final_batch = final_batch / rescaling_constant
        if normalization:
            final_batch = functional.normalize(
                final_batch, mean=normalization[0], std=normalization[1]
            )
        return final_batch.contiguous(), [image_metadata]
    raise ModelRuntimeError(
        f"Unsupported pre-processing mode: {pre_processing_config.mode}"
    )


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
                ImageDimensions(height=image.shape[2], width=image.shape[3])
            )
        return image_dimensions
    if not isinstance(images, list):
        raise ModelRuntimeError(
            "Pre-processing supports only np.array or torch.Tensor or list of above."
        )
    if not len(images):
        raise ModelRuntimeError("Detected empty input to the model")
    if isinstance(images[0], np.ndarray):
        return [ImageDimensions(height=i.shape[0], width=i.shape[1]) for i in images]
    if isinstance(images[0], torch.Tensor):
        image_dimensions = []
        for image in images:
            image_dimensions.append(
                ImageDimensions(height=image.shape[2], width=image.shape[3])
            )
        return image_dimensions
    raise ModelRuntimeError(f"Detected unknown input batch element: {type(images[0])}")


def images_to_pillow(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat] = None,
    model_color_format: ColorFormat = "rgb",
) -> Tuple[List[Image], List[ImageDimensions]]:
    # Fast type dispatch to correct backend
    if isinstance(images, np.ndarray):
        input_color_format = input_color_format or "bgr"
        return _ndarray_to_pillow(images, input_color_format, model_color_format)
    if isinstance(images, torch.Tensor):
        input_color_format = input_color_format or "rgb"
        return _torch_to_pillow(images, input_color_format, model_color_format)
    if not isinstance(images, list):
        raise ModelRuntimeError(
            "Pre-processing supports only np.array or torch.Tensor or list of above."
        )
    if not len(images):
        raise ModelRuntimeError("Detected empty input to the model")
    if isinstance(images[0], np.ndarray):
        input_color_format = input_color_format or "bgr"
        return _list_ndarray_to_pillow(images, input_color_format, model_color_format)
    if isinstance(images[0], torch.Tensor):
        input_color_format = input_color_format or "rgb"
        return _list_torch_to_pillow(images, input_color_format, model_color_format)
    raise ModelRuntimeError(f"Detected unknown input batch element: {type(images[0])}")


def _convert_bgr_to_rgb_np(imgs):
    """Convert one or batched BGR ndarray(s) to RGB in-place when possible."""
    if imgs.ndim == 3:
        return (
            imgs[:, :, ::-1].copy()
            if not imgs.flags["C_CONTIGUOUS"]
            else imgs[:, :, ::-1]
        )
    # Possible support for batch if used in future
    raise ValueError("Only 3D ndarray supported for color conversion.")


def _ndarray_to_pillow(images, input_color_format, model_color_format):
    if input_color_format != model_color_format:
        images = _convert_bgr_to_rgb_np(images)
    h, w = images.shape[:2]
    return [PIL.Image.fromarray(images)], [ImageDimensions(height=h, width=w)]


def _torch_to_pillow(images, input_color_format, model_color_format):
    # Always work on 4D batch for unified handling; don't trigger unnecessary copies
    if images.dim() == 3:
        images = images.unsqueeze(0)

    # Efficient BGR<->RGB conversion via indexing if needed
    if input_color_format != model_color_format:
        images = images.index_select(1, torch.tensor([2, 1, 0], device=images.device))

    # Pre-allocate lists for batch conversion, avoid append in loop
    num_imgs = images.shape[0]
    pil_images = [None] * num_imgs
    dims = [None] * num_imgs

    # Batched transfer to cpu avoids re-launching kernel for each image (if not on cpu)
    if not images.device.type == "cpu":
        images = images.cpu()

    np_images = images.permute(0, 2, 3, 1).contiguous().numpy()  # (N,H,W,C)

    for i in range(num_imgs):
        arr = np_images[i]
        pil_images[i] = PIL.Image.fromarray(arr)
        dims[i] = ImageDimensions(height=arr.shape[0], width=arr.shape[1])
    return pil_images, dims


def _list_ndarray_to_pillow(images, input_color_format, model_color_format):
    if input_color_format != model_color_format:
        images = [i[:, :, ::-1] for i in images]
    dims = [ImageDimensions(height=i.shape[0], width=i.shape[1]) for i in images]
    # List comprehension is fastest for this use
    pil_images = [PIL.Image.fromarray(i) for i in images]
    return pil_images, dims


def _list_torch_to_pillow(images, input_color_format, model_color_format):
    num_imgs = len(images)
    pil_images = [None] * num_imgs
    dims = [None] * num_imgs
    idx = None
    if input_color_format != model_color_format:
        idx = torch.tensor([2, 1, 0])
    for i, img in enumerate(images):
        if idx is not None:
            img = img.index_select(0, idx.to(img.device))
        arr = img.permute(1, 2, 0).cpu().numpy()
        pil_images[i] = PIL.Image.fromarray(arr)
        dims[i] = ImageDimensions(height=arr.shape[0], width=arr.shape[1])
    return pil_images, dims
