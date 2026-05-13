"""RFDETR-specific preprocessing matching the training pipeline.

numpy / PIL inputs:
    [optional cv2 dataset-version resize] → PIL F.resize → F.to_tensor → F.normalize

The cv2 dataset-version resize runs first when `resize_mode != STRETCH_TO`
and `dataset_version_resize_dimensions` is set; the PIL stretch then takes
the result to `training_input_size`.

torch.Tensor inputs (advanced caller, float CHW [0, 1]):
    tensor F.resize → F.normalize

Triton path: for the common case (single-stage resize, no contrast) 
we invoke a single PIL-exact Triton kernel that writes the normalized
fp32 tensor directly to `target_device`.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from inference_models.configuration import USE_TRITON_FOR_PREPROCESSING
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

try:
    from inference_models.models.rfdetr.triton_preprocess import (
        TRITON_AVAILABLE as _TRITON_AVAILABLE,
        build_resample_tables,
        triton_preprocess_rfdetr_stretch, ResampleTables,
)
except ImportError:
    _TRITON_AVAILABLE = False
    build_resample_tables = None
    triton_preprocess_rfdetr_stretch = None

# Resample tables for PIL identical image resize [Key: (device_str, src_h, src_w, th, tw)]
RESAMPLE_TABLES_CACHE: Dict[Tuple[str, int, int, int, int], ResampleTables] = {}


def get_resample_tables(
    device: torch.device, src_h: int, src_w: int, th: int, tw: int
)->ResampleTables:
    key = (str(device), src_h, src_w, th, tw)
    t = RESAMPLE_TABLES_CACHE.get(key)
    if t is None:
        t = build_resample_tables(
            src_h=src_h, src_w=src_w, target_h=th, target_w=tw, device=device
        )
        RESAMPLE_TABLES_CACHE[key] = t
    return t


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

    if triton_path_eligible(
        image_list=image_list,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=target_device,
        pre_processing_overrides=pre_processing_overrides,
    ):
        return triton_path_preprocess(
            image_list=image_list,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_size=target_size,
            target_device=target_device,
            input_color_mode=input_color_mode,
            pre_processing_overrides=pre_processing_overrides,
        )

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
                _tensor_to_hwc_uint8(img)
                if isinstance(img, torch.Tensor)
                else _ensure_hwc_uint8(img)
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


def triton_path_eligible(
    image_list: List[Union[np.ndarray, torch.Tensor]],
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_device: torch.device,
    pre_processing_overrides: Optional[PreProcessingOverrides],
) -> bool:
    """True when every item in `image_list` can go through the fused Triton
    kernel. Predicate is intentionally conservative — any miss → PIL path."""
    if not USE_TRITON_FOR_PREPROCESSING:
        return False
    if not _TRITON_AVAILABLE:
        return False
    # Kernel runs on CUDA.
    if target_device.type != "cuda":
        return False
    # grayscale / contrast aren't implemented in the kernel; static_crop is
    # (as a load-time offset + effective-dims substitution).
    ipp = image_pre_processing
    if (ipp.contrast is not None and ipp.contrast.enabled) or (
        ipp.grayscale is not None and ipp.grayscale.enabled
    ):
        return False
    # Two-stage dataset-version resize isn't in the kernel.
    ni = network_input
    if _needs_two_step_resize(ni):
        return False
    if ni.input_channels != 3:
        return False
    if ni.scaling_factor not in (None, 255):
        return False
    if ni.normalization is None:
        return False
    if ni.resize_mode not in (
        ResizeMode.STRETCH_TO,
        ResizeMode.LETTERBOX,
        ResizeMode.CENTER_CROP,
        ResizeMode.LETTERBOX_REFLECT_EDGES,
    ):
        return False
    if not image_list:
        return False
    # `pre_process_network_input` already unbinds 4D inputs (batch tensors
    # and 4D numpy arrays) into a list of 3D items before calling us, so
    # the per-item check below only needs to handle 3D.
    for img in image_list:
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8 or img.ndim != 3:
                return False
            if img.shape[2] != 3 and not looks_like_chw(img.shape):
                return False
        elif isinstance(img, torch.Tensor):
            # Only uint8 3-channel images (HWC or CHW). Float tensors keep
            # the existing tensor branch (F.resize bilinear *without* PIL's
            # antialias — a caller-accepted divergence we don't silently
            # change).
            if img.dtype != torch.uint8 or img.ndim != 3:
                return False
            if img.shape[-1] != 3 and not looks_like_chw(img.shape):
                return False
        else:
            return False
    return True


def looks_like_chw(shape) -> bool:
    """Matches _tensor_to_hwc_uint8's CHW heuristic: first dim is 1/3/4 and
    last dim is not. Catches torchvision.io.read_image's CHW output."""
    return (
        len(shape) == 3
        and shape[0] in (1, 3, 4)
        and shape[-1] not in (1, 3, 4)
    )


def as_hwc_uint8_cuda(
    img: Union[np.ndarray, torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Return a contiguous (H, W, 3) uint8 CUDA tensor, copying if needed.
    Accepts HWC or CHW 3D inputs (CHW is torchvision.io.read_image's layout)."""
    if isinstance(img, torch.Tensor):
        if looks_like_chw(img.shape):
            img = img.permute(1, 2, 0)
        if img.device != device:
            img = img.to(device=device, non_blocking=True)
        return img.contiguous()
    if looks_like_chw(img.shape):
        img = np.transpose(img, (1, 2, 0))
    t = torch.from_numpy(np.ascontiguousarray(img))
    return t.to(device=device, non_blocking=True)


def triton_path_preprocess(
    image_list: List[Union[np.ndarray, torch.Tensor]],
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    target_size: ImageDimensions,
    target_device: torch.device,
    input_color_mode: Optional[ColorMode],
    pre_processing_overrides: Optional[PreProcessingOverrides],
) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
    """Per-image Triton launch + stack. Assumes `triton_path_eligible` passed."""
    means, stds = network_input.normalization
    means_t = (float(means[0]), float(means[1]), float(means[2]))
    stds_t = (float(stds[0]), float(stds[1]), float(stds[2]))
    th, tw = target_size.height, target_size.width

    # Match _pre_process_numpy's swap semantics: the PIL path does
    swap_rb = input_color_mode != network_input.color_mode

    # Resolve whether static_crop is active for this call. Computed once
    # because the config + override are call-level.
    static_crop_overridden = (
        pre_processing_overrides is not None
        and pre_processing_overrides.disable_static_crop is True
    )
    crop_cfg = image_pre_processing.static_crop
    crop_active = (
        crop_cfg is not None and crop_cfg.enabled and not static_crop_overridden
    )

    outs: List[torch.Tensor] = []
    metas: List[PreProcessingMetadata] = []
    for img in image_list:
        src_gpu = as_hwc_uint8_cuda(img, target_device)
        sh, sw = int(src_gpu.shape[0]), int(src_gpu.shape[1])

        if crop_active:
            # Matches apply_static_crop_to_numpy_image: percentage-based.
            x0 = int(crop_cfg.x_min / 100 * sw)
            y0 = int(crop_cfg.y_min / 100 * sh)
            x1 = int(crop_cfg.x_max / 100 * sw)
            y1 = int(crop_cfg.y_max / 100 * sh)
            crop_w = x1 - x0
            crop_h = y1 - y0
        else:
            x0 = y0 = 0
            crop_w, crop_h = sw, sh

        tables = get_resample_tables(target_device, crop_h, crop_w, th, tw)
        out = triton_preprocess_rfdetr_stretch(
            src=src_gpu,
            tables=tables,
            target_h=th,
            target_w=tw,
            means=means_t,
            stds=stds_t,
            swap_rb=swap_rb,
            crop_offset_y=y0,
            crop_offset_x=x0,
            crop_h=crop_h,
            crop_w=crop_w,
        )
        outs.append(out[0])  # drop leading batch dim to match stack below
        metas.append(
            PreProcessingMetadata(
                pad_left=0,
                pad_top=0,
                pad_right=0,
                pad_bottom=0,
                original_size=ImageDimensions(width=sw, height=sh),
                size_after_pre_processing=ImageDimensions(width=crop_w, height=crop_h),
                inference_size=ImageDimensions(width=tw, height=th),
                scale_width=tw / crop_w,
                scale_height=th / crop_h,
                static_crop_offset=StaticCropOffset(
                    offset_x=x0, offset_y=y0, crop_width=crop_w, crop_height=crop_h
                ),
            )
        )

    batch = torch.stack(outs).contiguous()
    return batch, metas


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
    return dims is not None and network_input.resize_mode != ResizeMode.STRETCH_TO


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
        network_input.scaling_factor is not None and network_input.scaling_factor != 255
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
