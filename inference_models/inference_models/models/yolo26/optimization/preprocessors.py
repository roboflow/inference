"""Selectable YOLO26 depth-estimation preprocessing implementations."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from inference_models import ColorFormat
from inference_models.entities import ImageDimensions
from inference_models.errors import ModelRuntimeError
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
)
from inference_models.models.common.roboflow.pre_processing import (
    apply_pre_processing_to_numpy_image,
    pre_process_network_input,
)
from inference_models.models.optimization.contracts import (
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
    immutable_mapping,
    metadata_supports_context,
)
from inference_models.models.yolo26.optimization.ids import (
    YOLO26_DEPTH_PREPROCESSOR_BASE,
    YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_FUSED_CONVERT_V1,
)
from inference_models.models.yolo26.triton_depth_preprocess import (
    ExactTritonImageTensorConverter,
)

_BASE_DISPATCH_MAX_SOURCE_ELEMENTS = 640 * 480
_SUPPORTED_FUSED_RESIZE_MODES = {
    ResizeMode.STRETCH_TO,
    ResizeMode.LETTERBOX,
    ResizeMode.LETTERBOX_REFLECT_EDGES,
}
ImageInput = Union[
    torch.Tensor,
    List[torch.Tensor],
    np.ndarray,
    List[np.ndarray],
]


def _raise_incompatible_candidate(*reasons: str) -> None:
    raise ModelRuntimeError(
        message=(
            "Input is incompatible with the explicit YOLO26 Triton preprocessor: "
            + "; ".join(reasons)
        ),
        help_url=(
            "https://inference-models.roboflow.com/errors/"
            "models-runtime/#modelruntimeerror"
        ),
    )


def _extract_single_numpy_image(images: ImageInput) -> np.ndarray:
    if isinstance(images, np.ndarray):
        image = images
    elif (
        isinstance(images, list)
        and len(images) == 1
        and isinstance(images[0], np.ndarray)
    ):
        image = images[0]
    else:
        _raise_incompatible_candidate(
            "requires one numpy image (directly or in a one-element list)"
        )
    if image.ndim != 3 or image.shape[-1] != 3:
        _raise_incompatible_candidate(
            f"requires uint8 HWC input with three channels, received {image.shape}"
        )
    if image.dtype != np.uint8:
        _raise_incompatible_candidate(
            f"requires uint8 HWC input, received dtype={image.dtype}"
        )

    return image


def _use_base_preprocess_path(image: np.ndarray) -> bool:
    """Keep the lower-overhead preserved path for the frozen base shape."""
    return image.shape[0] * image.shape[1] <= _BASE_DISPATCH_MAX_SOURCE_ELEMENTS


def _prepare_large_numpy_image(
    *,
    image: np.ndarray,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    pre_processing_overrides: Optional[PreProcessingOverrides],
) -> Tuple[np.ndarray, PreProcessingMetadata]:
    """Apply the preserved CPU image transforms and build identical metadata."""
    reasons = []
    if network_input.input_channels != 3:
        reasons.append(
            f"network input must have three channels, received {network_input.input_channels}"
        )
    if network_input.resize_mode not in _SUPPORTED_FUSED_RESIZE_MODES:
        reasons.append(
            "resize mode must be stretch or letterbox, received "
            f"{network_input.resize_mode.value!r}"
        )
    if network_input.scaling_factor != 255:
        reasons.append(
            "network scaling factor must be exactly 255, received "
            f"{network_input.scaling_factor!r}"
        )
    if network_input.normalization is not None:
        reasons.append("network normalization must be disabled")
    if input_color_mode not in {ColorMode.BGR, ColorMode.RGB}:
        reasons.append(
            f"input color mode must be BGR or RGB, received {input_color_mode!r}"
        )
    if network_input.color_mode not in {ColorMode.BGR, ColorMode.RGB}:
        reasons.append(
            "network color mode must be BGR or RGB, received "
            f"{network_input.color_mode!r}"
        )
    if reasons:
        _raise_incompatible_candidate(*reasons)

    original_size = ImageDimensions(height=image.shape[0], width=image.shape[1])
    image, static_crop_offset = apply_pre_processing_to_numpy_image(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input_channels=network_input.input_channels,
        input_color_mode=input_color_mode,
        pre_processing_overrides=pre_processing_overrides,
    )
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        _raise_incompatible_candidate(
            "preserved CPU transforms must produce uint8 HWC data with three "
            f"channels, received dtype={image.dtype} shape={image.shape}"
        )

    size_after_pre_processing = ImageDimensions(
        height=image.shape[0],
        width=image.shape[1],
    )
    target_size = ImageDimensions(
        height=network_input.training_input_size.height,
        width=network_input.training_input_size.width,
    )
    if network_input.resize_mode is ResizeMode.STRETCH_TO:
        prepared_image = cv2.resize(
            image,
            (target_size.width, target_size.height),
        )
        pad_left = pad_top = pad_right = pad_bottom = 0
        scale_width = target_size.width / size_after_pre_processing.width
        scale_height = target_size.height / size_after_pre_processing.height
    else:
        scale_width = target_size.width / size_after_pre_processing.width
        scale_height = target_size.height / size_after_pre_processing.height
        scale = min(scale_width, scale_height)
        new_width = int(size_after_pre_processing.width * scale)
        new_height = int(size_after_pre_processing.height * scale)
        pad_top = int((target_size.height - new_height) / 2)
        pad_left = int((target_size.width - new_width) / 2)
        pad_right = target_size.width - pad_left - new_width
        pad_bottom = target_size.height - pad_top - new_height
        scaled_image = cv2.resize(image, (new_width, new_height))
        padding_value = network_input.padding_value or 0
        if pad_left == pad_top == pad_right == pad_bottom == 0:
            prepared_image = scaled_image
        else:
            if not 0 <= padding_value <= 255:
                _raise_incompatible_candidate(
                    "letterbox padding value must fit uint8, received "
                    f"{padding_value!r}"
                )
            prepared_image = np.full(
                (target_size.height, target_size.width, 3),
                padding_value,
                dtype=np.uint8,
            )
            prepared_image[
                pad_top : pad_top + new_height,
                pad_left : pad_left + new_width,
            ] = scaled_image
        scale_width = scale
        scale_height = scale

    if not prepared_image.flags.c_contiguous:
        _raise_incompatible_candidate(
            "OpenCV resize output must be contiguous; implicit copies are disabled"
        )
    metadata = PreProcessingMetadata(
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        inference_size=target_size,
        scale_width=scale_width,
        scale_height=scale_height,
        static_crop_offset=static_crop_offset,
    )

    return prepared_image, metadata


class BaseYOLO26DepthPreprocessor:
    """Preserve the shared Roboflow preprocessing implementation."""

    metadata = OptimizationMetadata(
        implementation_id=YOLO26_DEPTH_PREPROCESSOR_BASE,
        stage=OptimizationStage.PREPROCESS,
        version="1",
        target=DeviceCompatibility(device_kind="gpu"),
        inputs=InputCompatibility(
            scenarios=("*",),
            axis_constraints=immutable_mapping({"batch": ">=1"}),
            dtypes=("uint8", "floating point"),
            layouts=("HWC", "NHWC", "CHW", "NCHW"),
        ),
        dependencies=("torch", "torchvision"),
        fallback_id=YOLO26_DEPTH_PREPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "torch.Tensor",
                "dtype": "model preprocessing dependent",
                "layout": "contiguous NCHW",
                "ownership": "per-call tensor",
            }
        ),
        numerical_behavior="preserved shared Roboflow preprocessing path",
        stream_behavior="runs on the caller preprocessing stream",
    )

    def is_compatible(self, context: ExecutionContext) -> bool:
        return metadata_supports_context(self.metadata, context)

    def preprocess(
        self,
        *,
        images: ImageInput,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        target_device: torch.device,
        input_color_format: Optional[ColorFormat],
        pre_processing_overrides: Optional[PreProcessingOverrides],
        context: ExecutionContext,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        del context
        with torch.cuda.nvtx.range("yolo26-depth.preprocess[effective=base]"):
            result = pre_process_network_input(
                images=images,
                image_pre_processing=image_pre_processing,
                network_input=network_input,
                target_device=target_device,
                input_color_format=input_color_format,
                pre_processing_overrides=pre_processing_overrides,
            )

        return result


class TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor:
    """Keep exact OpenCV resize and fuse GPU channel/layout/scale conversion."""

    metadata = OptimizationMetadata(
        implementation_id=(
            YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_FUSED_CONVERT_V1
        ),
        stage=OptimizationStage.PREPROCESS,
        version="1",
        target=DeviceCompatibility(
            device_kind="gpu",
            minimum_compute_capability=(7, 0),
        ),
        inputs=InputCompatibility(
            scenarios=(
                "camera_640x480_batch_1_base",
                "camera_3840x2160_batch_1_high",
            ),
            axis_constraints=immutable_mapping(
                {
                    "batch": 1,
                    "channels": 3,
                    "base_dispatch_max_source_elements": (
                        _BASE_DISPATCH_MAX_SOURCE_ELEMENTS
                    ),
                    "fused_resize_modes": (
                        ResizeMode.STRETCH_TO.value,
                        ResizeMode.LETTERBOX.value,
                        ResizeMode.LETTERBOX_REFLECT_EDGES.value,
                    ),
                    "scaling_factor": 255,
                    "normalization": None,
                }
            ),
            dtypes=("uint8",),
            layouts=("HWC",),
        ),
        dependencies=("opencv-python", "torch", "triton"),
        fallback_id=YOLO26_DEPTH_PREPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "torch.Tensor",
                "dtype": "float32",
                "shape": "1x3xHxW",
                "layout": "contiguous NCHW",
                "ownership": "per-call source staging and output tensors",
                "per_call_allocations": (
                    "one CUDA uint8 HWC staging tensor and one float32 NCHW output"
                ),
                "aliasing": "none",
            }
        ),
        numerical_behavior=(
            "uses the preserved CPU OpenCV resize; the large path reverses channels, "
            "changes layout, and applies IEEE round-to-nearest float32 division by "
            "255 in one Triton launch; the base source shape uses the preserved path; "
            "exact target snapshot validation is required"
        ),
        stream_behavior=(
            "the H2D copy and Triton launch run on the active caller stream without "
            "a private stream or additional synchronization"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        self._converter = ExactTritonImageTensorConverter(device=device)

    def is_compatible(self, context: ExecutionContext) -> bool:
        return metadata_supports_context(self.metadata, context)

    def preprocess(
        self,
        *,
        images: ImageInput,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        target_device: torch.device,
        input_color_format: Optional[ColorFormat],
        pre_processing_overrides: Optional[PreProcessingOverrides],
        context: ExecutionContext,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        del context
        image = _extract_single_numpy_image(images)
        if _use_base_preprocess_path(image):
            with torch.cuda.nvtx.range(
                "yolo26-depth.preprocess["
                "effective=triton-cv2-resize-fused-convert-v1,path=base]"
            ):
                return pre_process_network_input(
                    images=images,
                    image_pre_processing=image_pre_processing,
                    network_input=network_input,
                    target_device=target_device,
                    input_color_format=input_color_format,
                    pre_processing_overrides=pre_processing_overrides,
                )

        input_color_mode = (
            ColorMode(input_color_format)
            if input_color_format is not None
            else ColorMode.BGR
        )
        with torch.cuda.nvtx.range(
            "yolo26-depth.preprocess["
            "effective=triton-cv2-resize-fused-convert-v1,path=fused-large]"
        ):
            with torch.cuda.nvtx.range("yolo26-depth.preprocess.cv2-resize"):
                prepared_image, metadata = _prepare_large_numpy_image(
                    image=image,
                    image_pre_processing=image_pre_processing,
                    network_input=network_input,
                    input_color_mode=input_color_mode,
                    pre_processing_overrides=pre_processing_overrides,
                )
            with torch.cuda.nvtx.range("yolo26-depth.preprocess.h2d"):
                image_tensor = torch.from_numpy(prepared_image).to(device=target_device)
            with torch.cuda.nvtx.range("yolo26-depth.preprocess.fused-convert"):
                output = self._converter.convert(
                    image=image_tensor,
                    reverse_channels=input_color_mode != network_input.color_mode,
                    scaling_factor=float(network_input.scaling_factor),
                )

        return output, [metadata]
