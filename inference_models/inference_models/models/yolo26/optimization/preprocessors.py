"""Selectable YOLO26 depth-estimation preprocessing implementations."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

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
    StaticCropOffset,
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
    YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_PINNED_FUSED_CONVERT_V2,
    YOLO26_DEPTH_PREPROCESSOR_VPI_CUDA_LETTERBOX_FUSED_CONVERT_V3,
)
from inference_models.models.yolo26.triton_depth_preprocess import (
    ExactTritonImageTensorConverter,
)
from inference_models.models.yolo26.vpi_depth_preprocess import (
    VPICUDALetterboxResizer,
)

_BASE_DISPATCH_MAX_SOURCE_ELEMENTS = 640 * 480
_PINNED_STAGING_SLOT_COUNT = 2
_VPI_EXACT_SOURCE_SIZE = (2160, 3840)
_VPI_EXACT_TARGET_SIZE = (768, 768)
_VPI_EXACT_RESIZED_SIZE = (432, 768)
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


@dataclass
class _PinnedImageSlot:
    tensor: torch.Tensor
    array: np.ndarray
    reuse_event: torch.cuda.Event
    transfer_pending: bool = False


class _PinnedImageSlotPool:
    """Bounded pinned target-image storage with H2D reuse ordering."""

    def __init__(self, *, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._slots = queue.LifoQueue(maxsize=_PINNED_STAGING_SLOT_COUNT)
        for _ in range(_PINNED_STAGING_SLOT_COUNT):
            tensor = torch.empty(
                (height, width, 3),
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._slots.put(
                _PinnedImageSlot(
                    tensor=tensor,
                    array=tensor.numpy(),
                    reuse_event=torch.cuda.Event(),
                )
            )

    def acquire(self) -> _PinnedImageSlot:
        slot = self._slots.get()
        if slot.transfer_pending:
            with torch.cuda.nvtx.range(
                "yolo26-depth.preprocess.pinned-slot-reuse-wait"
            ):
                slot.reuse_event.synchronize()
            slot.transfer_pending = False

        return slot

    def release(self, slot: _PinnedImageSlot) -> None:
        self._slots.put(slot)


@dataclass(frozen=True)
class _PreparedLargeImageSource:
    image: np.ndarray
    original_size: ImageDimensions
    size_after_pre_processing: ImageDimensions
    target_size: ImageDimensions
    static_crop_offset: StaticCropOffset


@dataclass(frozen=True)
class _LetterboxGeometry:
    new_height: int
    new_width: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    scale: float


def _raise_incompatible_candidate(*reasons: str) -> None:
    raise ModelRuntimeError(
        message=(
            "Input is incompatible with the explicit YOLO26 preprocessor: "
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


def _prepare_large_numpy_source(
    *,
    image: np.ndarray,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    pre_processing_overrides: Optional[PreProcessingOverrides],
) -> _PreparedLargeImageSource:
    """Apply preserved CPU transforms before a selectable resize backend."""
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

    return _PreparedLargeImageSource(
        image=image,
        original_size=original_size,
        size_after_pre_processing=size_after_pre_processing,
        target_size=target_size,
        static_crop_offset=static_crop_offset,
    )


def _calculate_letterbox_geometry(
    *,
    source_size: ImageDimensions,
    target_size: ImageDimensions,
) -> _LetterboxGeometry:
    scale_width = target_size.width / source_size.width
    scale_height = target_size.height / source_size.height
    scale = min(scale_width, scale_height)
    new_width = int(source_size.width * scale)
    new_height = int(source_size.height * scale)
    pad_top = int((target_size.height - new_height) / 2)
    pad_left = int((target_size.width - new_width) / 2)

    return _LetterboxGeometry(
        new_height=new_height,
        new_width=new_width,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=target_size.height - pad_top - new_height,
        pad_right=target_size.width - pad_left - new_width,
        scale=scale,
    )


def _build_pre_processing_metadata(
    *,
    prepared: _PreparedLargeImageSource,
    pad_top: int,
    pad_left: int,
    pad_bottom: int,
    pad_right: int,
    scale_width: float,
    scale_height: float,
) -> PreProcessingMetadata:
    return PreProcessingMetadata(
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        original_size=prepared.original_size,
        size_after_pre_processing=prepared.size_after_pre_processing,
        inference_size=prepared.target_size,
        scale_width=scale_width,
        scale_height=scale_height,
        static_crop_offset=prepared.static_crop_offset,
    )


def _prepare_large_numpy_image(
    *,
    image: np.ndarray,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    pre_processing_overrides: Optional[PreProcessingOverrides],
    output_buffer: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, PreProcessingMetadata]:
    """Apply the preserved CPU image transforms and build identical metadata."""
    prepared = _prepare_large_numpy_source(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        input_color_mode=input_color_mode,
        pre_processing_overrides=pre_processing_overrides,
    )
    image = prepared.image
    target_size = prepared.target_size
    if output_buffer is not None and (
        output_buffer.dtype != np.uint8
        or output_buffer.shape != (target_size.height, target_size.width, 3)
        or not output_buffer.flags.c_contiguous
    ):
        _raise_incompatible_candidate(
            "pinned output must be contiguous uint8 HWC with target shape, received "
            f"dtype={output_buffer.dtype} shape={output_buffer.shape} "
            f"contiguous={output_buffer.flags.c_contiguous}"
        )
    if network_input.resize_mode is ResizeMode.STRETCH_TO:
        if output_buffer is None:
            prepared_image = cv2.resize(
                image,
                (target_size.width, target_size.height),
            )
        else:
            resized_image = cv2.resize(
                image,
                (target_size.width, target_size.height),
                dst=output_buffer,
            )
            if not np.shares_memory(resized_image, output_buffer):
                np.copyto(output_buffer, resized_image)
            prepared_image = output_buffer
        pad_left = pad_top = pad_right = pad_bottom = 0
        scale_width = target_size.width / prepared.size_after_pre_processing.width
        scale_height = target_size.height / prepared.size_after_pre_processing.height
    else:
        geometry = _calculate_letterbox_geometry(
            source_size=prepared.size_after_pre_processing,
            target_size=target_size,
        )
        new_width = geometry.new_width
        new_height = geometry.new_height
        pad_top = geometry.pad_top
        pad_left = geometry.pad_left
        pad_right = geometry.pad_right
        pad_bottom = geometry.pad_bottom
        padding_value = network_input.padding_value or 0
        if not 0 <= padding_value <= 255:
            _raise_incompatible_candidate(
                "letterbox padding value must fit uint8, received " f"{padding_value!r}"
            )
        if output_buffer is None:
            scaled_image = cv2.resize(image, (new_width, new_height))
            if pad_left == pad_top == pad_right == pad_bottom == 0:
                prepared_image = scaled_image
            else:
                prepared_image = np.full(
                    (target_size.height, target_size.width, 3),
                    padding_value,
                    dtype=np.uint8,
                )
                prepared_image[
                    pad_top : pad_top + new_height,
                    pad_left : pad_left + new_width,
                ] = scaled_image
        else:
            output_buffer.fill(padding_value)
            resized_region = output_buffer[
                pad_top : pad_top + new_height,
                pad_left : pad_left + new_width,
            ]
            if resized_region.flags.c_contiguous:
                scaled_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    dst=resized_region,
                )
                if not np.shares_memory(scaled_image, output_buffer):
                    np.copyto(resized_region, scaled_image)
            else:
                np.copyto(
                    resized_region,
                    cv2.resize(image, (new_width, new_height)),
                )
            prepared_image = output_buffer
        scale_width = geometry.scale
        scale_height = geometry.scale

    if not prepared_image.flags.c_contiguous:
        _raise_incompatible_candidate(
            "OpenCV resize output must be contiguous; implicit copies are disabled"
        )
    metadata = _build_pre_processing_metadata(
        prepared=prepared,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        scale_width=scale_width,
        scale_height=scale_height,
    )

    return prepared_image, metadata


def _prepare_exact_vpi_letterbox_request(
    *,
    image: np.ndarray,
    image_pre_processing: ImagePreProcessing,
    network_input: NetworkInputDefinition,
    input_color_mode: ColorMode,
    pre_processing_overrides: Optional[PreProcessingOverrides],
) -> Tuple[
    np.ndarray,
    _LetterboxGeometry,
    int,
    PreProcessingMetadata,
]:
    """Build the fixed 5x letterbox request proven exact on the target."""
    prepared = _prepare_large_numpy_source(
        image=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        input_color_mode=input_color_mode,
        pre_processing_overrides=pre_processing_overrides,
    )
    reasons = []
    if network_input.resize_mode is not ResizeMode.LETTERBOX:
        reasons.append(
            "resize mode must be letterbox, received "
            f"{network_input.resize_mode.value!r}"
        )
    source_size = (
        prepared.size_after_pre_processing.height,
        prepared.size_after_pre_processing.width,
    )
    if source_size != _VPI_EXACT_SOURCE_SIZE:
        reasons.append(
            "post-transform source size must be "
            f"{_VPI_EXACT_SOURCE_SIZE}, received {source_size}"
        )
    target_size = (prepared.target_size.height, prepared.target_size.width)
    if target_size != _VPI_EXACT_TARGET_SIZE:
        reasons.append(
            f"target size must be {_VPI_EXACT_TARGET_SIZE}, received {target_size}"
        )
    if not prepared.image.flags.c_contiguous:
        reasons.append(
            "post-transform image must be contiguous HWC; implicit copies are disabled"
        )
    if reasons:
        _raise_incompatible_candidate(*reasons)

    geometry = _calculate_letterbox_geometry(
        source_size=prepared.size_after_pre_processing,
        target_size=prepared.target_size,
    )
    resized_size = (geometry.new_height, geometry.new_width)
    if resized_size != _VPI_EXACT_RESIZED_SIZE:
        _raise_incompatible_candidate(
            "letterbox content size must be "
            f"{_VPI_EXACT_RESIZED_SIZE}, received {resized_size}"
        )
    padding_value = network_input.padding_value or 0
    if not 0 <= padding_value <= 255:
        _raise_incompatible_candidate(
            f"letterbox padding value must fit uint8, received {padding_value!r}"
        )
    metadata = _build_pre_processing_metadata(
        prepared=prepared,
        pad_top=geometry.pad_top,
        pad_left=geometry.pad_left,
        pad_bottom=geometry.pad_bottom,
        pad_right=geometry.pad_right,
        scale_width=geometry.scale,
        scale_height=geometry.scale,
    )

    return prepared.image, geometry, padding_value, metadata


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
        image = _extract_single_numpy_image(images)
        if _use_base_preprocess_path(image):
            with torch.cuda.nvtx.range(
                "yolo26-depth.preprocess[effective="
                f"{self.metadata.implementation_id},path=base]"
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
            "yolo26-depth.preprocess[effective="
            f"{self.metadata.implementation_id},path=fused-large]"
        ):
            output, metadata = self._prepare_and_convert(
                image=image,
                image_pre_processing=image_pre_processing,
                network_input=network_input,
                target_device=target_device,
                input_color_mode=input_color_mode,
                pre_processing_overrides=pre_processing_overrides,
                context=context,
            )

        return output, [metadata]

    def _prepare_and_convert(
        self,
        *,
        image: np.ndarray,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        target_device: torch.device,
        input_color_mode: ColorMode,
        pre_processing_overrides: Optional[PreProcessingOverrides],
        context: ExecutionContext,
    ) -> Tuple[torch.Tensor, PreProcessingMetadata]:
        del context
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

        return output, metadata


class TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor(
    TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor
):
    """Preserve OpenCV pixels while using bounded pinned asynchronous staging."""

    metadata = OptimizationMetadata(
        implementation_id=(
            YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_PINNED_FUSED_CONVERT_V2
        ),
        stage=OptimizationStage.PREPROCESS,
        version="2",
        target=DeviceCompatibility(
            device_kind="gpu",
            minimum_compute_capability=(7, 0),
        ),
        inputs=TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor.metadata.inputs,
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
                "ownership": "per-call CUDA staging and output tensors",
                "persistent_allocations": (
                    f"{_PINNED_STAGING_SLOT_COUNT} target-sized pinned uint8 HWC "
                    "host slots, allocated only after large-path dispatch"
                ),
                "aliasing": "none",
            }
        ),
        numerical_behavior=(
            "uses the preserved CPU OpenCV resize directly into bounded pinned "
            "target storage where layout permits; channel reversal, layout, and "
            "IEEE round-to-nearest float32 division remain unchanged; the base "
            "source shape uses the preserved path"
        ),
        stream_behavior=(
            "non-blocking H2D and the Triton conversion run on the caller stream; "
            "pinned host slots are not reused before their H2D completion event"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        super().__init__(device=device)
        self._pool_lock = threading.Lock()
        self._pinned_pool: Optional[_PinnedImageSlotPool] = None

    def _prepare_and_convert(
        self,
        *,
        image: np.ndarray,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        target_device: torch.device,
        input_color_mode: ColorMode,
        pre_processing_overrides: Optional[PreProcessingOverrides],
        context: ExecutionContext,
    ) -> Tuple[torch.Tensor, PreProcessingMetadata]:
        stream = context.current_stream
        if stream is None:
            _raise_incompatible_candidate(
                "pinned staging requires a caller preprocessing CUDA stream"
            )
        target_height = network_input.training_input_size.height
        target_width = network_input.training_input_size.width
        pool = self._get_pinned_pool(height=target_height, width=target_width)
        slot = pool.acquire()
        copy_enqueued = False
        try:
            with torch.cuda.nvtx.range("yolo26-depth.preprocess.cv2-resize"):
                _, metadata = _prepare_large_numpy_image(
                    image=image,
                    image_pre_processing=image_pre_processing,
                    network_input=network_input,
                    input_color_mode=input_color_mode,
                    pre_processing_overrides=pre_processing_overrides,
                    output_buffer=slot.array,
                )
            with torch.cuda.stream(stream):
                with torch.cuda.nvtx.range("yolo26-depth.preprocess.h2d-pinned"):
                    image_tensor = torch.empty(
                        slot.tensor.shape,
                        dtype=torch.uint8,
                        device=target_device,
                    )
                    image_tensor.copy_(slot.tensor, non_blocking=True)
                    copy_enqueued = True
                    slot.reuse_event.record(stream)
                    slot.transfer_pending = True
                with torch.cuda.nvtx.range("yolo26-depth.preprocess.fused-convert"):
                    output = self._converter.convert(
                        image=image_tensor,
                        reverse_channels=(input_color_mode != network_input.color_mode),
                        scaling_factor=float(network_input.scaling_factor),
                    )
        except Exception:
            if copy_enqueued and not slot.transfer_pending:
                stream.synchronize()
            raise
        finally:
            pool.release(slot)

        return output, metadata

    def _get_pinned_pool(self, *, height: int, width: int) -> _PinnedImageSlotPool:
        with self._pool_lock:
            if self._pinned_pool is None:
                self._pinned_pool = _PinnedImageSlotPool(
                    height=height,
                    width=width,
                )
            elif self._pinned_pool.height != height or self._pinned_pool.width != width:
                _raise_incompatible_candidate(
                    "pinned staging target shape is fixed per model instance; "
                    f"initialized {(self._pinned_pool.height, self._pinned_pool.width)} "
                    f"but received {(height, width)}"
                )

            return self._pinned_pool


class VPICUDALetterboxFusedConvertYOLO26DepthPreprocessor(
    TritonCV2ResizeFusedConvertYOLO26DepthPreprocessor
):
    """Use VPI for the exact fixed 5x letterbox resize on Jetson."""

    metadata = OptimizationMetadata(
        implementation_id=(
            YOLO26_DEPTH_PREPROCESSOR_VPI_CUDA_LETTERBOX_FUSED_CONVERT_V3
        ),
        stage=OptimizationStage.PREPROCESS,
        version="3",
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
                    "large_source_size": _VPI_EXACT_SOURCE_SIZE,
                    "target_size": _VPI_EXACT_TARGET_SIZE,
                    "resized_content_size": _VPI_EXACT_RESIZED_SIZE,
                    "resize_mode": ResizeMode.LETTERBOX.value,
                    "scaling_factor": 255,
                    "normalization": None,
                }
            ),
            dtypes=("uint8",),
            layouts=("contiguous HWC",),
        ),
        dependencies=("VPI", "torch", "triton"),
        fallback_id=YOLO26_DEPTH_PREPROCESSOR_BASE,
        changes_numerics=False,
        supports_concurrency=True,
        supports_cuda_graphs=False,
        output_contract=immutable_mapping(
            {
                "type": "torch.Tensor",
                "dtype": "float32",
                "shape": "1x3x768x768",
                "layout": "contiguous NCHW",
                "ownership": "per-call Torch output tensor",
                "persistent_allocations": (
                    "two 768x432 VPI BGR8 CUDA images, VPI streams, and CUDA "
                    "reuse events, allocated only after large-path dispatch"
                ),
                "interop": (
                    "VPI output is borrowed by Torch through the CUDA Array "
                    "Interface without a copy"
                ),
                "aliasing": "none at the returned TensorRT input",
            }
        ),
        numerical_behavior=(
            "the large path is restricted to the validated 3840x2160 to 768x432 "
            "5x letterbox resize, where VPI CUDA and OpenCV select the same source "
            "pixels; one Triton launch synthesizes padding, reverses channels, "
            "changes layout, and applies IEEE round-to-nearest float32 division "
            "by 255; the base source shape uses the preserved path; exact target "
            "snapshot validation is required"
        ),
        stream_behavior=(
            "VPI resize runs on a per-slot VPI stream and is host-synchronized "
            "before zero-copy Torch borrowing; Triton conversion runs on the "
            "caller preprocessing stream, and the VPI CUDA lock is retained until "
            "a recorded conversion-completion event signals"
        ),
    )

    def __init__(
        self,
        *,
        device: torch.device,
        vpi_module: Optional[Any] = None,
        converter: Optional[ExactTritonImageTensorConverter] = None,
    ) -> None:
        self._converter = (
            converter
            if converter is not None
            else ExactTritonImageTensorConverter(device=device)
        )
        self._resizer = VPICUDALetterboxResizer(
            device=device,
            vpi_module=vpi_module,
        )

    def _prepare_and_convert(
        self,
        *,
        image: np.ndarray,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        target_device: torch.device,
        input_color_mode: ColorMode,
        pre_processing_overrides: Optional[PreProcessingOverrides],
        context: ExecutionContext,
    ) -> Tuple[torch.Tensor, PreProcessingMetadata]:
        stream = context.current_stream
        if stream is None:
            _raise_incompatible_candidate(
                "VPI interoperability requires a caller preprocessing CUDA stream"
            )
        resized_source, geometry, padding_value, metadata = (
            _prepare_exact_vpi_letterbox_request(
                image=image,
                image_pre_processing=image_pre_processing,
                network_input=network_input,
                input_color_mode=input_color_mode,
                pre_processing_overrides=pre_processing_overrides,
            )
        )
        borrowed_image = self._resizer.resize(
            image=resized_source,
            output_height=geometry.new_height,
            output_width=geometry.new_width,
            target_device=target_device,
        )
        try:
            with torch.cuda.stream(stream):
                with torch.cuda.nvtx.range(
                    "yolo26-depth.preprocess.vpi-fused-letterbox-convert"
                ):
                    output = self._converter.convert_letterbox(
                        image=borrowed_image.tensor,
                        target_size=_VPI_EXACT_TARGET_SIZE,
                        padding=(
                            geometry.pad_top,
                            geometry.pad_left,
                            geometry.pad_bottom,
                            geometry.pad_right,
                        ),
                        padding_value=padding_value,
                        reverse_channels=(input_color_mode != network_input.color_mode),
                        scaling_factor=float(network_input.scaling_factor),
                    )
            borrowed_image.mark_consumed_and_release(stream=stream)
        except Exception:
            borrowed_image.abort_and_release(stream=stream)
            raise

        return output, metadata
