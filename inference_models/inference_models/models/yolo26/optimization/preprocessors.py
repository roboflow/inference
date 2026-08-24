"""Selectable YOLO26 depth-estimation preprocessing implementations."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

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
    YOLO26_DEPTH_PREPROCESSOR_OPENCV_FIXED_MAP_5X_PINNED_FUSED_CONVERT_V4,
    YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_FUSED_CONVERT_V1,
    YOLO26_DEPTH_PREPROCESSOR_TRITON_CV2_RESIZE_PINNED_FUSED_CONVERT_V2,
)
from inference_models.models.yolo26.triton_depth_preprocess import (
    ExactTritonImageTensorConverter,
)

_BASE_DISPATCH_MAX_SOURCE_ELEMENTS = 640 * 480
_PINNED_STAGING_SLOT_COUNT = 2
_FIXED_MAP_SOURCE_SIZE = (2160, 3840)
_FIXED_MAP_TARGET_SIZE = (768, 768)
_FIXED_MAP_RESIZED_SIZE = (432, 768)
_FIXED_MAP_SCALE = 5
_FIXED_MAP_OFFSET = 2
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


class _Exact5xFixedMapRemapper:
    """Apply the fixed 5x bilinear sampling geometry through OpenCV remap."""

    def __init__(self) -> None:
        self._map_lock = threading.Lock()
        self._coordinate_map: Optional[np.ndarray] = None

    def resize(self, source: np.ndarray, destination: np.ndarray) -> None:
        expected_source_shape = (*_FIXED_MAP_SOURCE_SIZE, 3)
        expected_destination_shape = (*_FIXED_MAP_RESIZED_SIZE, 3)
        reasons = []
        if source.dtype != np.uint8 or source.shape != expected_source_shape:
            reasons.append(
                "fixed-map source must be uint8 HWC with shape "
                f"{expected_source_shape}, received dtype={source.dtype} "
                f"shape={source.shape}"
            )
        if not source.flags.c_contiguous:
            reasons.append("fixed-map source must be contiguous HWC")
        if destination.dtype != np.uint8 or destination.shape != (
            expected_destination_shape
        ):
            reasons.append(
                "fixed-map destination must be uint8 HWC with shape "
                f"{expected_destination_shape}, received dtype={destination.dtype} "
                f"shape={destination.shape}"
            )
        if not destination.flags.c_contiguous:
            reasons.append("fixed-map destination must be contiguous HWC")
        if not destination.flags.writeable:
            reasons.append("fixed-map destination must be writeable")
        if np.shares_memory(source, destination):
            reasons.append("fixed-map source and destination must not alias")
        if reasons:
            _raise_incompatible_candidate(*reasons)

        resized = cv2.remap(
            source,
            self._get_coordinate_map(),
            None,
            interpolation=cv2.INTER_NEAREST,
            dst=destination,
        )
        if not np.shares_memory(resized, destination):
            _raise_incompatible_candidate(
                "OpenCV remap did not write into the provided pinned destination"
            )

    def _get_coordinate_map(self) -> np.ndarray:
        with self._map_lock:
            if self._coordinate_map is None:
                output_height, output_width = _FIXED_MAP_RESIZED_SIZE
                coordinate_map = np.empty(
                    (output_height, output_width, 2),
                    dtype=np.int16,
                )
                coordinate_map[..., 0] = (
                    np.arange(output_width, dtype=np.int16) * _FIXED_MAP_SCALE
                    + _FIXED_MAP_OFFSET
                )
                coordinate_map[..., 1] = (
                    np.arange(output_height, dtype=np.int16)[:, None] * _FIXED_MAP_SCALE
                    + _FIXED_MAP_OFFSET
                )
                coordinate_map.setflags(write=False)
                self._coordinate_map = coordinate_map

            return self._coordinate_map


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
    output_buffer: Optional[np.ndarray] = None,
    letterbox_resize_function: Optional[
        Callable[[np.ndarray, np.ndarray], None]
    ] = None,
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
    if (
        letterbox_resize_function is not None
        and network_input.resize_mode is not ResizeMode.LETTERBOX
    ):
        _raise_incompatible_candidate(
            "fixed-map resize requires the standard letterbox mode"
        )

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
        padding_value = network_input.padding_value or 0
        if not 0 <= padding_value <= 255:
            _raise_incompatible_candidate(
                "letterbox padding value must fit uint8, received " f"{padding_value!r}"
            )
        if output_buffer is None:
            if letterbox_resize_function is not None:
                _raise_incompatible_candidate(
                    "fixed-map resize requires a pinned output buffer"
                )
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
            if letterbox_resize_function is not None:
                letterbox_resize_function(image, resized_region)
            elif resized_region.flags.c_contiguous:
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

    _RESIZE_RANGE = "yolo26-depth.preprocess.cv2-resize"

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
            with torch.cuda.nvtx.range(self._RESIZE_RANGE):
                _, metadata = self._prepare_pinned_image(
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

    def _prepare_pinned_image(
        self,
        *,
        image: np.ndarray,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        input_color_mode: ColorMode,
        pre_processing_overrides: Optional[PreProcessingOverrides],
        output_buffer: np.ndarray,
    ) -> Tuple[np.ndarray, PreProcessingMetadata]:
        return _prepare_large_numpy_image(
            image=image,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            input_color_mode=input_color_mode,
            pre_processing_overrides=pre_processing_overrides,
            output_buffer=output_buffer,
        )

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


class OpenCVFixedMap5xPinnedFusedConvertYOLO26DepthPreprocessor(
    TritonCV2ResizePinnedFusedConvertYOLO26DepthPreprocessor
):
    """Replace the exact fixed 5x OpenCV resize with a cached sampling map."""

    _RESIZE_RANGE = "yolo26-depth.preprocess.fixed-map-remap"

    metadata = OptimizationMetadata(
        implementation_id=(
            YOLO26_DEPTH_PREPROCESSOR_OPENCV_FIXED_MAP_5X_PINNED_FUSED_CONVERT_V4
        ),
        stage=OptimizationStage.PREPROCESS,
        version="4",
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
                    "large_source_size": _FIXED_MAP_SOURCE_SIZE,
                    "target_size": _FIXED_MAP_TARGET_SIZE,
                    "resized_content_size": _FIXED_MAP_RESIZED_SIZE,
                    "resize_mode": ResizeMode.LETTERBOX.value,
                    "scaling_factor": 255,
                    "normalization": None,
                }
            ),
            dtypes=("uint8",),
            layouts=("contiguous HWC",),
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
                "shape": "1x3x768x768",
                "layout": "contiguous NCHW",
                "ownership": "per-call CUDA staging and output tensors",
                "persistent_allocations": (
                    f"{_PINNED_STAGING_SLOT_COUNT} target-sized pinned uint8 HWC "
                    "host slots and one immutable 432x768 CV_16SC2 coordinate map, "
                    "allocated only after large-path dispatch"
                ),
                "aliasing": "none",
            }
        ),
        numerical_behavior=(
            "the guarded 3840x2160 to 768x432 5x letterbox reduction maps each "
            "OpenCV bilinear sample center to source coordinate (5*x+2, 5*y+2); "
            "a cached nearest-neighbor remap therefore preserves exact uint8 "
            "pixels; channel reversal, layout, and IEEE round-to-nearest float32 "
            "division remain unchanged; the base source shape uses the preserved path"
        ),
        stream_behavior=(
            "the CPU remap writes directly into a bounded pinned slot; non-blocking "
            "H2D and Triton conversion run on the caller stream; slots are not "
            "reused before their H2D completion event"
        ),
    )

    def __init__(self, *, device: torch.device) -> None:
        super().__init__(device=device)
        self._fixed_map_remapper = _Exact5xFixedMapRemapper()

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
        expected_source_shape = (*_FIXED_MAP_SOURCE_SIZE, 3)
        target_size = (
            network_input.training_input_size.height,
            network_input.training_input_size.width,
        )
        reasons = []
        if (
            image.dtype != np.uint8
            or image.shape != expected_source_shape
            or not image.flags.c_contiguous
        ):
            reasons.append(
                "fixed-map input must be contiguous uint8 HWC with shape "
                f"{expected_source_shape}, received dtype={image.dtype} "
                f"shape={image.shape} contiguous={image.flags.c_contiguous}"
            )
        if target_size != _FIXED_MAP_TARGET_SIZE:
            reasons.append(
                "fixed-map target size must be "
                f"{_FIXED_MAP_TARGET_SIZE}, received {target_size}"
            )
        if network_input.resize_mode is not ResizeMode.LETTERBOX:
            reasons.append(
                "fixed-map resize mode must be "
                f"{ResizeMode.LETTERBOX.value!r}, received "
                f"{network_input.resize_mode.value!r}"
            )
        if reasons:
            _raise_incompatible_candidate(*reasons)

        return super()._prepare_and_convert(
            image=image,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            target_device=target_device,
            input_color_mode=input_color_mode,
            pre_processing_overrides=pre_processing_overrides,
            context=context,
        )

    def _prepare_pinned_image(
        self,
        *,
        image: np.ndarray,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        input_color_mode: ColorMode,
        pre_processing_overrides: Optional[PreProcessingOverrides],
        output_buffer: np.ndarray,
    ) -> Tuple[np.ndarray, PreProcessingMetadata]:
        return _prepare_large_numpy_image(
            image=image,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
            input_color_mode=input_color_mode,
            pre_processing_overrides=pre_processing_overrides,
            output_buffer=output_buffer,
            letterbox_resize_function=self._fixed_map_remapper.resize,
        )
