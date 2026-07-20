"""Explicit universal CUDA preprocessing for RF-DETR TensorRT adapters.

The uint8 path uses the PIL-exact RF-DETR Triton kernels. NumPy and CPU tensor
inputs are staged through a two-slot pinned HWC ring; CUDA tensors are
consumed directly. Floating-point tensors retain RF-DETR's tensor-input
semantics and use torchvision's antialiased CUDA resize.

This runtime remains strict when called directly. The RF-DETR implementation
selector checks its declared compatibility before execution and may choose the
base preprocessor for unsupported contracts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF

from inference_models import PreProcessingOverrides
from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
    StaticCropOffset,
)
from inference_models.models.optimization.contracts import CompatibilityResult
from inference_models.models.rfdetr.triton_jit_fallback import is_triton_jit_failure
from inference_models.models.rfdetr.triton_preprocess import (
    TRITON_AVAILABLE,
    ResampleTables,
    build_resample_tables,
    resolve_two_pass_launch_config,
    triton_preprocess_rfdetr_stretch_two_pass_preallocated,
)

ImageInput = Union[np.ndarray, torch.Tensor]
_STAGING_RING_SIZE = 2


@dataclass(frozen=True)
class UniversalFastPreprocessResult:
    tensor: torch.Tensor
    metadata: List[PreProcessingMetadata]
    ready_event: torch.cuda.Event
    input_kind: str


@dataclass(frozen=True)
class _CanonicalBatch:
    items: Tuple[ImageInput, ...]
    kind: Literal["uint8", "float"]
    height: int
    width: int


class _Uint8State:
    """Reusable double-buffered staging and scratch state for one shape pair."""

    __slots__ = (
        "src_h",
        "src_w",
        "target_h",
        "target_w",
        "pinned_host_ring",
        "src_gpu_ring",
        "slot_events",
        "tmp",
        "tables",
        "launch_config",
        "reuse_event",
    )

    def __init__(
        self,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
        pinned_host_ring: Tuple[torch.Tensor, ...],
        src_gpu_ring: Tuple[torch.Tensor, ...],
        tmp: torch.Tensor,
        tables: ResampleTables,
        launch_config: Tuple[int, int, int, int],
    ) -> None:
        self.src_h = src_h
        self.src_w = src_w
        self.target_h = target_h
        self.target_w = target_w
        self.pinned_host_ring = pinned_host_ring
        self.src_gpu_ring = src_gpu_ring
        self.slot_events: List[Optional[torch.cuda.Event]] = [
            None for _ in range(_STAGING_RING_SIZE)
        ]
        self.tmp = tmp
        self.tables = tables
        self.launch_config = launch_config
        self.reuse_event: Optional[torch.cuda.Event] = None

    @classmethod
    def build(
        cls,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
        device: torch.device,
    ) -> "_Uint8State":
        return cls(
            src_h=src_h,
            src_w=src_w,
            target_h=target_h,
            target_w=target_w,
            pinned_host_ring=tuple(
                torch.empty((src_h, src_w, 3), dtype=torch.uint8, pin_memory=True)
                for _ in range(_STAGING_RING_SIZE)
            ),
            src_gpu_ring=tuple(
                torch.empty((src_h, src_w, 3), dtype=torch.uint8, device=device)
                for _ in range(_STAGING_RING_SIZE)
            ),
            tmp=torch.empty((3, src_h, target_w), dtype=torch.uint8, device=device),
            tables=build_resample_tables(
                src_h=src_h,
                src_w=src_w,
                target_h=target_h,
                target_w=target_w,
                device=device,
            ),
            launch_config=resolve_two_pass_launch_config(),
        )

    def matches(self, src_h: int, src_w: int, target_h: int, target_w: int) -> bool:
        return (
            self.src_h == src_h
            and self.src_w == src_w
            and self.target_h == target_h
            and self.target_w == target_w
        )


class UniversalFastPreprocessRuntime:
    """Run one explicit CUDA preprocessing plan for supported RF-DETR inputs.

    The runtime serializes access to its reusable staging/scratch buffers.
    Output tensors are allocated per call from PyTorch's CUDA allocator, so
    concurrent callers never alias outputs. CUDA inputs are read on the
    supplied preprocessing stream and have their allocator lifetime recorded.
    """

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ModelRuntimeError(
                message="triton-universal-v1 requires a CUDA target device.",
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )
        self._device = device
        self._uint8_state: Optional[_Uint8State] = None
        self._state_lock = threading.Lock()

    def preprocess(
        self,
        images,
        input_color_format: Optional[ColorFormat],
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        pre_processing_overrides: Optional[PreProcessingOverrides],
        stream: torch.cuda.Stream,
    ) -> UniversalFastPreprocessResult:
        model_compatibility = self.check_model_compatibility(
            image_pre_processing=image_pre_processing,
            network_input=network_input,
        )
        self._raise_for_incompatibility(model_compatibility)
        request_compatibility = self.check_request_compatibility(
            images=images,
            pre_processing_overrides=pre_processing_overrides,
        )
        self._raise_for_incompatibility(request_compatibility)
        batch = _canonicalize_batch(images)
        caller_mode = (
            ColorMode(input_color_format)
            if input_color_format is not None
            else ColorMode.BGR
        )
        swap_rb = caller_mode != network_input.color_mode

        means, stds = network_input.normalization
        means_t = tuple(float(value) for value in means)
        stds_t = tuple(float(value) for value in stds)
        target_h = network_input.training_input_size.height
        target_w = network_input.training_input_size.width

        if batch.kind == "uint8":
            return self._preprocess_uint8(
                batch=batch,
                target_h=target_h,
                target_w=target_w,
                means=means_t,
                stds=stds_t,
                swap_rb=swap_rb,
                stream=stream,
            )
        return self._preprocess_float(
            batch=batch,
            target_h=target_h,
            target_w=target_w,
            means=means_t,
            stds=stds_t,
            swap_rb=swap_rb,
            stream=stream,
        )

    def _preprocess_uint8(
        self,
        batch: _CanonicalBatch,
        target_h: int,
        target_w: int,
        means: Tuple[float, float, float],
        stds: Tuple[float, float, float],
        swap_rb: bool,
        stream: torch.cuda.Stream,
    ) -> UniversalFastPreprocessResult:
        if not TRITON_AVAILABLE:
            raise ModelRuntimeError(
                message=(
                    "triton-universal-v1 requires the existing Triton runtime "
                    "dependency, but Triton is not installed."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "runtime-environment/#missingdependencyerror"
                ),
            )

        with self._state_lock:
            state = self._uint8_state
            if state is not None and state.reuse_event is not None:
                state.reuse_event.synchronize()
            if state is None or not state.matches(
                src_h=batch.height,
                src_w=batch.width,
                target_h=target_h,
                target_w=target_w,
            ):
                with torch.cuda.stream(stream):
                    state = _Uint8State.build(
                        src_h=batch.height,
                        src_w=batch.width,
                        target_h=target_h,
                        target_w=target_w,
                        device=self._device,
                    )
                self._uint8_state = state

            output = torch.empty(
                (len(batch.items), 3, target_h, target_w),
                dtype=torch.float32,
                device=self._device,
            )
            try:
                for index, item in enumerate(batch.items):
                    staging_slot = index % _STAGING_RING_SIZE
                    source = self._prepare_uint8_source(
                        item=item,
                        state=state,
                        stream=stream,
                        staging_slot=staging_slot,
                    )
                    with torch.cuda.stream(stream):
                        triton_preprocess_rfdetr_stretch_two_pass_preallocated(
                            src=source,
                            out=output[index : index + 1],
                            tmp=state.tmp,
                            tables=state.tables,
                            target_h=target_h,
                            target_w=target_w,
                            means=means,
                            stds=stds,
                            swap_rb=swap_rb,
                            launch_config=state.launch_config,
                        )
                        if not (isinstance(item, torch.Tensor) and item.is_cuda):
                            slot_event = torch.cuda.Event()
                            slot_event.record(stream)
                            state.slot_events[staging_slot] = slot_event
            except Exception as error:
                if not is_triton_jit_failure(error):
                    raise
                raise ModelRuntimeError(
                    message=(
                        "triton-universal-v1 failed to compile or launch its "
                        f"preprocessing kernel: {type(error).__name__}: {error}"
                    ),
                    help_url=(
                        "https://inference-models.roboflow.com/errors/"
                        "models-runtime/#modelruntimeerror"
                    ),
                ) from error

            with torch.cuda.stream(stream):
                ready_event = torch.cuda.Event()
                ready_event.record(stream)
                output.record_stream(stream)
            state.reuse_event = ready_event

        return UniversalFastPreprocessResult(
            tensor=output,
            metadata=_build_metadata_batch(
                batch_size=len(batch.items),
                source_h=batch.height,
                source_w=batch.width,
                target_h=target_h,
                target_w=target_w,
            ),
            ready_event=ready_event,
            input_kind="uint8-triton",
        )

    def _prepare_uint8_source(
        self,
        item: ImageInput,
        state: _Uint8State,
        stream: torch.cuda.Stream,
        staging_slot: int,
    ) -> torch.Tensor:
        if isinstance(item, torch.Tensor) and item.is_cuda:
            item.record_stream(stream)
            return item

        slot_event = state.slot_events[staging_slot]
        if slot_event is not None:
            slot_event.synchronize()
        pinned_host = state.pinned_host_ring[staging_slot]
        src_gpu = state.src_gpu_ring[staging_slot]

        if isinstance(item, np.ndarray):
            np.copyto(pinned_host.numpy(), item, casting="no")
        else:
            pinned_host.copy_(item)

        with torch.cuda.stream(stream):
            src_gpu.copy_(pinned_host, non_blocking=True)
        # The per-slot event is recorded after the corresponding Triton launch.
        # A two-slot ring lets the host fill the next frame while the GPU copies
        # and processes the current frame, without overwriting live storage.
        return src_gpu

    def _preprocess_float(
        self,
        batch: _CanonicalBatch,
        target_h: int,
        target_w: int,
        means: Tuple[float, float, float],
        stds: Tuple[float, float, float],
        swap_rb: bool,
        stream: torch.cuda.Stream,
    ) -> UniversalFastPreprocessResult:
        with torch.cuda.stream(stream):
            cuda_items = []
            for item in batch.items:
                assert isinstance(item, torch.Tensor)
                tensor = item.to(
                    device=self._device,
                    dtype=torch.float32,
                    non_blocking=item.device.type == "cpu" and item.is_pinned(),
                )
                if item.is_cuda:
                    item.record_stream(stream)
                cuda_items.append(tensor)
            tensor_batch = torch.stack(cuda_items, dim=0)
            if swap_rb:
                tensor_batch = tensor_batch[:, [2, 1, 0], :, :]
            resized = TF.resize(
                tensor_batch,
                (target_h, target_w),
                antialias=True,
            )
            output = TF.normalize(
                resized,
                mean=list(means),
                std=list(stds),
            ).contiguous()
            ready_event = torch.cuda.Event()
            ready_event.record(stream)
            output.record_stream(stream)

        return UniversalFastPreprocessResult(
            tensor=output,
            metadata=_build_metadata_batch(
                batch_size=len(batch.items),
                source_h=batch.height,
                source_w=batch.width,
                target_h=target_h,
                target_w=target_w,
            ),
            ready_event=ready_event,
            input_kind="float-torch-cuda",
        )

    @staticmethod
    def check_model_compatibility(
        *,
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
    ) -> CompatibilityResult:
        """Check static model configuration supported by the Triton runtime.

        Args:
            image_pre_processing: Model-package image transformations.
            network_input: Model-package network input definition.

        Returns:
            Compatibility result with every unsupported configuration element.
        """
        unsupported = []
        if network_input.resize_mode is not ResizeMode.STRETCH_TO:
            unsupported.append(f"resize_mode={network_input.resize_mode!r}")
        if network_input.dataset_version_resize_dimensions is not None:
            unsupported.append("dataset-version resize")
        if network_input.input_channels != 3:
            unsupported.append(f"input_channels={network_input.input_channels}")
        if network_input.scaling_factor not in (None, 255):
            unsupported.append(f"scaling_factor={network_input.scaling_factor}")
        if network_input.normalization is None:
            unsupported.append("missing normalization")
        elif any(len(values) != 3 for values in network_input.normalization):
            unsupported.append("normalization must contain three channels")
        if (
            image_pre_processing.static_crop is not None
            and image_pre_processing.static_crop.enabled
        ):
            unsupported.append("static crop")
        if (
            image_pre_processing.contrast is not None
            and image_pre_processing.contrast.enabled
        ):
            unsupported.append("contrast")
        if (
            image_pre_processing.grayscale is not None
            and image_pre_processing.grayscale.enabled
        ):
            unsupported.append("grayscale")
        if (
            image_pre_processing.auto_orient is not None
            and image_pre_processing.auto_orient.enabled
        ):
            unsupported.append("auto orient")
        if unsupported:
            result = CompatibilityResult.incompatible(*unsupported)
        else:
            result = CompatibilityResult.compatible()

        return result

    @staticmethod
    def check_request_compatibility(
        *,
        images,
        pre_processing_overrides: Optional[PreProcessingOverrides],
    ) -> CompatibilityResult:
        """Check request-specific constraints without performing GPU work.

        Args:
            images: Single image or batch supplied to preprocessing.
            pre_processing_overrides: Optional request transformation overrides.

        Returns:
            Compatibility result with every unsupported request characteristic.
        """
        unsupported = []
        if pre_processing_overrides is not None:
            unsupported.append("pre-processing overrides")

        raw_items = _raw_batch_items(images)
        if not raw_items:
            unsupported.append("empty image batch")
        kinds = []
        shapes = []
        for item in raw_items:
            item_contract = _inspect_item_contract(item)
            if isinstance(item_contract, str):
                unsupported.append(item_contract)
                continue
            kind, shape = item_contract
            kinds.append(kind)
            shapes.append(shape)
        if len(set(kinds)) > 1:
            unsupported.append("mixed uint8 and floating tensor semantics")
        if len(set(shapes)) > 1:
            unsupported.append(f"heterogeneous source dimensions: {shapes}")
        if "uint8" in kinds and not TRITON_AVAILABLE:
            unsupported.append("Triton is not installed for uint8 preprocessing")
        if unsupported:
            result = CompatibilityResult.incompatible(*unsupported)
        else:
            result = CompatibilityResult.compatible()

        return result

    @staticmethod
    def _raise_for_incompatibility(compatibility: CompatibilityResult) -> None:
        if compatibility.supported:
            return

        raise ModelRuntimeError(
            message=(
                "triton-universal-v1 cannot preserve this preprocessing contract: "
                f"{compatibility.reason}. Select 'base' for this configuration."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/models-runtime/"
                "#modelruntimeerror"
            ),
        )


def _canonicalize_batch(images) -> _CanonicalBatch:
    raw_items = _raw_batch_items(images)

    if not raw_items:
        raise ModelRuntimeError(
            message="triton-universal-v1 received an empty image batch.",
            help_url=(
                "https://inference-models.roboflow.com/errors/input-validation/"
                "#modelinputerror"
            ),
        )

    items = []
    kinds = []
    shapes = []
    for item in raw_items:
        if isinstance(item, np.ndarray):
            canonical, kind = _canonicalize_numpy(item)
        elif isinstance(item, torch.Tensor):
            canonical, kind = _canonicalize_tensor(item)
        else:
            raise ModelRuntimeError(
                message=(
                    "triton-universal-v1 accepts numpy.ndarray and torch.Tensor "
                    f"inputs, received {type(item).__name__}."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/input-validation/"
                    "#modelinputerror"
                ),
            )
        items.append(canonical)
        kinds.append(kind)
        if kind == "uint8":
            shapes.append((int(canonical.shape[0]), int(canonical.shape[1])))
        else:
            shapes.append((int(canonical.shape[1]), int(canonical.shape[2])))

    if len(set(kinds)) != 1:
        raise ModelRuntimeError(
            message=(
                "triton-universal-v1 requires a homogeneous batch; mixing uint8 "
                "image semantics with floating tensor semantics is unsupported."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/input-validation/"
                "#modelinputerror"
            ),
        )
    if len(set(shapes)) != 1:
        raise ModelRuntimeError(
            message=(
                "triton-universal-v1 currently requires equal source dimensions "
                f"within one batch; received {shapes}."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/input-validation/"
                "#modelinputerror"
            ),
        )

    height, width = shapes[0]
    if height < 1 or width < 1:
        raise ModelRuntimeError(
            message=(
                "triton-universal-v1 requires non-empty image dimensions; "
                f"received height={height}, width={width}."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/input-validation/"
                "#modelinputerror"
            ),
        )
    return _CanonicalBatch(
        items=tuple(items),
        kind=kinds[0],
        height=height,
        width=width,
    )


def _raw_batch_items(images) -> List[Any]:
    if isinstance(images, list):
        items = list(images)
    elif isinstance(images, torch.Tensor) and images.ndim == 4:
        items = list(images.unbind(0))
    elif isinstance(images, np.ndarray) and images.ndim == 4:
        items = list(images)
    else:
        items = [images]

    return items


def _inspect_item_contract(
    image,
) -> Union[Tuple[Literal["uint8", "float"], Tuple[int, int]], str]:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            return f"NumPy input must be HWC with three channels: {tuple(image.shape)}"
        return "uint8", (int(image.shape[0]), int(image.shape[1]))
    if not isinstance(image, torch.Tensor):
        return f"unsupported input type: {type(image).__name__}"
    if image.ndim != 3:
        return f"tensor input must be rank 3: {tuple(image.shape)}"

    is_chw = image.shape[0] == 3
    is_hwc = image.shape[-1] == 3
    if not is_chw and not is_hwc:
        return (
            f"tensor input must be CHW or HWC with three channels: {tuple(image.shape)}"
        )
    if not image.is_floating_point() and image.dtype != torch.uint8:
        return f"integer tensor input must use uint8: {image.dtype}"

    if is_chw:
        shape = (int(image.shape[1]), int(image.shape[2]))
    else:
        shape = (int(image.shape[0]), int(image.shape[1]))
    kind = "float" if image.is_floating_point() else "uint8"

    return kind, shape


def _canonicalize_numpy(image: np.ndarray) -> Tuple[np.ndarray, Literal["uint8"]]:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ModelRuntimeError(
            message=(
                "triton-universal-v1 requires NumPy images in HWC layout with "
                f"three channels; received shape={tuple(image.shape)}."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/input-validation/"
                "#modelinputerror"
            ),
        )
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image), "uint8"
    if np.issubdtype(image.dtype, np.floating):
        converted = (image * 255.0).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(converted), "uint8"
    return np.ascontiguousarray(image.astype(np.uint8)), "uint8"


def _canonicalize_tensor(
    image: torch.Tensor,
) -> Tuple[torch.Tensor, Literal["uint8", "float"]]:
    if image.ndim != 3:
        raise ModelRuntimeError(
            message=(
                "triton-universal-v1 requires each tensor image to be rank 3 "
                f"(CHW or HWC); received shape={tuple(image.shape)}."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/input-validation/"
                "#modelinputerror"
            ),
        )

    is_chw = image.shape[0] == 3
    is_hwc = image.shape[-1] == 3
    if not is_chw and not is_hwc:
        raise ModelRuntimeError(
            message=(
                "triton-universal-v1 requires three-channel CHW or HWC tensors; "
                f"received shape={tuple(image.shape)}."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/input-validation/"
                "#modelinputerror"
            ),
        )

    if image.is_floating_point():
        chw = image if is_chw else image.permute(2, 0, 1)
        return chw, "float"
    if image.dtype != torch.uint8:
        raise ModelRuntimeError(
            message=(
                "triton-universal-v1 accepts only uint8 integer tensors or "
                f"floating tensors; received dtype={image.dtype}."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/input-validation/"
                "#modelinputerror"
            ),
        )
    hwc = image.permute(1, 2, 0) if is_chw else image
    return hwc, "uint8"


def _build_metadata_batch(
    batch_size: int,
    source_h: int,
    source_w: int,
    target_h: int,
    target_w: int,
) -> List[PreProcessingMetadata]:
    original_size = ImageDimensions(width=source_w, height=source_h)
    target_size = ImageDimensions(width=target_w, height=target_h)
    static_crop_offset = StaticCropOffset(
        offset_x=0,
        offset_y=0,
        crop_width=source_w,
        crop_height=source_h,
    )
    return [
        PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=original_size,
            size_after_pre_processing=original_size,
            inference_size=target_size,
            scale_width=target_w / source_w,
            scale_height=target_h / source_h,
            static_crop_offset=static_crop_offset,
        )
        for _ in range(batch_size)
    ]
