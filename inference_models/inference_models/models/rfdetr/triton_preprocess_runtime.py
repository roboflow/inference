"""Runtime glue for RF-DETR TensorRT Triton preprocessing.

This module owns the pieces that are specific to running the Triton
preprocessor inside the TensorRT RF-DETR model adapter: fast-path eligibility,
warning throttling, reusable CUDA buffers, and CUDA event handoff to the TRT
inference stream. The numerical resize/normalize kernels live in
``triton_preprocess.py``; this file only decides when and how to launch them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from inference_models.configuration import (
    INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED,
)
from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
    StaticCropOffset,
)
from inference_models.models.rfdetr.triton_jit_fallback import (
    is_triton_jit_failure,
    warn_triton_jit_fallback,
)

try:
    from inference_models.models.rfdetr.triton_preprocess import (
        TRITON_AVAILABLE as _TRITON_AVAILABLE,
        ResampleTables,
        build_resample_tables,
        resolve_two_pass_launch_config,
        triton_preprocess_rfdetr_stretch_two_pass_preallocated,
    )
except ImportError:  # pragma: no cover - import-time dependency guard
    _TRITON_AVAILABLE = False
    ResampleTables = None
    build_resample_tables = None
    resolve_two_pass_launch_config = None
    triton_preprocess_rfdetr_stretch_two_pass_preallocated = None


_FAST_PATH_ENABLED = INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED
_BUFFER_RING_SIZE = 3


@dataclass(frozen=True)
class FastPreprocessResult:
    """Result returned after the Triton preprocessing work is enqueued.

    Attributes:
        tensor: CUDA fp32 tensor with shape ``(1, 3, target_h, target_w)`` in
            network color order and normalized with the model's mean/std.
        metadata: Single-item preprocessing metadata list matching the reference
            RF-DETR TRT preprocessing contract.
        ready_event: Event recorded on the preprocessing stream after the HtoD
            copy and both Triton kernels. The TensorRT stream must wait on this
            event before consuming ``tensor``.
    """

    tensor: torch.Tensor
    metadata: List[PreProcessingMetadata]
    ready_event: torch.cuda.Event


class FastPreprocessState:
    """Reusable buffers for one source shape and one network input shape.

    The state is keyed by ``(src_h, src_w, target_h, target_w)`` because those
    dimensions define both the PIL fixed-point resample tables and every buffer
    size used by the Triton kernels.

    Attributes:
        pinned_hosts: Ring of pinned CPU HWC uint8 staging tensors. The incoming numpy
            image is copied here first so the following host-to-device copy can
            be submitted as ``non_blocking=True`` on the preprocessing stream.
        src_gpus: Ring of CUDA HWC uint8 tensors consumed by the horizontal Triton kernel.
        out_buffers: Ring of CUDA fp32 ``(1, 3, target_h, target_w)`` outputs.
            The returned tensor can still be owned by TensorRT or response
            finalization while Python prepares the next frame, so the ring avoids
            overwriting an output that downstream work may still read.
        tmp_buffers: Matching ring of CUDA uint8 ``(3, src_h, target_w)``
            horizontal-resize scratch buffers. They are paired with output
            buffers so each in-flight preprocessing submission has independent
            scratch storage until the vertical kernel finishes.
        tables: CUDA int32 PIL-compatible resample tables. ``xmin``/``wx`` are
            used by the horizontal pass and ``ymin``/``wy`` by the vertical pass.
        launch_config: Tuned block sizes for the two Triton kernels, resolved
            once from env vars when the shape-specific state is built.
    """

    __slots__ = (
        "src_h",
        "src_w",
        "target_h",
        "target_w",
        "pinned_hosts",
        "src_gpus",
        "out_buffers",
        "tmp_buffers",
        "out_buffer_index",
        "tables",
        "launch_config",
    )

    def __init__(
        self,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
        pinned_hosts: List[torch.Tensor],
        src_gpus: List[torch.Tensor],
        out_buffers: List[torch.Tensor],
        tmp_buffers: List[torch.Tensor],
        tables: ResampleTables,
        launch_config: Tuple[int, int, int, int],
    ) -> None:
        self.src_h = src_h
        self.src_w = src_w
        self.target_h = target_h
        self.target_w = target_w
        self.pinned_hosts = pinned_hosts
        self.src_gpus = src_gpus
        self.out_buffers = out_buffers
        self.tmp_buffers = tmp_buffers
        self.out_buffer_index = 0
        self.tables = tables
        self.launch_config = launch_config

    @classmethod
    def build(
        cls,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
        device: torch.device,
    ) -> "FastPreprocessState":
        """Allocate shape-specific buffers and build GPU resample tables."""
        pinned_hosts = [
            torch.empty((src_h, src_w, 3), dtype=torch.uint8, pin_memory=True)
            for _ in range(_BUFFER_RING_SIZE)
        ]
        src_gpus = [
            torch.empty((src_h, src_w, 3), dtype=torch.uint8, device=device)
            for _ in range(_BUFFER_RING_SIZE)
        ]
        out_buffers = [
            torch.empty((1, 3, target_h, target_w), dtype=torch.float32, device=device)
            for _ in range(_BUFFER_RING_SIZE)
        ]
        tmp_buffers = [
            torch.empty((3, src_h, target_w), dtype=torch.uint8, device=device)
            for _ in range(_BUFFER_RING_SIZE)
        ]
        tables = build_resample_tables(
            src_h=src_h,
            src_w=src_w,
            target_h=target_h,
            target_w=target_w,
            device=device,
        )
        return cls(
            src_h=src_h,
            src_w=src_w,
            target_h=target_h,
            target_w=target_w,
            pinned_hosts=pinned_hosts,
            src_gpus=src_gpus,
            out_buffers=out_buffers,
            tmp_buffers=tmp_buffers,
            tables=tables,
            launch_config=resolve_two_pass_launch_config(),
        )

    def is_stale(self, src_h: int, src_w: int, target_h: int, target_w: int) -> bool:
        """Return true when image/network dimensions no longer match state."""
        return (
            self.src_h != src_h
            or self.src_w != src_w
            or self.target_h != target_h
            or self.target_w != target_w
        )

    def next_buffers(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the next output/scratch pair from the ring."""
        idx = self.out_buffer_index
        pinned_host = self.pinned_hosts[idx]
        src_gpu = self.src_gpus[idx]
        out = self.out_buffers[idx]
        tmp = self.tmp_buffers[idx]
        self.out_buffer_index = (idx + 1) % len(self.out_buffers)
        return pinned_host, src_gpu, out, tmp


class FastPreprocessRuntime:
    """Eligibility, launch, and CUDA handoff manager for Triton preprocessing.

    ``RFDetrForInstanceSegmentationTRT`` owns one runtime instance for the life
    of the model adapter. The runtime keeps all mutable fast-path state here so
    the adapter only has to call ``try_preprocess(...)`` and, when a result is
    returned, make the TensorRT stream wait on ``result.ready_event``.

    Stream lifetime:
        The caller supplies the preprocessing CUDA stream for each launch. This
        runtime records the returned event on that stream after the host-to-
        device copy and both Triton kernels. The output tensor also records the
        stream so PyTorch's caching allocator cannot reuse its storage before
        the asynchronous preprocessing work has finished.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._state: Optional[FastPreprocessState] = None
        self._warned_reasons: set[str] = set()
        self._jit_disabled = False

    def try_preprocess(
        self,
        images,
        input_color_format: Optional[ColorFormat],
        image_size: Optional[Tuple[int, int]],
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
        stream: torch.cuda.Stream,
    ) -> Optional[FastPreprocessResult]:
        """Enqueue Triton preprocessing when the request is supported.

        Args:
            images: Single uint8 HWC numpy image, or a single-item list
                containing one. Batch sizes greater than one use the reference
                preprocessing path.
            input_color_format: Caller-provided image color order. ``None`` is
                treated as BGR, matching the model adapter default.
            image_size: Per-call size override. Overrides are rejected because
                the fast path is keyed to the model's configured network input.
            image_pre_processing: Model package preprocessing config used for
                fast-path eligibility checks.
            network_input: Model package network-input config. Its resize mode,
                color mode, normalization, and target size define the kernel
                contract.
            stream: CUDA stream where the HtoD copy and Triton kernels are
                enqueued. It must be a real stream when the request is eligible.

        Returns:
            ``FastPreprocessResult`` after the GPU work has been scheduled, or
            ``None`` when the opt-in flag is disabled or the request falls
            outside the conservative fast-path contract. In that case the caller
            should run the reference preprocessing path.
        """
        if not _FAST_PATH_ENABLED or self._jit_disabled:
            return None
        unsupported_reason = self._unsupported_reason(
            images=images,
            image_size=image_size,
            image_pre_processing=image_pre_processing,
            network_input=network_input,
        )
        if unsupported_reason is not None:
            self._warn_unsupported(unsupported_reason)
            return None

        candidate = images[0] if isinstance(images, list) else images
        caller_mode = (
            ColorMode(input_color_format)
            if input_color_format is not None
            else ColorMode.BGR
        )
        swap_rb = caller_mode != network_input.color_mode

        means, stds = network_input.normalization
        means_t = (float(means[0]), float(means[1]), float(means[2]))
        stds_t = (float(stds[0]), float(stds[1]), float(stds[2]))
        target_h = network_input.training_input_size.height
        target_w = network_input.training_input_size.width
        orig_h, orig_w = int(candidate.shape[0]), int(candidate.shape[1])

        state = self._state
        if state is None or state.is_stale(
            src_h=orig_h,
            src_w=orig_w,
            target_h=target_h,
            target_w=target_w,
        ):
            state = FastPreprocessState.build(
                src_h=orig_h,
                src_w=orig_w,
                target_h=target_h,
                target_w=target_w,
                device=self._device,
            )
            self._state = state

        pinned_host, src_gpu, out_buffer, tmp_buffer = state.next_buffers()
        preproc_ready_event = getattr(pinned_host, "_preproc_ready_event", None)
        if preproc_ready_event is not None:
            preproc_ready_event.synchronize()
        np.copyto(pinned_host.numpy(), candidate, casting="no")

        try:
            with torch.cuda.stream(stream):
                trt_consumed_event = getattr(out_buffer, "_trt_consumed_event", None)
                if trt_consumed_event is not None:
                    stream.wait_event(trt_consumed_event)
                src_gpu.copy_(pinned_host, non_blocking=True)
                triton_preprocess_rfdetr_stretch_two_pass_preallocated(
                    src=src_gpu,
                    out=out_buffer,
                    tmp=tmp_buffer,
                    tables=state.tables,
                    target_h=target_h,
                    target_w=target_w,
                    means=means_t,
                    stds=stds_t,
                    swap_rb=swap_rb,
                    launch_config=state.launch_config,
                )
                ready_event = torch.cuda.Event()
                ready_event.record(stream)
                out_buffer._trt_ready_event = ready_event  # type: ignore[attr-defined]
                pinned_host._preproc_ready_event = ready_event  # type: ignore[attr-defined]
                out_buffer.record_stream(stream)
        except Exception as exc:
            if not is_triton_jit_failure(exc):
                raise
            self._jit_disabled = True
            warn_triton_jit_fallback(
                path="preprocess",
                exc=exc,
                warned_reasons=self._warned_reasons,
            )
            return None

        metadata = [
            PreProcessingMetadata(
                pad_left=0,
                pad_top=0,
                pad_right=0,
                pad_bottom=0,
                original_size=ImageDimensions(width=orig_w, height=orig_h),
                size_after_pre_processing=ImageDimensions(
                    width=orig_w,
                    height=orig_h,
                ),
                inference_size=ImageDimensions(width=target_w, height=target_h),
                scale_width=target_w / orig_w,
                scale_height=target_h / orig_h,
                static_crop_offset=StaticCropOffset(
                    offset_x=0,
                    offset_y=0,
                    crop_width=orig_w,
                    crop_height=orig_h,
                ),
            )
        ]
        out_buffer._pre_processing_meta = metadata  # type: ignore[attr-defined]
        return FastPreprocessResult(
            tensor=out_buffer,
            metadata=metadata,
            ready_event=ready_event,
        )

    def _unsupported_reason(
        self,
        images,
        image_size: Optional[Tuple[int, int]],
        image_pre_processing: ImagePreProcessing,
        network_input: NetworkInputDefinition,
    ) -> Optional[str]:
        """Explain why the request must use the reference preprocessing path."""
        if not _TRITON_AVAILABLE:
            return "triton is not installed"
        if self._device.type != "cuda":
            return "CUDA device is required"
        if image_size is not None:
            return "custom image_size overrides are not supported"

        # Overrides can only disable configured transforms; they cannot enable
        # transforms. The fast path deliberately rejects model configs that ask
        # for transforms whose pixel semantics are not implemented in Triton.
        if (
            (
                image_pre_processing.static_crop is not None
                and image_pre_processing.static_crop.enabled
            )
            or (
                image_pre_processing.contrast is not None
                and image_pre_processing.contrast.enabled
            )
            or (
                image_pre_processing.grayscale is not None
                and image_pre_processing.grayscale.enabled
            )
        ):
            return "static crop, contrast, and grayscale preprocessing are unsupported"

        if network_input.dataset_version_resize_dimensions is not None:
            return "dataset-version resize is unsupported"
        if network_input.input_channels != 3:
            return "only 3-channel inputs are supported"
        if network_input.scaling_factor not in (None, 255):
            return "only scaling_factor None or 255 is supported"
        if network_input.normalization is None:
            return "normalization is required"
        if network_input.resize_mode not in (
            ResizeMode.STRETCH_TO,
            ResizeMode.LETTERBOX,
            ResizeMode.CENTER_CROP,
            ResizeMode.LETTERBOX_REFLECT_EDGES,
        ):
            return f"resize mode {network_input.resize_mode!r} is unsupported"

        if isinstance(images, list):
            if len(images) != 1:
                return "only batch size 1 is supported"
            candidate = images[0]
        else:
            candidate = images
        if not isinstance(candidate, np.ndarray):
            return "only numpy ndarray inputs are supported"
        if (
            candidate.dtype != np.uint8
            or candidate.ndim != 3
            or candidate.shape[2] != 3
        ):
            return "input must be uint8 HWC with 3 channels"
        return None

    def _warn_unsupported(self, reason: str) -> None:
        if reason in self._warned_reasons:
            return
        self._warned_reasons.add(reason)
        warnings.warn(
            f"RF-DETR Triton preprocess path is unsupported: {reason}",
            RuntimeWarning,
            stacklevel=4,
        )