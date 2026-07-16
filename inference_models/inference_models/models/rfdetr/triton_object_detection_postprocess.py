"""Explicit fused CUDA postprocessing for RF-DETR object detection.

The implementation preserves RF-DETR's batched sigmoid and flat top-k
selection, then fuses class filtering/remapping, thresholding, stable
compaction, box conversion, metadata rescaling, and clipping into one Triton
kernel. A single device-to-host count handoff is required because the public
``Detections`` result contains variable-length tensors.

Explicit selection is strict: unsupported inputs raise an actionable error and
never silently fall back to the reference PyTorch implementation.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch

from inference_models import Detections
from inference_models.errors import ModelRuntimeError
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.rfdetr.class_remapping import ClassesReMapping
from inference_models.models.rfdetr.triton_jit_fallback import is_triton_jit_failure

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional GPU package
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _compact_transform_detections_kernel(
        scores_ptr,
        topk_indices_ptr,
        bboxes_ptr,
        metadata_ptr,
        class_mapping_ptr,
        thresholds_ptr,
        output_scores_ptr,
        output_classes_ptr,
        output_boxes_ptr,
        output_counts_ptr,
        num_queries,
        num_logits_classes,
        num_output_classes,
        threshold_scalar,
        MAPPING_SIZE: tl.constexpr,
        HAS_CLASS_MAPPING: tl.constexpr,
        HAS_CLASS_THRESHOLDS: tl.constexpr,
        BLOCK_QUERIES: tl.constexpr,
        METADATA_STRIDE: tl.constexpr,
    ):
        """Compact one batch row while preserving sorted top-k order."""
        batch_idx = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_QUERIES)
        in_bounds = offsets < num_queries
        input_offsets = batch_idx * num_queries + offsets

        scores = tl.load(scores_ptr + input_offsets, mask=in_bounds, other=0.0)
        flat_indices = tl.load(
            topk_indices_ptr + input_offsets, mask=in_bounds, other=0
        )
        raw_classes = flat_indices % num_logits_classes
        query_indices = flat_indices // num_logits_classes

        if HAS_CLASS_MAPPING:
            mapping_in_bounds = in_bounds & (raw_classes < MAPPING_SIZE)
            mapped_classes = tl.load(
                class_mapping_ptr + raw_classes,
                mask=mapping_in_bounds,
                other=-1,
            ).to(tl.int32)
            valid = mapping_in_bounds & (mapped_classes >= 0)
        else:
            mapped_classes = raw_classes.to(tl.int32)
            valid = in_bounds & (raw_classes < num_output_classes)

        if HAS_CLASS_THRESHOLDS:
            class_thresholds = tl.load(
                thresholds_ptr + mapped_classes,
                mask=valid,
                other=0.0,
            )
            valid = valid & (scores > class_thresholds)
        else:
            valid = valid & (scores > threshold_scalar)

        valid_int = valid.to(tl.int32)
        compact_offsets = tl.cumsum(valid_int, axis=0) - 1
        output_offsets = batch_idx * num_queries + compact_offsets

        tl.store(output_scores_ptr + output_offsets, scores, mask=valid)
        tl.store(
            output_classes_ptr + output_offsets,
            mapped_classes,
            mask=valid,
        )

        box_offsets = (batch_idx * num_queries + query_indices) * 4
        cx = tl.load(bboxes_ptr + box_offsets, mask=in_bounds, other=0.0)
        cy = tl.load(bboxes_ptr + box_offsets + 1, mask=in_bounds, other=0.0)
        width = tl.load(bboxes_ptr + box_offsets + 2, mask=in_bounds, other=0.0)
        height = tl.load(bboxes_ptr + box_offsets + 3, mask=in_bounds, other=0.0)

        metadata_base = batch_idx * METADATA_STRIDE
        denorm_width = tl.load(metadata_ptr + metadata_base)
        denorm_height = tl.load(metadata_ptr + metadata_base + 1)
        pad_left = tl.load(metadata_ptr + metadata_base + 2)
        pad_top = tl.load(metadata_ptr + metadata_base + 3)
        scale_width = tl.load(metadata_ptr + metadata_base + 4)
        scale_height = tl.load(metadata_ptr + metadata_base + 5)
        crop_offset_x = tl.load(metadata_ptr + metadata_base + 6)
        crop_offset_y = tl.load(metadata_ptr + metadata_base + 7)
        original_width = tl.load(metadata_ptr + metadata_base + 8)
        original_height = tl.load(metadata_ptr + metadata_base + 9)

        x1 = ((cx - 0.5 * width) * denorm_width - pad_left) / scale_width
        y1 = ((cy - 0.5 * height) * denorm_height - pad_top) / scale_height
        x2 = ((cx + 0.5 * width) * denorm_width - pad_left) / scale_width
        y2 = ((cy + 0.5 * height) * denorm_height - pad_top) / scale_height
        x1 = tl.maximum(0.0, tl.minimum(x1 + crop_offset_x, original_width))
        y1 = tl.maximum(0.0, tl.minimum(y1 + crop_offset_y, original_height))
        x2 = tl.maximum(0.0, tl.minimum(x2 + crop_offset_x, original_width))
        y2 = tl.maximum(0.0, tl.minimum(y2 + crop_offset_y, original_height))

        output_box_offsets = output_offsets * 4
        tl.store(output_boxes_ptr + output_box_offsets, x1, mask=valid)
        tl.store(output_boxes_ptr + output_box_offsets + 1, y1, mask=valid)
        tl.store(output_boxes_ptr + output_box_offsets + 2, x2, mask=valid)
        tl.store(output_boxes_ptr + output_box_offsets + 3, y2, mask=valid)

        count = tl.sum(valid_int, axis=0)
        tl.store(output_counts_ptr + batch_idx, count)


@dataclass(frozen=True)
class _PreparedInputs:
    metadata: torch.Tensor
    class_mapping: torch.Tensor
    thresholds: torch.Tensor
    threshold_scalar: float
    has_class_mapping: bool
    has_class_thresholds: bool


@dataclass(frozen=True)
class _CachedTensor:
    tensor: torch.Tensor
    ready_event: torch.cuda.Event


class FusedObjectDetectionPostprocessor:
    """Batched top-k plus one fused Triton object-detection postprocess."""

    _MAX_CACHE_ENTRIES = 16
    _METADATA_STRIDE = 10

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ModelRuntimeError(
                message="triton-fused-v1 requires a CUDA target device.",
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )
        self._device = device
        self._metadata_cache: OrderedDict[Tuple[Any, ...], _CachedTensor] = (
            OrderedDict()
        )
        self._threshold_cache: OrderedDict[Tuple[Any, ...], _CachedTensor] = (
            OrderedDict()
        )
        self._cache_lock = threading.Lock()

    def postprocess(
        self,
        bboxes: torch.Tensor,
        logits: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: Union[float, torch.Tensor],
        num_classes: int,
        classes_re_mapping: Optional[ClassesReMapping],
        stream: torch.cuda.Stream,
    ) -> List[Detections]:
        self._validate_inputs(
            bboxes=bboxes,
            logits=logits,
            pre_processing_meta=pre_processing_meta,
            threshold=threshold,
            num_classes=num_classes,
            classes_re_mapping=classes_re_mapping,
        )
        if not TRITON_AVAILABLE:
            raise ModelRuntimeError(
                message="triton-fused-v1 requires Triton, but Triton is not installed.",
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "runtime-environment/#missingdependencyerror"
                ),
            )

        batch_size, num_queries, num_logits_classes = logits.shape
        prepared = self._prepare_inputs(
            metadata=pre_processing_meta,
            threshold=threshold,
            dtype=logits.dtype,
            classes_re_mapping=classes_re_mapping,
            stream=stream,
        )

        with torch.cuda.stream(stream):
            sigmoid_logits = torch.sigmoid(logits)
            scores, topk_indices = torch.topk(
                sigmoid_logits.reshape(batch_size, -1),
                num_queries,
                dim=1,
                largest=True,
                sorted=True,
            )
            output_scores = torch.empty_like(scores)
            output_classes = torch.empty(
                (batch_size, num_queries),
                dtype=torch.int32,
                device=self._device,
            )
            output_boxes = torch.empty(
                (batch_size, num_queries, 4),
                dtype=bboxes.dtype,
                device=self._device,
            )
            output_counts = torch.empty(
                (batch_size,), dtype=torch.int32, device=self._device
            )

            try:
                _compact_transform_detections_kernel[(batch_size,)](
                    scores,
                    topk_indices,
                    bboxes,
                    prepared.metadata,
                    prepared.class_mapping,
                    prepared.thresholds,
                    output_scores,
                    output_classes,
                    output_boxes,
                    output_counts,
                    num_queries,
                    num_logits_classes,
                    num_classes,
                    prepared.threshold_scalar,
                    MAPPING_SIZE=int(prepared.class_mapping.numel()),
                    HAS_CLASS_MAPPING=prepared.has_class_mapping,
                    HAS_CLASS_THRESHOLDS=prepared.has_class_thresholds,
                    BLOCK_QUERIES=triton.next_power_of_2(num_queries),
                    METADATA_STRIDE=self._METADATA_STRIDE,
                    num_warps=8,
                )
            except Exception as error:
                if not is_triton_jit_failure(error):
                    raise
                raise ModelRuntimeError(
                    message=(
                        "triton-fused-v1 failed to compile or launch its "
                        f"postprocessing kernel: {type(error).__name__}: {error}"
                    ),
                    help_url=(
                        "https://inference-models.roboflow.com/errors/"
                        "models-runtime/#modelruntimeerror"
                    ),
                ) from error

            # Variable-length Python results require one bounded count handoff.
            # This synchronizes the fused kernel once for the whole batch.
            counts = output_counts.cpu().tolist()
            results = [
                Detections(
                    xyxy=output_boxes[index, :count].round().int(),
                    confidence=output_scores[index, :count],
                    class_id=output_classes[index, :count],
                )
                for index, count in enumerate(counts)
            ]
            for tensor in (output_scores, output_classes, output_boxes):
                tensor.record_stream(stream)
        return results

    def _validate_inputs(
        self,
        bboxes: torch.Tensor,
        logits: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        threshold: Union[float, torch.Tensor],
        num_classes: int,
        classes_re_mapping: Optional[ClassesReMapping],
    ) -> None:
        unsupported = []
        if not bboxes.is_cuda or not logits.is_cuda:
            unsupported.append("boxes and logits must be CUDA tensors")
        if bboxes.device != logits.device or bboxes.device != self._device:
            unsupported.append(
                "boxes, logits, and target must use the same CUDA device"
            )
        if bboxes.dtype != torch.float32 or logits.dtype != torch.float32:
            unsupported.append("boxes and logits must use float32")
        if not bboxes.is_contiguous() or not logits.is_contiguous():
            unsupported.append("boxes and logits must be contiguous")
        if bboxes.ndim != 3 or bboxes.shape[-1] != 4:
            unsupported.append(f"boxes shape must be BQ4, got {tuple(bboxes.shape)}")
        if logits.ndim != 3:
            unsupported.append(f"logits shape must be BQC, got {tuple(logits.shape)}")
        if bboxes.ndim == 3 and logits.ndim == 3:
            if bboxes.shape[:2] != logits.shape[:2]:
                unsupported.append("boxes and logits batch/query dimensions must match")
            if logits.shape[1] < 1 or logits.shape[1] > 1024:
                unsupported.append("query count must be between 1 and 1024")
            if logits.shape[2] < 1:
                unsupported.append("logits must contain at least one class")
            if len(pre_processing_meta) != logits.shape[0]:
                unsupported.append("metadata length must match batch size")
        if num_classes < 1:
            unsupported.append("num_classes must be positive")
        if isinstance(threshold, torch.Tensor):
            if threshold.ndim != 1 or threshold.numel() != num_classes:
                unsupported.append("per-class threshold must have num_classes values")
        if classes_re_mapping is not None:
            if not classes_re_mapping.class_mapping.is_cuda:
                unsupported.append("class mapping must be a CUDA tensor")
            if classes_re_mapping.class_mapping.device != self._device:
                unsupported.append("class mapping must use the target CUDA device")
        if unsupported:
            raise ModelRuntimeError(
                message=(
                    "triton-fused-v1 cannot preserve this postprocessing contract: "
                    + "; ".join(unsupported)
                    + ". Select 'base' for this configuration."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )

    def _prepare_inputs(
        self,
        metadata: List[PreProcessingMetadata],
        threshold: Union[float, torch.Tensor],
        dtype: torch.dtype,
        classes_re_mapping: Optional[ClassesReMapping],
        stream: torch.cuda.Stream,
    ) -> _PreparedInputs:
        metadata_values = tuple(
            value for element in metadata for value in _metadata_values(element)
        )
        metadata_key = (self._device.index, dtype, metadata_values)
        metadata_tensor = self._cached_device_tensor(
            cache=self._metadata_cache,
            key=metadata_key,
            values=metadata_values,
            dtype=dtype,
            shape=(len(metadata), self._METADATA_STRIDE),
            stream=stream,
        )

        if classes_re_mapping is None:
            class_mapping = metadata_tensor
        else:
            class_mapping = classes_re_mapping.class_mapping

        if isinstance(threshold, torch.Tensor):
            if threshold.is_cuda:
                thresholds = threshold.to(device=self._device, dtype=dtype)
                threshold.record_stream(stream)
                thresholds.record_stream(stream)
            else:
                threshold_values = tuple(float(value) for value in threshold.tolist())
                threshold_key = (self._device.index, dtype, threshold_values)
                thresholds = self._cached_device_tensor(
                    cache=self._threshold_cache,
                    key=threshold_key,
                    values=threshold_values,
                    dtype=dtype,
                    shape=(len(threshold_values),),
                    stream=stream,
                )
            threshold_scalar = 0.0
            has_class_thresholds = True
        else:
            thresholds = metadata_tensor
            threshold_scalar = float(threshold)
            has_class_thresholds = False

        return _PreparedInputs(
            metadata=metadata_tensor,
            class_mapping=class_mapping,
            thresholds=thresholds,
            threshold_scalar=threshold_scalar,
            has_class_mapping=classes_re_mapping is not None,
            has_class_thresholds=has_class_thresholds,
        )

    def _cached_device_tensor(
        self,
        cache: OrderedDict[Tuple[Any, ...], _CachedTensor],
        key: Tuple[Any, ...],
        values: Tuple[float, ...],
        dtype: torch.dtype,
        shape: Tuple[int, ...],
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        with self._cache_lock:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                with torch.cuda.stream(stream):
                    stream.wait_event(cached.ready_event)
                    cached.tensor.record_stream(stream)
                return cached.tensor
            with torch.cuda.stream(stream):
                tensor = torch.tensor(values, dtype=dtype, device=self._device).reshape(
                    shape
                )
                ready_event = torch.cuda.Event()
                ready_event.record(stream)
                tensor.record_stream(stream)
            cache[key] = _CachedTensor(tensor=tensor, ready_event=ready_event)
            while len(cache) > self._MAX_CACHE_ENTRIES:
                cache.popitem(last=False)
            return tensor


def _metadata_values(metadata: PreProcessingMetadata) -> Tuple[float, ...]:
    denorm_size = metadata.nonsquare_intermediate_size or metadata.inference_size
    return (
        float(denorm_size.width),
        float(denorm_size.height),
        float(metadata.pad_left),
        float(metadata.pad_top),
        float(metadata.scale_width),
        float(metadata.scale_height),
        float(metadata.static_crop_offset.offset_x),
        float(metadata.static_crop_offset.offset_y),
        float(metadata.original_size.width),
        float(metadata.original_size.height),
    )
