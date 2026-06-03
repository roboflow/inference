"""Sparse Triton RF-DETR instance-segmentation post-processing.

The normal PyTorch path upsamples every selected mask to image resolution and
then immediately converts that dense boolean tensor to COCO RLE. For 1080p
frames that dense intermediate is the expensive part. This module keeps the
same RF-DETR selection semantics, but asks Triton to interpolate only the
active mask region and emit sparse RLE run records directly.

The CUDA side writes two buffers:

* ``metadata``: one fixed-width row per output detection candidate containing
  active flag, mapped class id, score, clipped xyxy box, source query id, sort
  key, and debug ROI bounds.
* ``records``: a flat list of ``(rank, run_start, run_end)`` triples in COCO's
  column-major order. ``records[0, 0]`` is the run count and ``records[0, 1]``
  is an overflow / retry flag.

CPU code then performs only the small ordered assembly step: copy metadata and
run records back, sort detections by score, convert each candidate's runs into
compressed COCO RLE counts, and wrap them in ``InstanceDetections``.
"""

import warnings
from collections import OrderedDict
from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.rfdetr.class_remapping import ClassesReMapping

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on optional GPU package
    triton = None
    tl = None


_HEADER_SIZE = 16
# One Triton program scans this many output rows per column tile. Keeping this
# bounded avoids materializing a full HxW mask while still amortizing per-tile
# interpolation setup.
_BLOCK_ROI_H = 512
# RLE flat positions are stored as int32 and converted exactly through fp32
# metadata fields. Keep H*W below the fp32 exact-integer range.
_MAX_EXACT_FLAT_INDEX = 1 << 24
# The sparse RLE kernel processes columns in bands. The common case has small
# active mask bounds, but the while loop below can advance through wider ROIs.
_SPARSE_MAX_ROI_WIDTH = 512
_SPARSE_BLOCK_COLS = 8
_SPARSE_MAX_TOTAL_RUNS = 8192
_SPARSE_MAX_CLASSES_PER_QUERY = 4
_SPARSE_TOPK_MAX_TOTAL_RUNS = _SPARSE_MAX_TOTAL_RUNS * _SPARSE_MAX_CLASSES_PER_QUERY
_MAX_INTERPOLATION_WEIGHT_CACHE_ENTRIES = 16
_INTERPOLATION_WEIGHT_CACHE = OrderedDict()
_INTERPOLATION_WEIGHT_CACHE_LOCK = Lock()


def _get_interpolation_weights(
    src_size: int,
    output_size: int,
    device: torch.device,
    axis: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sparse two-tap bilinear interpolation tables for one axis.

    The Triton RLE kernel needs to reproduce ``torchvision.functional.resize``
    with bilinear antialiasing, but it cannot call PyTorch's resize from inside
    a kernel. We build the interpolation matrix once by resizing an identity
    basis, keep only the non-zero source index and weight pairs for each output
    coordinate, and cache those small tables per device/shape/axis.
    """
    device_key = _interpolation_cache_key(src_size, output_size, device, axis)
    with _INTERPOLATION_WEIGHT_CACHE_LOCK:
        cached = _INTERPOLATION_WEIGHT_CACHE.get(device_key)
        if cached is not None:
            _INTERPOLATION_WEIGHT_CACHE.move_to_end(device_key)
            return cached

    # Resize an identity basis so PyTorch gives us exactly the interpolation
    # coefficients used by the reference path. This keeps the Triton path tied
    # to PyTorch semantics instead of maintaining a second resize formula.
    if axis == "height":
        basis = torch.eye(src_size, device=device).reshape(src_size, 1, src_size, 1)
        weights = F.interpolate(
            basis,
            size=(output_size, 1),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[:, 0, :, 0].T.contiguous()
    else:
        basis = torch.eye(src_size, device=device).reshape(src_size, 1, 1, src_size)
        weights = F.interpolate(
            basis,
            size=(1, output_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[:, 0, 0, :].T.contiguous()

    nonzero = weights != 0
    if int(nonzero.sum(dim=1).max().item()) > 2:
        raise ValueError("Expected antialiased bilinear interpolation to use 2 taps")
    indices = torch.zeros((output_size, 2), dtype=torch.int32, device=device)
    values = torch.zeros((output_size, 2), dtype=torch.float32, device=device)
    for output_index in range(output_size):
        source_indices = nonzero[output_index].nonzero(as_tuple=True)[0]
        indices[output_index, : source_indices.numel()] = source_indices.to(
            dtype=torch.int32
        )
        values[output_index, : source_indices.numel()] = weights[
            output_index, source_indices
        ]

    cached_value = (indices, values)
    with _INTERPOLATION_WEIGHT_CACHE_LOCK:
        cached = _INTERPOLATION_WEIGHT_CACHE.get(device_key)
        if cached is not None:
            _INTERPOLATION_WEIGHT_CACHE.move_to_end(device_key)
            return cached
        _INTERPOLATION_WEIGHT_CACHE[device_key] = cached_value
        while (
            len(_INTERPOLATION_WEIGHT_CACHE) > _MAX_INTERPOLATION_WEIGHT_CACHE_ENTRIES
        ):
            _INTERPOLATION_WEIGHT_CACHE.popitem(last=False)
    return cached_value


def _interpolation_cache_key(
    src_size: int,
    output_size: int,
    device: torch.device,
    axis: str,
) -> Tuple[str, int, int, int, int, str]:
    """Build the LRU key for interpolation tables.

    CUDA tensors may have ``device.index is None`` when they refer to the
    current device, so the key also includes ``torch.cuda.current_device()`` to
    avoid reusing tables across devices in multi-GPU processes.
    """
    return (
        device.type,
        -1 if device.index is None else device.index,
        src_size,
        output_size,
        torch.cuda.current_device() if device.type == "cuda" else -1,
        axis,
    )


def post_process_single_instance_segmentation_result_to_rle_masks_triton(
    image_bboxes: torch.Tensor,
    image_scores: torch.Tensor,
    image_masks: torch.Tensor,
    image_meta: PreProcessingMetadata,
    threshold: Union[float, torch.Tensor],
    classes_re_mapping: Optional[ClassesReMapping],
) -> Optional[InstanceDetections]:
    """Run the sparse Triton RF-DETR RLE postprocess path for one image.

    Returns an ``InstanceDetections`` object when the input shape and metadata
    are supported. Returns ``None`` when the caller should use the reference
    PyTorch/RLE implementation instead.

    The fast path first emits one candidate per query. If any query has more
    than one class above threshold, the first pass asks for a retry and the
    second pass emits up to ``_SPARSE_MAX_CLASSES_PER_QUERY`` query-class
    candidates per query.
    """
    unsupported_reason = _unsupported_triton_postprocess_reason(
        image_bboxes=image_bboxes,
        image_scores=image_scores,
        image_masks=image_masks,
        image_meta=image_meta,
        threshold=threshold,
        classes_re_mapping=classes_re_mapping,
    )
    if unsupported_reason is not None:
        warnings.warn(
            f"RF-DETR Triton postprocess path is unsupported: {unsupported_reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    image_scores = image_scores.contiguous()
    image_bboxes = image_bboxes.contiguous()
    image_masks = image_masks.contiguous()
    class_mapping = classes_re_mapping.class_mapping.contiguous()
    num_queries, num_classes = image_scores.shape
    mask_height, mask_width = image_masks.shape[-2:]
    output_height = image_meta.original_size.height
    output_width = image_meta.original_size.width
    confidence_threshold = float(threshold)

    # Precompute resize tables outside the hot kernel. The tables are tiny
    # compared with the full-resolution masks and can be reused across frames.
    y_idx, y_weight = _get_interpolation_weights(
        src_size=mask_height,
        output_size=output_height,
        device=image_masks.device,
        axis="height",
    )
    x_idx, x_weight = _get_interpolation_weights(
        src_size=mask_width,
        output_size=output_width,
        device=image_masks.device,
        axis="width",
    )

    # First pass: keep the common case small by selecting only the best class
    # for each query and emitting sparse RLE runs for those query masks.
    metadata = torch.empty(
        (num_queries, _HEADER_SIZE),
        dtype=torch.float32,
        device=image_scores.device,
    )
    records = torch.empty(
        (_SPARSE_MAX_TOTAL_RUNS + 1, 3),
        dtype=torch.int32,
        device=image_scores.device,
    )
    _select_best_query_metadata_kernel[(num_queries,)](
        image_scores,
        image_bboxes,
        class_mapping,
        metadata,
        records,
        confidence_threshold,
        num_queries,
        num_classes,
        class_mapping.shape[0],
        output_height,
        output_width,
        BLOCK_CLASSES=triton.next_power_of_2(num_classes),
        METADATA_STRIDE=_HEADER_SIZE,
        FLAG_MULTICLASS=True,
    )
    _sparse_atomic_rle_from_metadata_kernel[
        (num_queries, triton.cdiv(_SPARSE_MAX_ROI_WIDTH, _SPARSE_BLOCK_COLS))
    ](
        image_masks,
        y_idx,
        y_weight,
        x_idx,
        x_weight,
        metadata,
        records,
        num_queries,
        mask_height,
        mask_width,
        output_height,
        output_width,
        image_masks.stride(0),
        image_masks.stride(1),
        image_masks.stride(2),
        BLOCK_MASK=triton.next_power_of_2(mask_height * mask_width),
        BLOCK_OUT_H=triton.next_power_of_2(output_height),
        BLOCK_OUT_W=triton.next_power_of_2(output_width),
        BLOCK_ROI_H=_BLOCK_ROI_H,
        MAX_ROI_WIDTH=_SPARSE_MAX_ROI_WIDTH,
        MAX_TOTAL_RUNS=_SPARSE_MAX_TOTAL_RUNS,
        METADATA_STRIDE=_HEADER_SIZE,
        BLOCK_COLS=_SPARSE_BLOCK_COLS,
    )

    metadata_host = metadata.cpu().numpy()
    result = _instance_detections_from_sparse_records(
        metadata_host=metadata_host,
        records=records,
        max_total_runs=_SPARSE_MAX_TOTAL_RUNS,
        height=output_height,
        width=output_width,
    )
    if result is not None:
        return result
    if not _should_retry_sparse_topk_metadata(
        metadata_host=metadata_host,
        records=records,
        max_total_runs=_SPARSE_MAX_TOTAL_RUNS,
    ):
        return None

    # Retry only when the first pass detected multiple passing classes for a
    # query. This preserves RF-DETR's flat top-k query-class semantics without
    # paying the expanded metadata/RLE cost on the usual one-class-per-query
    # path.
    topk_metadata_rows = num_queries * _SPARSE_MAX_CLASSES_PER_QUERY
    metadata = torch.empty(
        (topk_metadata_rows, _HEADER_SIZE),
        dtype=torch.float32,
        device=image_scores.device,
    )
    records = torch.empty(
        (_SPARSE_TOPK_MAX_TOTAL_RUNS + 1, 3),
        dtype=torch.int32,
        device=image_scores.device,
    )
    _select_topk_query_class_metadata_kernel[(num_queries,)](
        image_scores,
        image_bboxes,
        class_mapping,
        metadata,
        records,
        confidence_threshold,
        num_queries,
        num_classes,
        class_mapping.shape[0],
        output_height,
        output_width,
        BLOCK_CLASSES=triton.next_power_of_2(num_classes),
        METADATA_STRIDE=_HEADER_SIZE,
        MAX_CLASSES_PER_QUERY=_SPARSE_MAX_CLASSES_PER_QUERY,
        FLAG_OVERFLOW_CLASSES=True,
    )
    _sparse_atomic_rle_from_metadata_kernel[
        (
            topk_metadata_rows,
            triton.cdiv(_SPARSE_MAX_ROI_WIDTH, _SPARSE_BLOCK_COLS),
        )
    ](
        image_masks,
        y_idx,
        y_weight,
        x_idx,
        x_weight,
        metadata,
        records,
        topk_metadata_rows,
        mask_height,
        mask_width,
        output_height,
        output_width,
        image_masks.stride(0),
        image_masks.stride(1),
        image_masks.stride(2),
        BLOCK_MASK=triton.next_power_of_2(mask_height * mask_width),
        BLOCK_OUT_H=triton.next_power_of_2(output_height),
        BLOCK_OUT_W=triton.next_power_of_2(output_width),
        BLOCK_ROI_H=_BLOCK_ROI_H,
        MAX_ROI_WIDTH=_SPARSE_MAX_ROI_WIDTH,
        MAX_TOTAL_RUNS=_SPARSE_TOPK_MAX_TOTAL_RUNS,
        METADATA_STRIDE=_HEADER_SIZE,
        BLOCK_COLS=_SPARSE_BLOCK_COLS,
    )
    metadata_host = metadata.cpu().numpy()
    return _instance_detections_from_sparse_records(
        metadata_host=metadata_host,
        records=records,
        max_total_runs=_SPARSE_TOPK_MAX_TOTAL_RUNS,
        height=output_height,
        width=output_width,
        max_detections=num_queries,
    )


def _instance_detections_from_sparse_records(
    metadata_host: np.ndarray,
    records: torch.Tensor,
    max_total_runs: int,
    height: int,
    width: int,
    max_detections: Optional[int] = None,
) -> Optional[InstanceDetections]:
    """Convert sparse device records into ``InstanceDetections``.

    ``metadata_host`` is already on CPU because it is small and needed to decide
    ordering and retry/fallback. ``records`` may still live on CUDA; this helper
    copies it only after the metadata indicates at least one active candidate.

    ``None`` means the sparse device result is incomplete or overflowed and the
    caller should retry or fall back to the reference implementation.
    """
    active_ranks = np.flatnonzero(metadata_host[:, 0] > 0.5)
    if active_ranks.size == 0:
        return InstanceDetections(
            xyxy=torch.empty((0, 4), dtype=torch.int32),
            confidence=torch.empty((0,), dtype=torch.float32),
            class_id=torch.empty((0,), dtype=torch.int32),
            mask=InstancesRLEMasks.from_coco_rle_masks(
                image_size=(height, width),
                masks=[],
            ),
        )
    if np.any(metadata_host[active_ranks, 8] > 0.5):
        return None
    records_host = records.cpu().numpy()
    total_runs = int(records_host[0, 0])
    if int(records_host[0, 1]) != 0 or total_runs < 0 or total_runs > max_total_runs:
        return None

    # Match RF-DETR's descending score order. ``metadata[:, 10]`` is the flat
    # query-class index and gives a deterministic secondary order for equal
    # scores without touching the mask records.
    order = np.lexsort(
        (
            -metadata_host[active_ranks, 10],
            -metadata_host[active_ranks, 2],
        )
    )
    active_ranks = active_ranks[order]
    if max_detections is not None:
        active_ranks = active_ranks[:max_detections]
    records_host = records_host[1 : total_runs + 1] if total_runs else None
    boxes = torch.from_numpy(metadata_host[active_ranks, 3:7].copy()).round().int()
    confidence = torch.from_numpy(metadata_host[active_ranks, 2].copy())
    class_id = torch.from_numpy(metadata_host[active_ranks, 1].copy()).int()

    rle_masks = []
    for rank in active_ranks.tolist():
        if records_host is None:
            rank_records = np.empty((0, 3), dtype=np.int32)
        else:
            rank_records = records_host[records_host[:, 0] == rank]
        if rank_records.size:
            starts_array = rank_records[:, 1].astype(np.int64, copy=False)
            ends_array = rank_records[:, 2].astype(np.int64, copy=False)
            # Atomic writes from different column tiles are not globally
            # ordered, so sort runs before converting them into COCO counts.
            order = np.argsort(starts_array, kind="stable")
            starts_array = starts_array[order]
            ends_array = ends_array[order]
        else:
            starts_array = np.empty((0,), dtype=np.int64)
            ends_array = np.empty((0,), dtype=np.int64)
        counts = _counts_from_runs(
            starts=starts_array,
            ends=ends_array,
            height=height,
            width=width,
        )
        rle_masks.append(_rle_from_counts(counts=counts, height=height, width=width))

    instances_masks = InstancesRLEMasks.from_coco_rle_masks(
        image_size=(height, width),
        masks=rle_masks,
    )
    return InstanceDetections(
        xyxy=boxes,
        confidence=confidence,
        class_id=class_id,
        mask=instances_masks,
    )


def _should_retry_sparse_topk_metadata(
    metadata_host: np.ndarray,
    records: torch.Tensor,
    max_total_runs: int,
) -> bool:
    """Return whether first-pass sparse metadata needs query-class expansion."""
    active_ranks = np.flatnonzero(metadata_host[:, 0] > 0.5)
    if active_ranks.size == 0 or np.any(metadata_host[active_ranks, 8] > 0.5):
        return False
    records_host = records.cpu().numpy()
    total_runs = int(records_host[0, 0])
    if int(records_host[0, 1]) == 0 or total_runs < 0 or total_runs > max_total_runs:
        return False
    return True


def _supports_triton_postprocess_path(
    image_bboxes: torch.Tensor,
    image_scores: torch.Tensor,
    image_masks: torch.Tensor,
    image_meta: PreProcessingMetadata,
    threshold: Union[float, torch.Tensor],
    classes_re_mapping: Optional[ClassesReMapping],
) -> bool:
    """Return ``True`` when the sparse Triton path can represent this input."""
    return (
        _unsupported_triton_postprocess_reason(
            image_bboxes=image_bboxes,
            image_scores=image_scores,
            image_masks=image_masks,
            image_meta=image_meta,
            threshold=threshold,
            classes_re_mapping=classes_re_mapping,
        )
        is None
    )


def _unsupported_triton_postprocess_reason(
    image_bboxes: torch.Tensor,
    image_scores: torch.Tensor,
    image_masks: torch.Tensor,
    image_meta: PreProcessingMetadata,
    threshold: Union[float, torch.Tensor],
    classes_re_mapping: Optional[ClassesReMapping],
) -> Optional[str]:
    """Explain why the Triton path should not run, or ``None`` when supported."""
    if triton is None:
        return "triton_unavailable"
    if classes_re_mapping is None:
        return "class_remapping_required"
    if isinstance(threshold, torch.Tensor):
        return "tensor_threshold_unsupported"
    if image_scores.ndim != 2 or image_bboxes.ndim != 2 or image_masks.ndim != 3:
        return "invalid_tensor_rank"
    num_queries, num_classes = image_scores.shape
    if image_bboxes.shape != (num_queries, 4) or image_masks.shape[0] != num_queries:
        return "shape_mismatch"
    if classes_re_mapping.class_mapping.shape[0] < num_classes:
        return "class_mapping_too_small"
    mask_height, mask_width = image_masks.shape[-2:]
    output_height = image_meta.original_size.height
    output_width = image_meta.original_size.width
    if (
        num_queries * num_classes > 16384
        or mask_height * mask_width > 8192
        or output_height <= 0
        or output_width <= 0
        or output_height > 4096
        or output_width > 4096
        or output_height * output_width >= _MAX_EXACT_FLAT_INDEX
    ):
        return "input_size_exceeds_triton_limits"
    if (
        image_meta.pad_left != 0
        or image_meta.pad_top != 0
        or image_meta.pad_right != 0
        or image_meta.pad_bottom != 0
    ):
        return "padding_unsupported"
    if (
        image_meta.static_crop_offset.offset_x != 0
        or image_meta.static_crop_offset.offset_y != 0
    ):
        return "static_crop_unsupported"
    if (
        image_meta.size_after_pre_processing.height != output_height
        or image_meta.size_after_pre_processing.width != output_width
    ):
        return "resize_metadata_unsupported"
    if image_scores.device.type != "cuda":
        return "cuda_device_required"
    if (
        image_bboxes.device != image_scores.device
        or image_masks.device != image_scores.device
        or classes_re_mapping.class_mapping.device != image_scores.device
    ):
        return "device_mismatch"
    return None


def _counts_from_runs(
    starts: np.ndarray,
    ends: np.ndarray,
    height: int,
    width: int,
) -> List[int]:
    """Build uncompressed COCO RLE counts from sorted column-major runs."""
    total = height * width
    lengths = ends - starts
    valid = lengths > 0
    if starts.size and not np.all(valid):
        starts = starts[valid]
        ends = ends[valid]
        lengths = lengths[valid]

    if starts.size:
        run_count = starts.size
        # COCO counts alternate background gaps and foreground lengths. Starts
        # are absolute flat positions; subtract the prior end in-place to get
        # each background gap.
        gaps = starts.astype(np.int64, copy=True)
        gaps[1:] -= ends[:-1]
        tail = total - int(ends[-1])
        if tail > 0:
            counts = np.empty(run_count * 2 + 1, dtype=np.int64)
            counts[-1] = tail
        else:
            counts = np.empty(run_count * 2, dtype=np.int64)
        counts[: run_count * 2 : 2] = gaps
        counts[1 : run_count * 2 : 2] = lengths
        counts = counts.tolist()
    else:
        counts = [total]
    return counts


def _rle_from_counts(counts: List[int], height: int, width: int) -> dict:
    """Compress uncompressed COCO RLE counts with pycocotools."""
    return mask_utils.frPyObjects(
        {"counts": counts, "size": [height, width]}, height, width
    )


if triton is not None:

    @triton.jit
    def _select_best_query_metadata_kernel(
        scores,
        bboxes,
        class_mapping,
        metadata,
        records,
        threshold: tl.constexpr,
        num_queries: tl.constexpr,
        num_classes: tl.constexpr,
        class_mapping_size: tl.constexpr,
        output_height: tl.constexpr,
        output_width: tl.constexpr,
        BLOCK_CLASSES: tl.constexpr,
        METADATA_STRIDE: tl.constexpr,
        FLAG_MULTICLASS: tl.constexpr,
    ):
        """Select one query-level detection row from RF-DETR scores.

        Launch grid:
            ``(num_queries,)``. Program id 0 is also responsible for clearing
            the two-word ``records`` header before the RLE kernel runs.

        Args:
            scores: CUDA tensor with shape ``[num_queries, num_classes]`` and
                dtype float32. Values are sigmoid class probabilities.
            bboxes: CUDA tensor with shape ``[num_queries, 4]`` and dtype
                float32. Boxes are normalized ``cx, cy, width, height`` values.
            class_mapping: CUDA int tensor with at least ``num_classes`` values.
                ``class_mapping[class_id]`` is the public class id; negative
                entries mark model classes that should be ignored.
            metadata: CUDA float32 tensor with shape
                ``[num_queries, METADATA_STRIDE]``. The kernel writes columns
                ``0`` active flag, ``1`` mapped class id, ``2`` score, ``3:7``
                clipped xyxy pixel box, ``8`` unsupported flag, ``9`` source
                query id, ``10`` flat query-class sort key, and ``11:15`` zeroed
                ROI/debug fields later filled by the RLE kernel.
            records: CUDA int32 tensor with shape ``[MAX_TOTAL_RUNS + 1, 3]``.
                Only ``records[0, 0]`` and ``records[0, 1]`` are touched here:
                run count and retry/overflow flag.
            threshold: Confidence threshold applied after the best valid mapped
                class is selected.
            num_queries: Number of RF-DETR object queries, matching
                ``scores.shape[0]`` and ``bboxes.shape[0]``.
            num_classes: Number of model class columns in ``scores``.
            class_mapping_size: Number of entries available in
                ``class_mapping``.
            output_height: Original image height used to convert normalized box
                coordinates to pixel coordinates.
            output_width: Original image width used to convert normalized box
                coordinates to pixel coordinates.
            BLOCK_CLASSES: Power-of-two tile width covering ``num_classes``.
            METADATA_STRIDE: Number of float32 fields per metadata row.
            FLAG_MULTICLASS: When true, writes ``records[0, 1] = 1`` if more
                than one mapped class for this query exceeds ``threshold`` so
                the caller can rerun the top-k query-class path.
        """
        rank = tl.program_id(0)
        meta_base = rank * METADATA_STRIDE
        if rank == 0:
            tl.store(records + 0, 0)
            tl.store(records + 1, 0)

        class_offsets = tl.arange(0, BLOCK_CLASSES)
        class_active = class_offsets < num_classes
        mapped_classes = tl.load(
            class_mapping + class_offsets,
            mask=class_active & (class_offsets < class_mapping_size),
            other=-1,
        ).to(tl.int32)
        class_scores = tl.load(
            scores + rank * num_classes + class_offsets,
            mask=class_active,
            other=-1.0,
        )
        valid_classes = class_active & (mapped_classes >= 0)
        passing_classes = valid_classes & (class_scores > threshold)
        passing_class_count = tl.sum(tl.where(passing_classes, 1, 0), axis=0)
        if FLAG_MULTICLASS and passing_class_count > 1:
            tl.store(records + 1, 1)
        # Select over valid mapped classes, not just passing classes. The
        # threshold is applied after selection so inactive metadata rows still
        # carry a stable class/score shape.
        selected_score = tl.max(tl.where(valid_classes, class_scores, -1.0), axis=0)
        selected_class = tl.max(
            tl.where(
                valid_classes & (class_scores == selected_score),
                class_offsets,
                -1,
            ),
            axis=0,
        ).to(tl.int32)
        mapped_class = tl.load(
            class_mapping + selected_class,
            mask=(selected_class >= 0) & (selected_class < class_mapping_size),
            other=-1,
        ).to(tl.int32)
        is_valid_detection = (mapped_class >= 0) & (selected_score > threshold)
        query_index = rank
        selected_index = rank * num_classes + selected_class

        # Metadata is float32 because Python copies it back as one compact
        # array; integer-like fields stay below the exact fp32 integer range.
        tl.store(
            metadata + meta_base + 0,
            tl.where(is_valid_detection, 1.0, 0.0),
        )
        tl.store(metadata + meta_base + 1, mapped_class.to(tl.float32))
        tl.store(
            metadata + meta_base + 2,
            tl.where(is_valid_detection, selected_score, 0.0),
        )
        tl.store(metadata + meta_base + 7, 0.0)
        tl.store(metadata + meta_base + 8, 0.0)
        tl.store(metadata + meta_base + 9, query_index.to(tl.float32))
        tl.store(metadata + meta_base + 10, selected_index.to(tl.float32))
        tl.store(metadata + meta_base + 11, 0.0)
        tl.store(metadata + meta_base + 12, 0.0)
        tl.store(metadata + meta_base + 13, 0.0)
        tl.store(metadata + meta_base + 14, 0.0)
        tl.store(metadata + meta_base + 15, 0.0)

        bbox_base = query_index * 4
        cx = tl.load(bboxes + bbox_base, mask=is_valid_detection, other=0.0)
        cy = tl.load(bboxes + bbox_base + 1, mask=is_valid_detection, other=0.0)
        width = tl.load(bboxes + bbox_base + 2, mask=is_valid_detection, other=0.0)
        height = tl.load(
            bboxes + bbox_base + 3,
            mask=is_valid_detection,
            other=0.0,
        )
        x1 = tl.maximum(
            0.0,
            tl.minimum((cx - 0.5 * width) * output_width, output_width),
        )
        y1 = tl.maximum(
            0.0,
            tl.minimum((cy - 0.5 * height) * output_height, output_height),
        )
        x2 = tl.maximum(
            0.0,
            tl.minimum((cx + 0.5 * width) * output_width, output_width),
        )
        y2 = tl.maximum(
            0.0,
            tl.minimum((cy + 0.5 * height) * output_height, output_height),
        )
        tl.store(metadata + meta_base + 3, x1)
        tl.store(metadata + meta_base + 4, y1)
        tl.store(metadata + meta_base + 5, x2)
        tl.store(metadata + meta_base + 6, y2)

    @triton.jit
    def _select_topk_query_class_metadata_kernel(
        scores,
        bboxes,
        class_mapping,
        metadata,
        records,
        threshold: tl.constexpr,
        num_queries: tl.constexpr,
        num_classes: tl.constexpr,
        class_mapping_size: tl.constexpr,
        output_height: tl.constexpr,
        output_width: tl.constexpr,
        BLOCK_CLASSES: tl.constexpr,
        METADATA_STRIDE: tl.constexpr,
        MAX_CLASSES_PER_QUERY: tl.constexpr,
        FLAG_OVERFLOW_CLASSES: tl.constexpr,
    ):
        """Emit top passing query-class metadata rows for one RF-DETR query.

        Launch grid:
            ``(num_queries,)``. Each program scans all class scores for one
            query and writes up to ``MAX_CLASSES_PER_QUERY`` rows. The current
            implementation uses a static loop of four iterations, so
            ``MAX_CLASSES_PER_QUERY`` is expected to be ``4``.

        Args:
            scores: CUDA float32 tensor with shape
                ``[num_queries, num_classes]`` containing sigmoid class scores.
            bboxes: CUDA float32 tensor with shape ``[num_queries, 4]`` in
                normalized ``cx, cy, width, height`` format.
            class_mapping: CUDA int tensor with class remap entries. Negative
                mapped ids are ignored.
            metadata: CUDA float32 tensor with shape
                ``[num_queries * MAX_CLASSES_PER_QUERY, METADATA_STRIDE]``.
                Row ``query_index * MAX_CLASSES_PER_QUERY + class_rank`` holds
                the ``class_rank``-th highest passing class for that query.
                Columns have the same layout as
                ``_select_best_query_metadata_kernel``.
            records: CUDA int32 tensor with shape ``[MAX_TOTAL_RUNS + 1, 3]``.
                Program 0 resets ``records[0, 0]`` and ``records[0, 1]`` before
                the RLE kernel appends runs.
            threshold: Minimum class score required for a metadata row to be
                marked active.
            num_queries: Number of query rows in ``scores`` and ``bboxes``.
            num_classes: Number of class columns in ``scores``.
            class_mapping_size: Number of valid entries in ``class_mapping``.
            output_height: Original image height used for xyxy box conversion.
            output_width: Original image width used for xyxy box conversion.
            BLOCK_CLASSES: Power-of-two tile width covering all class columns.
            METADATA_STRIDE: Number of float32 fields per metadata row.
            MAX_CLASSES_PER_QUERY: Number of metadata rows reserved per query.
            FLAG_OVERFLOW_CLASSES: When true, writes ``records[0, 1] = 1`` if
                more than ``MAX_CLASSES_PER_QUERY`` classes pass threshold; the
                caller treats that as unsupported for exact top-k parity.
        """
        query_index = tl.program_id(0)
        if query_index == 0:
            tl.store(records + 0, 0)
            tl.store(records + 1, 0)

        class_offsets = tl.arange(0, BLOCK_CLASSES)
        class_active = class_offsets < num_classes
        mapped_classes = tl.load(
            class_mapping + class_offsets,
            mask=class_active & (class_offsets < class_mapping_size),
            other=-1,
        ).to(tl.int32)
        class_scores = tl.load(
            scores + query_index * num_classes + class_offsets,
            mask=class_active,
            other=-1.0,
        )
        passing_classes = (
            class_active & (mapped_classes >= 0) & (class_scores > threshold)
        )
        passing_class_count = tl.sum(tl.where(passing_classes, 1, 0), axis=0)
        if FLAG_OVERFLOW_CLASSES and passing_class_count > MAX_CLASSES_PER_QUERY:
            tl.store(records + 1, 1)

        work_scores = tl.where(passing_classes, class_scores, -1.0)
        for class_rank in tl.static_range(0, 4):
            # Repeated max-and-mask avoids sorting all classes and keeps the
            # register footprint bounded by the configured class block.
            selected_score = tl.max(work_scores, axis=0)
            selected_class = tl.max(
                tl.where(
                    work_scores == selected_score,
                    class_offsets,
                    -1,
                ),
                axis=0,
            ).to(tl.int32)
            mapped_class = tl.load(
                class_mapping + selected_class,
                mask=(selected_class >= 0) & (selected_class < class_mapping_size),
                other=-1,
            ).to(tl.int32)
            is_valid_detection = (mapped_class >= 0) & (selected_score > threshold)
            metadata_rank = query_index * MAX_CLASSES_PER_QUERY + class_rank
            meta_base = metadata_rank * METADATA_STRIDE
            selected_index = query_index * num_classes + selected_class

            tl.store(
                metadata + meta_base + 0,
                tl.where(is_valid_detection, 1.0, 0.0),
            )
            tl.store(metadata + meta_base + 1, mapped_class.to(tl.float32))
            tl.store(
                metadata + meta_base + 2,
                tl.where(is_valid_detection, selected_score, 0.0),
            )
            tl.store(metadata + meta_base + 7, 0.0)
            tl.store(metadata + meta_base + 8, 0.0)
            tl.store(metadata + meta_base + 9, query_index.to(tl.float32))
            tl.store(metadata + meta_base + 10, selected_index.to(tl.float32))
            tl.store(metadata + meta_base + 11, 0.0)
            tl.store(metadata + meta_base + 12, 0.0)
            tl.store(metadata + meta_base + 13, 0.0)
            tl.store(metadata + meta_base + 14, 0.0)
            tl.store(metadata + meta_base + 15, 0.0)

            bbox_base = query_index * 4
            cx = tl.load(bboxes + bbox_base, mask=is_valid_detection, other=0.0)
            cy = tl.load(bboxes + bbox_base + 1, mask=is_valid_detection, other=0.0)
            width = tl.load(
                bboxes + bbox_base + 2,
                mask=is_valid_detection,
                other=0.0,
            )
            height = tl.load(
                bboxes + bbox_base + 3,
                mask=is_valid_detection,
                other=0.0,
            )
            x1 = tl.maximum(
                0.0,
                tl.minimum((cx - 0.5 * width) * output_width, output_width),
            )
            y1 = tl.maximum(
                0.0,
                tl.minimum((cy - 0.5 * height) * output_height, output_height),
            )
            x2 = tl.maximum(
                0.0,
                tl.minimum((cx + 0.5 * width) * output_width, output_width),
            )
            y2 = tl.maximum(
                0.0,
                tl.minimum((cy + 0.5 * height) * output_height, output_height),
            )
            tl.store(metadata + meta_base + 3, x1)
            tl.store(metadata + meta_base + 4, y1)
            tl.store(metadata + meta_base + 5, x2)
            tl.store(metadata + meta_base + 6, y2)
            work_scores = tl.where(class_offsets == selected_class, -1.0, work_scores)

    @triton.jit
    def _sparse_atomic_rle_from_metadata_kernel(
        masks,
        y_idx,
        y_weight,
        x_idx,
        x_weight,
        metadata,
        records,
        num_queries: tl.constexpr,
        mask_height: tl.constexpr,
        mask_width: tl.constexpr,
        output_height: tl.constexpr,
        output_width: tl.constexpr,
        mask_stride_q: tl.constexpr,
        mask_stride_h: tl.constexpr,
        mask_stride_w: tl.constexpr,
        BLOCK_MASK: tl.constexpr,
        BLOCK_OUT_H: tl.constexpr,
        BLOCK_OUT_W: tl.constexpr,
        BLOCK_ROI_H: tl.constexpr,
        MAX_ROI_WIDTH: tl.constexpr,
        MAX_TOTAL_RUNS: tl.constexpr,
        METADATA_STRIDE: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        """Interpolate active mask ROIs and emit COCO-order RLE run records.

        Launch grid:
            ``(metadata_rows, ceil(MAX_ROI_WIDTH / BLOCK_COLS))``. Program id 0
            selects a metadata row / output detection rank. Program id 1 selects
            the starting column tile. If a mask ROI is wider than
            ``MAX_ROI_WIDTH``, each program advances by ``MAX_ROI_WIDTH`` in a
            loop so large ROIs are still covered without launching a fallback.

        Args:
            masks: CUDA float32 tensor with logical shape
                ``[num_queries, mask_height, mask_width]``. Strides are passed
                separately because callers may hand in contiguous or view-backed
                tensors. Values are mask logits already in the RF-DETR mask
                space; output pixels are positive when bilinear interpolation is
                greater than zero.
            y_idx: CUDA int32 tensor with shape ``[output_height, 2]``. For each
                output row, stores the two source mask rows used by the
                reference antialiased bilinear resize.
            y_weight: CUDA float32 tensor with shape ``[output_height, 2]``.
                Weights matching ``y_idx``.
            x_idx: CUDA int32 tensor with shape ``[output_width, 2]``. For each
                output column, stores the two source mask columns used by the
                reference resize.
            x_weight: CUDA float32 tensor with shape ``[output_width, 2]``.
                Weights matching ``x_idx``.
            metadata: CUDA float32 tensor with shape
                ``[metadata_rows, METADATA_STRIDE]``. The kernel reads column
                ``0`` active flag and column ``9`` source query id. It writes
                columns ``11:15`` with ``roi_y_start, roi_y_end, roi_x_start,
                roi_x_end`` for diagnostics.
            records: CUDA int32 tensor with shape ``[MAX_TOTAL_RUNS + 1, 3]``.
                ``records[0, 0]`` is atomically incremented for every emitted
                run and ``records[0, 1]`` is set when capacity is exceeded.
                Data rows are ``(rank, start, end)`` where ``start`` and ``end``
                are flat COCO/Fortran-order positions ``x * output_height + y``.
            num_queries: Number of query masks in ``masks``.
            mask_height: Height of each low-resolution RF-DETR mask.
            mask_width: Width of each low-resolution RF-DETR mask.
            output_height: Original image height for the output RLE mask.
            output_width: Original image width for the output RLE mask.
            mask_stride_q: Stride between query masks in ``masks``.
            mask_stride_h: Row stride for ``masks``.
            mask_stride_w: Column stride for ``masks``.
            BLOCK_MASK: Power-of-two tile covering ``mask_height * mask_width``
                so the kernel can find positive source support in one vector.
            BLOCK_OUT_H: Power-of-two tile covering all output rows.
            BLOCK_OUT_W: Power-of-two tile covering all output columns.
            BLOCK_ROI_H: Number of output rows scanned per inner row tile.
            MAX_ROI_WIDTH: Column band width handled per program before the
                large-ROI loop advances to the next band.
            MAX_TOTAL_RUNS: Maximum number of sparse runs that fit in
                ``records`` excluding the header row.
            METADATA_STRIDE: Number of float32 fields per metadata row.
            BLOCK_COLS: Number of output columns scanned together.
        """
        rank = tl.program_id(0)
        tile_x = tl.program_id(1)
        local_x_offsets = tile_x * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        meta_base = rank * METADATA_STRIDE
        is_valid_detection = tl.load(metadata + meta_base + 0) > 0.5
        query_index = tl.load(metadata + meta_base + 9).to(tl.int32)

        if not is_valid_detection:
            return

        mask_offsets = tl.arange(0, BLOCK_MASK)
        mask_active = mask_offsets < (mask_height * mask_width)
        source_y = mask_offsets // mask_width
        source_x = mask_offsets - source_y * mask_width
        mask_values = tl.load(
            masks
            + query_index * mask_stride_q
            + source_y * mask_stride_h
            + source_x * mask_stride_w,
            mask=mask_active,
            other=-1.0,
        )
        positive_source = mask_active & (mask_values > 0.0)
        # Any output pixel depending only on non-positive source pixels cannot
        # cross the >0 threshold, so derive the minimal candidate ROI from the
        # positive source support plus a one-pixel interpolation halo.
        source_y_min = tl.min(tl.where(positive_source, source_y, mask_height), axis=0)
        source_y_max = tl.max(tl.where(positive_source, source_y, -1), axis=0)
        source_x_min = tl.min(tl.where(positive_source, source_x, mask_width), axis=0)
        source_x_max = tl.max(tl.where(positive_source, source_x, -1), axis=0)
        has_positive_source = source_y_max >= 0
        if not has_positive_source:
            return

        source_y_min = tl.maximum(source_y_min - 1, 0)
        source_y_max = tl.minimum(source_y_max + 1, mask_height - 1)
        source_x_min = tl.maximum(source_x_min - 1, 0)
        source_x_max = tl.minimum(source_x_max + 1, mask_width - 1)

        out_y_offsets = tl.arange(0, BLOCK_OUT_H)
        y_active = out_y_offsets < output_height
        interp_y0 = tl.load(y_idx + out_y_offsets * 2, mask=y_active, other=-1)
        interp_y1 = tl.load(y_idx + out_y_offsets * 2 + 1, mask=y_active, other=-1)
        interp_y_weight0 = tl.load(
            y_weight + out_y_offsets * 2,
            mask=y_active,
            other=0.0,
        )
        interp_y_weight1 = tl.load(
            y_weight + out_y_offsets * 2 + 1,
            mask=y_active,
            other=0.0,
        )
        candidate_y = y_active & (
            (
                (interp_y0 >= source_y_min)
                & (interp_y0 <= source_y_max)
                & (interp_y_weight0 != 0.0)
            )
            | (
                (interp_y1 >= source_y_min)
                & (interp_y1 <= source_y_max)
                & (interp_y_weight1 != 0.0)
            )
        )
        roi_y_start = tl.min(
            tl.where(candidate_y, out_y_offsets, output_height), axis=0
        )
        roi_y_end = tl.max(tl.where(candidate_y, out_y_offsets + 1, 0), axis=0)

        out_x_offsets = tl.arange(0, BLOCK_OUT_W)
        x_active = out_x_offsets < output_width
        interp_x0 = tl.load(x_idx + out_x_offsets * 2, mask=x_active, other=-1)
        interp_x1 = tl.load(x_idx + out_x_offsets * 2 + 1, mask=x_active, other=-1)
        interp_x_weight0 = tl.load(
            x_weight + out_x_offsets * 2,
            mask=x_active,
            other=0.0,
        )
        interp_x_weight1 = tl.load(
            x_weight + out_x_offsets * 2 + 1,
            mask=x_active,
            other=0.0,
        )
        candidate_x = x_active & (
            (
                (interp_x0 >= source_x_min)
                & (interp_x0 <= source_x_max)
                & (interp_x_weight0 != 0.0)
            )
            | (
                (interp_x1 >= source_x_min)
                & (interp_x1 <= source_x_max)
                & (interp_x_weight1 != 0.0)
            )
        )
        roi_x_start = tl.min(tl.where(candidate_x, out_x_offsets, output_width), axis=0)
        roi_x_end = tl.max(tl.where(candidate_x, out_x_offsets + 1, 0), axis=0)
        roi_width = roi_x_end - roi_x_start

        if tile_x == 0:
            # ROI bounds are diagnostic/fallback metadata; one tile writes them
            # to avoid redundant stores from every column group.
            tl.store(metadata + meta_base + 11, roi_y_start.to(tl.float32))
            tl.store(metadata + meta_base + 12, roi_y_end.to(tl.float32))
            tl.store(metadata + meta_base + 13, roi_x_start.to(tl.float32))
            tl.store(metadata + meta_base + 14, roi_x_end.to(tl.float32))

        if (roi_y_start >= roi_y_end) or (roi_x_start >= roi_x_end):
            return

        x_band_offset = tile_x * BLOCK_COLS
        rows = tl.arange(0, BLOCK_ROI_H)
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask_base = query_index * mask_stride_q

        while x_band_offset < roi_width:
            local_x_offsets = x_band_offset + col_offsets
            column_active = local_x_offsets < roi_width
            output_x = roi_x_start + local_x_offsets
            output_x_matrix = output_x[None, :]
            x_base = output_x_matrix * 2
            source_x0 = tl.load(
                x_idx + x_base,
                mask=column_active[None, :],
                other=0,
            ).to(tl.int64)
            source_x1 = tl.load(
                x_idx + x_base + 1,
                mask=column_active[None, :],
                other=0,
            ).to(tl.int64)
            x_weight0 = tl.load(
                x_weight + x_base,
                mask=column_active[None, :],
                other=0.0,
            )
            x_weight1 = tl.load(
                x_weight + x_base + 1,
                mask=column_active[None, :],
                other=0.0,
            )

            # Open slots carry a run that began in a prior row tile but has not
            # ended yet. The slot stores the record index whose end is pending.
            open_slots = tl.full((BLOCK_COLS,), -1, tl.int32)
            y_tile_start = roi_y_start
            while y_tile_start <= roi_y_end:
                row_y = y_tile_start + rows
                output_y = row_y[:, None]
                active = (row_y[:, None] < roi_y_end) & column_active[None, :]
                boundary_active = (row_y[:, None] <= roi_y_end) & column_active[None, :]

                y_base = output_y * 2
                source_y0 = tl.load(y_idx + y_base, mask=active, other=0).to(tl.int64)
                source_y1 = tl.load(y_idx + y_base + 1, mask=active, other=0).to(
                    tl.int64
                )
                y_weight0 = tl.load(y_weight + y_base, mask=active, other=0.0)
                y_weight1 = tl.load(y_weight + y_base + 1, mask=active, other=0.0)
                value00 = tl.load(
                    masks
                    + mask_base
                    + source_y0 * mask_stride_h
                    + source_x0 * mask_stride_w,
                    mask=active,
                    other=0.0,
                )
                value10 = tl.load(
                    masks
                    + mask_base
                    + source_y1 * mask_stride_h
                    + source_x0 * mask_stride_w,
                    mask=active,
                    other=0.0,
                )
                value01 = tl.load(
                    masks
                    + mask_base
                    + source_y0 * mask_stride_h
                    + source_x1 * mask_stride_w,
                    mask=active,
                    other=0.0,
                )
                value11 = tl.load(
                    masks
                    + mask_base
                    + source_y1 * mask_stride_h
                    + source_x1 * mask_stride_w,
                    mask=active,
                    other=0.0,
                )
                current_values = (
                    value00 * y_weight0 + value10 * y_weight1
                ) * x_weight0 + (value01 * y_weight0 + value11 * y_weight1) * x_weight1
                current_positive = active & (current_values > 0.0)

                # Starts/ends are transitions along a COCO column-major scan:
                # current positive after previous background starts a run;
                # previous positive followed by current background ends it.
                previous_y = output_y - 1
                previous_active = boundary_active & (row_y[:, None] > roi_y_start)
                previous_y_base = previous_y * 2
                prev_source_y0 = tl.load(
                    y_idx + previous_y_base,
                    mask=previous_active,
                    other=0,
                ).to(tl.int64)
                prev_source_y1 = tl.load(
                    y_idx + previous_y_base + 1,
                    mask=previous_active,
                    other=0,
                ).to(tl.int64)
                prev_y_weight0 = tl.load(
                    y_weight + previous_y_base,
                    mask=previous_active,
                    other=0.0,
                )
                prev_y_weight1 = tl.load(
                    y_weight + previous_y_base + 1,
                    mask=previous_active,
                    other=0.0,
                )
                prev_value00 = tl.load(
                    masks
                    + mask_base
                    + prev_source_y0 * mask_stride_h
                    + source_x0 * mask_stride_w,
                    mask=previous_active,
                    other=0.0,
                )
                prev_value10 = tl.load(
                    masks
                    + mask_base
                    + prev_source_y1 * mask_stride_h
                    + source_x0 * mask_stride_w,
                    mask=previous_active,
                    other=0.0,
                )
                prev_value01 = tl.load(
                    masks
                    + mask_base
                    + prev_source_y0 * mask_stride_h
                    + source_x1 * mask_stride_w,
                    mask=previous_active,
                    other=0.0,
                )
                prev_value11 = tl.load(
                    masks
                    + mask_base
                    + prev_source_y1 * mask_stride_h
                    + source_x1 * mask_stride_w,
                    mask=previous_active,
                    other=0.0,
                )
                previous_values = (
                    prev_value00 * prev_y_weight0 + prev_value10 * prev_y_weight1
                ) * x_weight0 + (
                    prev_value01 * prev_y_weight0 + prev_value11 * prev_y_weight1
                ) * x_weight1
                previous_positive = previous_active & (previous_values > 0.0)
                is_start = current_positive & ~previous_positive
                is_end = previous_positive & ~current_positive
                start_prefix = tl.cumsum(tl.where(is_start, 1, 0), 0)
                end_prefix = tl.cumsum(tl.where(is_end, 1, 0), 0)
                start_count = tl.max(start_prefix, axis=0)
                end_count = tl.max(end_prefix, axis=0)

                start_slots = start_prefix - 1
                run_flat = (output_x_matrix * output_height + output_y).to(tl.int32)
                for col in tl.static_range(0, BLOCK_COLS):
                    col_match = col_offsets == col
                    col_has_starts = (
                        tl.max(tl.where(col_match & (start_count > 0), 1, 0), axis=0)
                        != 0
                    )
                    col_start_count = tl.max(
                        tl.where(col_match, start_count, 0), axis=0
                    ).to(tl.int32)
                    col_end_count = tl.max(
                        tl.where(col_match, end_count, 0), axis=0
                    ).to(tl.int32)
                    open_slot = tl.max(tl.where(col_match, open_slots, -1), axis=0).to(
                        tl.int32
                    )
                    open_at_start = open_slot >= 0
                    open_at_start_i = tl.where(open_at_start, 1, 0).to(tl.int32)
                    # Reserve a contiguous span for this column's new starts.
                    # Atomic ordering between columns is irrelevant because CPU
                    # sorts records by flat start before building COCO counts.
                    col_base = tl.atomic_add(
                        records + 0,
                        col_start_count,
                        sem="relaxed",
                        mask=col_has_starts,
                    ).to(tl.int32)
                    col_base = tl.where(col_has_starts, col_base, 0)
                    if col_has_starts & ((col_base + col_start_count) > MAX_TOTAL_RUNS):
                        tl.store(records + 1, 1)

                    start_record_slots = col_base + start_slots
                    start_store = (
                        is_start
                        & col_match[None, :]
                        & (start_record_slots < MAX_TOTAL_RUNS)
                    )
                    tl.store(
                        records + (start_record_slots + 1) * 3,
                        tl.full((BLOCK_ROI_H, BLOCK_COLS), rank, tl.int32),
                        mask=start_store,
                    )
                    tl.store(
                        records + (start_record_slots + 1) * 3 + 1,
                        run_flat,
                        mask=start_store,
                    )

                    current_end_slots = col_base + end_prefix - 1 - open_at_start_i
                    end_record_slots = tl.where(
                        open_at_start & (end_prefix == 1),
                        open_slot,
                        current_end_slots,
                    )
                    end_store = (
                        is_end
                        & col_match[None, :]
                        & (end_record_slots >= 0)
                        & (end_record_slots < MAX_TOTAL_RUNS)
                    )
                    tl.store(
                        records + (end_record_slots + 1) * 3 + 2,
                        run_flat,
                        mask=end_store,
                    )

                    closed_current_starts = tl.maximum(
                        col_end_count - open_at_start_i, 0
                    )
                    unmatched_current_starts = col_start_count - closed_current_starts
                    open_after = (open_at_start_i + col_start_count - col_end_count) > 0
                    next_open_slot = tl.where(
                        open_after,
                        tl.where(
                            unmatched_current_starts > 0,
                            col_base + col_start_count - 1,
                            open_slot,
                        ),
                        -1,
                    ).to(tl.int32)
                    open_slots = tl.where(col_match, next_open_slot, open_slots)

                y_tile_start += BLOCK_ROI_H
            x_band_offset += MAX_ROI_WIDTH
