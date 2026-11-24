"""Per-image NMS for a whole batch, as ``torchvision.ops.batched_nms`` per image.

The base decoder calls ``batched_nms`` once per image, and every call waits for the
GPU (a scalar upload and ``masked_select``). ``segmented_nms`` does the same
per-image work with one kernel pass for all images and no wait::

    torchvision 0.28, per image b              segmented_nms, per batch
    m = boxes_b.max()                          B reductions (same op)
    boxes_b + ids_b.to(f32) * (m + 1)          same float32 ops, all rows at once
    scores_b.sort(stable, descending)          B sorts (same call)
    IoU(i, j) > float32(thr) bitmask, greedy   kernels.keep: every image, one pass
    order.masked_select(keep)[:max_det]        rows (B, cap), kept (B,) on device

An image torchvision sends to its vanilla branch (``boxes.numel()`` above its
limit) goes through ``torchvision.ops.batched_nms`` itself, and so does every
image of a call whose bitmask scratch would exceed the limit. Both are counted
in ``SegmentedNms.routes``; neither hides a kernel failure.

Arithmetic and ordering are read from the v0.28.0 tag (``ops/boxes.py``,
``csrc/ops/cuda/nms_kernel.cu``). The installed binary decides exactness; see
``calibration.calibrate``.
"""

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

import torch
import torchvision

WORD_BITS = 64
# Bitmask bytes per call above which every image takes the library route.
DEFAULT_SCRATCH_LIMIT_BYTES = 64 << 20


class NmsKernels(Protocol):
    """Greedy NMS of every image at once; ``nms_kernels.TritonNmsKernels`` in use."""

    def keep(
        self,
        boxes_for_nms: torch.Tensor,
        order: torch.Tensor,
        *,
        starts: torch.Tensor,
        counts: torch.Tensor,
        max_count: int,
        iou_threshold: float,
        rows: torch.Tensor,
        kept: torch.Tensor,
    ) -> None:
        """Write each image's kept rows, in NMS order, without a host wait.

        Args:
            boxes_for_nms: (K, 4) float32, class offsets applied, image-major.
            order: (K,) int64; ``order[starts[b] + i]`` is the image-local row
                of image b's i-th highest score (stable, descending).
            starts: (B,) int64 first row of each image.
            counts: (B,) int64 rows per image; 0 means skip the image.
            max_count: Host bound on ``counts``.
            iou_threshold: Suppress when float32 IoU is above float32(this).
            rows: (B, cap) int64 output; ``rows[b, t] = starts[b] + order[...]``
                for the t-th kept box, ``t < kept[b]``. Other entries untouched.
            kept: (B,) int64 output; boxes kept per image, at most ``cap``.
        """

    def identity(self) -> Dict[str, Any]:
        """JSON-friendly backend identity (kernel source, arithmetic, versions)."""


@dataclass(frozen=True)
class SegmentedNms:
    """Kept rows of every image; what ran is in ``routes``.

    ``rows[b, :kept[b]]`` are global row ids in image b's NMS order. Entries past
    ``kept[b]`` are 0 (a valid row) and never belong to a result. ``kept`` stays
    on the device; reading it on the host is the caller's (one) wait.
    """

    rows: torch.Tensor
    kept: torch.Tensor
    routes: Dict[str, int]
    scratch_bytes: int


def segmented_nms(
    xyxy: torch.Tensor,
    scores: torch.Tensor,
    nms_ids: torch.Tensor,
    *,
    image_index: torch.Tensor,
    counts: Sequence[int],
    counts_device: torch.Tensor,
    iou_threshold: float,
    max_detections: int,
    kernels: NmsKernels,
    scratch_limit_bytes: int = DEFAULT_SCRATCH_LIMIT_BYTES,
) -> SegmentedNms:
    """``batched_nms(...)[:max_detections]`` of every image, one kernel pass.

    Args:
        xyxy: (K, 4) float32 boxes, image-major, ascending anchors per image.
        scores: (K,) float32 scores of the same rows.
        nms_ids: (K,) int64 NMS class ids (zeros for class-agnostic NMS).
        image_index: (K,) int64 image of each row.
        counts: Rows per image, on the host (from the existing count wait).
        counts_device: The same counts as a (B,) int64 device tensor.
        iou_threshold: NMS IoU threshold.
        max_detections: Boxes kept per image, ``>= 0``.
        kernels: The greedy NMS backend.
        scratch_limit_bytes: Bitmask bytes above which the library route runs.

    Returns:
        Kept rows (B, cap) with ``cap = min(max_detections, max(counts))``,
        kept counts on the device, route counts and the bitmask bytes.
    """
    batch_size = len(counts)
    cap = min(max_detections, max(counts, default=0))
    rows = torch.zeros((batch_size, cap), dtype=torch.long, device=xyxy.device)
    kept = torch.zeros((batch_size,), dtype=torch.long, device=xyxy.device)
    routes: Counter = Counter(images_empty=counts.count(0))
    if cap == 0:
        return SegmentedNms(rows=rows, kept=kept, routes=dict(routes), scratch_bytes=0)

    # torchvision batched_nms: boxes.numel() > (4000 if cpu else 100_000) -> vanilla.
    vanilla_limit = 4000 if xyxy.device.type == "cpu" else 100_000
    library = [image for image, count in enumerate(counts) if 4 * count > vanilla_limit]
    routes["library_vanilla_branch_images"] = len(library)
    kernel = [
        image for image, count in enumerate(counts) if count and image not in library
    ]
    max_count = max((counts[image] for image in kernel), default=0)
    scratch_bytes = mask_scratch_bytes(batch_size, max_count=max_count)
    if scratch_bytes > scratch_limit_bytes:
        routes["library_scratch_limit_calls"] = 1
        routes["library_scratch_limit_images"] = len(kernel)
        library, kernel, scratch_bytes = sorted(library + kernel), [], 0
    routes["images_kernel"] = len(kernel)

    starts = list(itertools.accumulate(counts, initial=0))[:-1]
    if kernel:
        boxes_for_nms = _coordinate_trick(
            xyxy, nms_ids, image_index=image_index, counts=counts, images=kernel
        )
        order = _sorted_order(scores, counts=counts, starts=starts, images=kernel)
        kernel_counts = counts_device
        if library:
            kernel_counts = counts_device.clone()
            for image in library:
                kernel_counts[image] = 0
        kernels.keep(
            boxes_for_nms,
            order,
            # The same values as ``starts``, computed on the device: no upload wait.
            starts=counts_device.cumsum(0) - counts_device,
            counts=kernel_counts,
            max_count=max_count,
            iou_threshold=iou_threshold,
            rows=rows,
            kept=kept,
        )

    for image in library:
        # The original per-image call; this waits for the GPU.
        segment = slice(starts[image], starts[image] + counts[image])
        keep = torchvision.ops.batched_nms(
            xyxy[segment], scores[segment], nms_ids[segment], iou_threshold
        )[:cap]
        rows[image, : keep.numel()] = keep + segment.start
        kept[image] = keep.numel()

    result = SegmentedNms(
        rows=rows, kept=kept, routes=dict(routes), scratch_bytes=scratch_bytes
    )

    return result


def mask_scratch_bytes(batch_size: int, *, max_count: int) -> int:
    """Bytes of the padded IoU bitmask: (B, max_count, ceil(max_count / 64)) int64.

    Args:
        batch_size: Images in the call.
        max_count: Largest candidate count of an image on the kernel route.

    Returns:
        The allocation size in bytes.
    """
    words = math.ceil(max_count / WORD_BITS)
    size = batch_size * max_count * words * 8

    return size


def _coordinate_trick(
    xyxy: torch.Tensor,
    nms_ids: torch.Tensor,
    *,
    image_index: torch.Tensor,
    counts: Sequence[int],
    images: List[int],
) -> torch.Tensor:
    # torchvision _batched_nms_coordinate_trick for the given images at once:
    #     offsets = idxs.to(boxes) * (boxes.max() + torch.tensor(1).to(boxes))
    #     boxes_for_nms = boxes + offsets[:, None]
    # Each image's max is its own Tensor.max() (NaN propagates as there); the
    # add, multiply and add are the same float32 operations, one per element.
    # Rows of other images get offsets from a placeholder max and are not read.
    placeholder = xyxy.new_zeros(())
    maxima = torch.stack(
        [
            chunk.max() if image in images else placeholder
            for image, chunk in enumerate(xyxy.split(list(counts)))
        ]
    )
    offsets = nms_ids.to(xyxy) * (maxima + 1)[image_index]
    boxes_for_nms = xyxy + offsets[:, None]

    return boxes_for_nms


def _sorted_order(
    scores: torch.Tensor,
    *,
    counts: Sequence[int],
    starts: Sequence[int],
    images: List[int],
) -> torch.Tensor:
    # nms_kernel.cu sorts with scores.sort(/*stable=*/true, /*dim=*/0,
    # /*descending=*/true). The same call on the same contiguous values picks
    # the same algorithm, so ties and signed zeros order identically. Results
    # land in place: image-local positions at each image's own rows.
    order = torch.empty_like(scores, dtype=torch.long)
    values = torch.empty_like(scores)
    for image in images:
        segment = slice(starts[image], starts[image] + counts[image])
        torch.sort(
            scores[segment],
            stable=True,
            dim=0,
            descending=True,
            out=(values[segment], order[segment]),
        )

    return order
