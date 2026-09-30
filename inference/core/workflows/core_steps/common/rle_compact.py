"""Zero-densification bridge: inference-models full-frame COCO RLE masks
(``InstancesRLEMasks``) -> supervision ``CompactMask`` (per-crop RLE).

Why this exists
---------------
``inference_models.InstanceDetections`` can carry masks as ``InstancesRLEMasks``:
a list of **full-frame**, **column-major (Fortran-order)** COCO RLE byte strings
(one per instance), as produced by ``pycocotools``. The visualisation seam used
to turn those into an ``sv.Detections`` by decoding the whole stack to a dense
``(N, H, W)`` boolean array (``coco_rle_masks_to_numpy_mask`` ->
``pycocotools.mask.decode``). That is O(N·H·W) memory and time and is the
dominant cost when visualising many instances on high-resolution frames.

``supervision.CompactMask`` stores each mask as an RLE of its **bounding-box
crop** instead of a full ``(H, W)`` array, and its annotators paint directly
into the crop region (``_paint_masks_by_area``) — no full-frame allocation.

Compressed foreground runs are clipped to the requested bounding box and
translated directly into crop coordinates. Background columns outside the crop
are skipped without constructing per-column lists or decoding mask pixels.

The result is pixel-identical to ``CompactMask.from_dense`` with the same boxes,
including clipping and degenerate-box handling.
"""

from typing import List, Sequence, Tuple, Union

import numpy as np
from supervision.detection.compact_mask import CompactMask

from inference_models.models.base.types import InstancesRLEMasks


def _decode_coco_counts(counts: Union[bytes, str]) -> List[int]:
    """Decompress a COCO compressed-RLE ``counts`` string to plain run lengths.

    Inverse of ``pycocotools``' ``rleToString`` (``maskApi.c``): a LEB128-style
    codec using 5 payload bits per character (ascii offset 48), a continuation
    bit (0x20), a sign bit (0x10) on the final char, and a delta against the
    value two positions back for every run from index 3 onward.

    Returns the uncompressed, column-major (F-order) run lengths: ``[False_run,
    True_run, False_run, ...]`` summing to ``H*W``. This is pure integer
    arithmetic — it never materialises pixels.
    """
    data = counts.encode("ascii") if isinstance(counts, str) else bytes(counts)
    cnts: List[int] = []
    p = 0
    n = len(data)
    m = 0
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            c = data[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= (-1) << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    return cnts


def _crop_rle_counts(
    full_counts: Sequence[int],
    image_height: int,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    """Clip foreground intervals in F-order; never expand background columns."""
    x1, y1, x2, y2 = box
    crop_h = y2 - y1 + 1
    crop_w = x2 - x1 + 1
    first = x1 * image_height
    stop = (x2 + 1) * image_height
    position = 0
    previous_end = 0
    output = [0]
    for index, length in enumerate(full_counts):
        end = position + length
        if index % 2:
            cursor = max(position, first)
            limit = min(end, stop)
            while cursor < limit:
                col, row = divmod(cursor, image_height)
                column_end = min(limit, (col + 1) * image_height)
                lo = max(row, y1)
                hi = min(column_end - col * image_height, y2 + 1)
                if hi > lo:
                    start = (col - x1) * crop_h + lo - y1
                    size = hi - lo
                    if len(output) == 1:
                        output[0] = start
                        output.append(size)
                    elif start == previous_end:
                        output[-1] += size
                    else:
                        output.extend((start - previous_end, size))
                    previous_end = start + size
                cursor = column_end
        position = end
        if position >= stop:
            break
    trailing = crop_h * crop_w - previous_end
    if len(output) == 1:
        output[0] = trailing
    elif trailing:
        output.append(trailing)
    return np.asarray(output, dtype=np.int32)


def compact_mask_from_coco_rle(
    image_shape: Tuple[int, int],
    masks_counts: Sequence[Union[bytes, str]],
    xyxy: np.ndarray,
) -> CompactMask:
    """Build a :class:`CompactMask` from full-frame COCO RLE counts, no densify.

    Args:
        image_shape: ``(H, W)`` of the full image.
        masks_counts: one COCO compressed-RLE ``counts`` per instance,
            column-major, full-frame (``InstancesRLEMasks.masks``).
        xyxy: ``(N, 4)`` boxes ``[x1, y1, x2, y2]`` (supervision inclusive-max
            convention), used as the crop bounds — identical to
            ``CompactMask.from_dense``.

    Returns:
        A :class:`CompactMask` decode-equal to
        ``CompactMask.from_dense(decode(masks_counts), xyxy, image_shape)``.
    """
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    num_masks = len(masks_counts)

    if num_masks == 0:
        return CompactMask(
            [],
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            (img_h, img_w),
        )

    rles: List[np.ndarray] = []
    crop_shapes: List[Tuple[int, int]] = []
    offsets: List[Tuple[int, int]] = []

    for i in range(num_masks):
        x1, y1, x2, y2 = xyxy[i]
        x1c = int(max(0, min(int(x1), img_w - 1)))
        y1c = int(max(0, min(int(y1), img_h - 1)))
        x2c = int(max(0, min(int(x2), img_w - 1)))
        y2c = int(max(0, min(int(y2), img_h - 1)))

        # Mirror CompactMask.from_dense's degenerate-box handling exactly.
        # The CLIPPED coordinates decide this, not the raw box, and the order is
        # load-bearing: clipping resurrects boxes like [0, 0, -1, -1] into a legal
        # 1x1 crop at the origin (max(0, min(-1, w - 1)) == 0), so testing the raw
        # box here would emit the all-False fallback where from_dense emits the
        # real pixel — a silent ONE-pixel divergence at [0, 0]. See
        # test_degenerate_box_decides_on_clipped_coordinates.
        if x2c < x1c or y2c < y1c:
            rles.append(np.array([1], dtype=np.int32))
            crop_shapes.append((1, 1))
            offsets.append((x1c, y1c))
            continue

        crop_h = y2c - y1c + 1
        crop_w = x2c - x1c + 1

        full_counts = _decode_coco_counts(masks_counts[i])
        crop_rle = _crop_rle_counts(full_counts, img_h, (x1c, y1c, x2c, y2c))

        rles.append(crop_rle)
        crop_shapes.append((crop_h, crop_w))
        offsets.append((x1c, y1c))

    return CompactMask(
        rles,
        np.array(crop_shapes, dtype=np.int32),
        np.array(offsets, dtype=np.int32),
        (img_h, img_w),
    )


def instances_rle_to_compact_mask(
    masks: InstancesRLEMasks,
    xyxy: np.ndarray,
) -> CompactMask:
    """Adapter: ``InstancesRLEMasks`` -> ``CompactMask`` (zero densification).

    ``xyxy`` must be the full-frame boxes for the same instances, in the
    supervision inclusive-max convention (the visualisation seam already has
    them as a host numpy array).
    """
    return compact_mask_from_coco_rle(
        image_shape=masks.image_size,
        masks_counts=masks.masks,
        xyxy=xyxy,
    )
