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

The key observation that makes a *direct* transcode possible: supervision's
per-crop RLE (``_mask_to_rle_counts``) and COCO's uncompressed counts use the
**same encoding** — column-major run lengths starting with a leading
``False`` (background) run, alternating False/True. They differ only in *scope*
(full image vs. bbox crop). So converting one to the other is a matter of:

1. decompressing the COCO counts to plain run lengths (run arithmetic, NOT a
   pixel decode — see :func:`_decode_coco_counts`);
2. splitting the full-frame run list into per-column run lists
   (``_rle_split_cols``);
3. selecting the bbox's columns and trimming each to the bbox's rows
   (:func:`_trim_col_runs`);
4. re-joining the selected/trimmed columns into a flat crop RLE
   (``_rle_join_cols``).

No ``(H, W)`` or ``(N, H, W)`` array is ever allocated.

Parity contract
---------------
:func:`compact_mask_from_coco_rle` is defined to produce **exactly** what
``CompactMask.from_dense(coco_rle_masks_to_numpy_mask(rle), xyxy, image_shape)``
would produce — same box clipping (clip to ``[0, dim-1]``, inclusive max
coords), same invalid-box handling (``x2 < x1`` or ``y2 < y1`` -> ``1x1``
all-False crop) — just without the dense intermediate. The accompanying unit
test asserts decoded-mask equality against that reference.

Note on layering / upstreaming
-------------------------------
This reuses supervision's private RLE primitives (``_rle_split_cols`` /
``_rle_join_cols``) to guarantee identical junction-merge semantics. The natural
long-term home for :func:`compact_mask_from_coco_rle` is a
``CompactMask.from_coco_rle`` classmethod in supervision itself, at which point
those imports become internal. It lives here for now so the tensor pipeline can
adopt it without waiting on a supervision release.
"""

from typing import List, Sequence, Tuple, Union

import numpy as np

# Private supervision primitives. Isolated here so a single import site breaks
# if supervision relocates them (rather than scattering the coupling).
from supervision.detection.compact_mask import (
    CompactMask,
    _rle_join_cols,
    _rle_split_cols,
)

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


def _trim_col_runs(col_runs: Sequence[int], y1: int, y2: int) -> List[int]:
    """Restrict one full-height column run list to rows ``[y1, y2]`` inclusive.

    ``col_runs`` (as produced by ``_rle_split_cols``) starts with a ``False``
    count and alternates, summing to the full column height. The returned list
    covers ``y2 - y1 + 1`` rows and also starts with a ``False`` count (a
    leading ``0`` is inserted when the window begins on a ``True`` pixel),
    matching the convention ``_rle_join_cols`` expects.
    """
    want = y2 - y1 + 1
    collected: List[Tuple[bool, int]] = []
    row = 0
    for idx, run_len in enumerate(col_runs):
        is_true = idx % 2 == 1
        start = row
        end = row + int(run_len)
        row = end
        lo = max(start, y1)
        hi = min(end, y2 + 1)
        if hi > lo:
            collected.append((is_true, hi - lo))
        if row > y2:
            break

    if not collected:
        return [want]

    out: List[int] = []
    if collected[0][0]:  # window starts on True -> leading False count of 0
        out.append(0)
    for is_true, length in collected:
        last_is_true = bool(out) and ((len(out) - 1) % 2 == 1)
        if out and last_is_true == is_true:
            out[-1] += length
        else:
            out.append(length)
    return out


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
        if x2c < x1c or y2c < y1c:
            rles.append(np.array([1], dtype=np.int32))
            crop_shapes.append((1, 1))
            offsets.append((x1c, y1c))
            continue

        crop_h = y2c - y1c + 1
        crop_w = x2c - x1c + 1

        full_counts = _decode_coco_counts(masks_counts[i])
        # Split the full-frame F-order RLE into one run list per image column
        # (each column = img_h pixels). Pure run arithmetic — no pixel buffer.
        columns = _rle_split_cols(np.asarray(full_counts, dtype=np.int64), img_h, img_w)
        selected = [
            _trim_col_runs(columns[col], y1c, y2c) for col in range(x1c, x2c + 1)
        ]
        crop_rle = _rle_join_cols(selected, crop_h * crop_w)

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
