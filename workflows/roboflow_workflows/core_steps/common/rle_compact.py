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
into the crop region — no full-frame allocation. ``CompactMask.from_coco_rle``
transcodes full-frame COCO RLE to that crop-scoped RLE with run-length
arithmetic, so no ``(N, H, W)`` array is ever allocated. This module only adapts
``InstancesRLEMasks`` to its input format.

Parity contract
---------------
:func:`compact_mask_from_coco_rle` produces **exactly** what
``CompactMask.from_dense(coco_rle_masks_to_numpy_mask(rle), xyxy, image_shape)``
would produce — same box clipping (clip to ``[0, dim-1]``, inclusive max
coords), same degenerate-box handling (judged on the raw integer box:
``x2 < x1``, ``y2 < y1``, or a box entirely outside the image -> ``1x1``
all-False crop) — just without the dense intermediate. The accompanying unit
test asserts decoded-mask equality against that reference.
"""

from typing import Sequence, Tuple, Union

import numpy as np
from supervision.detection.compact_mask import CompactMask

from inference_models.models.base.types import InstancesRLEMasks


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

    Raises:
        ValueError: If a ``counts`` entry is malformed or does not cover
            ``H*W`` pixels, ``xyxy`` is not shaped ``(len(masks_counts), 4)``,
            or ``image_shape`` is non-positive or exceeds supervision's
            per-side limit.
    """
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    rles = [{"size": [img_h, img_w], "counts": counts} for counts in masks_counts]

    compact_mask = CompactMask.from_coco_rle(rles, xyxy, (img_h, img_w))

    return compact_mask


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
