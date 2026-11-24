"""Class-aware duplicate removal for stitched predictions (NMS and NMM).

The rules follow supervision 0.29, which V1 ``detections_stitch`` uses::

    overlap = mask IoU or box IoU, chosen by the caller (``use_masks``)
    nms: visit rows by confidence, highest first; a kept row removes later
         rows of its class with overlap > threshold; survivors keep input order
    nmm: per class (ascending id), take the most confident remaining row as
         candidate; rows with overlap >= threshold join its group. For masks
         the candidate grows to the union of its group and the test repeats.
         Each group becomes one row: union box, area-weighted confidence,
         class and row metadata of its most confident member, union mask.

Ties in confidence are broken as supervision's stable sort breaks them: the
later row comes first. Mask overlaps use the full-resolution masks; supervision
first resizes masks to at most 640 pixels, so decisions can differ for masks
that overlap near the threshold.

Overlaps are computed on the predictions' device; the boolean decision matrix
or one count per merge step goes to the host, where the greedy loop runs.
"""

from dataclasses import replace
from typing import Any, List, Optional

import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.rle import decode_rle_masks

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

# Pixels per chunk of the mask-intersection products, times the row count,
# bounded so one chunk stays near 64 MiB of float32.
CHUNK_ELEMENTS = 16 * 1024 * 1024


def non_max_suppression(detections: Any, *, threshold: float, use_masks: bool) -> Any:
    """Remove lower-confidence duplicates of the same class.

    Args:
        detections: ``Detections`` or ``InstanceDetections``.
        threshold: Overlap above which the less confident row is removed.
        use_masks: Compare masks instead of boxes; needs instance masks.

    Returns:
        The surviving rows, all fields aligned, in input order.
    """
    rows = len(detections.xyxy)
    if rows == 0:
        return detections

    overlaps = _overlaps(detections, use_masks=use_masks)
    overlapping = (overlaps > threshold) & _same_class(detections)
    overlapping = overlapping.cpu().numpy()
    keep = np.ones(rows, dtype=bool)
    for row in _by_confidence(detections.confidence)[::-1]:
        if keep[row]:
            suppressed = overlapping[row].copy()
            suppressed[row] = False
            keep &= ~suppressed

    survivors = select_predictions(detections, indices=np.flatnonzero(keep).tolist())

    return survivors


def non_max_merge(detections: Any, *, threshold: float, use_masks: bool) -> Any:
    """Merge groups of overlapping rows of the same class into one row each.

    Args:
        detections: ``Detections`` or ``InstanceDetections``.
        threshold: Overlap from which rows join a group.
        use_masks: Compare masks instead of boxes; needs instance masks.

    Returns:
        One row per group, ordered by class id, then by group confidence.
    """
    if len(detections.xyxy) == 0:
        return detections

    # One decode of RLE masks serves both the grouping and the mask union.
    masks = _dense_masks(detections) if use_masks else None
    groups = _merge_groups(
        detections, threshold=threshold, use_masks=use_masks, masks=masks
    )
    merged = _merge_rows(detections, groups=groups, dense_masks=masks)

    return merged


def merge_groups(
    detections: Any, *, threshold: float, use_masks: bool
) -> List[List[int]]:
    """Return the NMM groups as row positions, most confident member first.

    Args:
        detections: ``Detections`` or ``InstanceDetections``, at least one row.
        threshold: Overlap from which rows join a group.
        use_masks: Compare masks instead of boxes; needs instance masks.

    Returns:
        Groups ordered by class id, then by candidate confidence.
    """
    masks = _dense_masks(detections) if use_masks else None
    groups = _merge_groups(
        detections, threshold=threshold, use_masks=use_masks, masks=masks
    )

    return groups


def _merge_groups(
    detections: Any,
    *,
    threshold: float,
    use_masks: bool,
    masks: Optional[torch.Tensor],
) -> List[List[int]]:
    # ``masks`` are the bool masks from ``_dense_masks`` when ``use_masks``.
    class_ids = detections.class_id.cpu().numpy()
    order = _by_confidence(detections.confidence)
    box_overlaps = (
        None if use_masks else _overlaps(detections, use_masks=False).cpu().numpy()
    )
    groups = []
    for class_id in np.unique(class_ids):
        remaining = [row for row in order if class_ids[row] == class_id]
        while remaining:
            candidate = remaining.pop()
            if masks is None:
                # Most confident first, as supervision lists group members.
                joined = [
                    row
                    for row in reversed(remaining)
                    if box_overlaps[candidate, row] >= threshold
                ]
            else:
                joined = _grow_mask_group(masks, candidate, remaining, threshold)
            joined_set = set(joined)
            remaining = [row for row in remaining if row not in joined_set]
            groups.append([candidate, *joined])

    return groups


def _grow_mask_group(
    masks: torch.Tensor, candidate: int, remaining: List[int], threshold: float
) -> List[int]:
    # supervision's union-growing rule: whoever overlaps the current union
    # joins, the union grows, repeat until nobody joins.
    union = masks[candidate]
    joined: List[int] = []
    pending = list(remaining)
    while pending:
        others = masks[pending]
        intersection = (others & union).flatten(1).sum(dim=1)
        union_area = (others | union).flatten(1).sum(dim=1)
        iou = (intersection / union_area.clamp(min=1)).cpu().numpy()
        above = iou >= threshold
        if not above.any():
            break
        newcomers = [row for row, hit in zip(pending, above) if hit]
        # supervision lists newcomers by descending position in its order.
        joined.extend(reversed(newcomers))
        union = union | masks[newcomers].any(dim=0)
        pending = [row for row, hit in zip(pending, above) if not hit]

    return joined


def _merge_rows(
    detections: Any,
    *,
    groups: List[List[int]],
    dense_masks: Optional[torch.Tensor],
) -> Any:
    winners = [group[0] for group in groups]
    merged = select_predictions(detections, indices=winners)
    xyxy = detections.xyxy
    confidence = detections.confidence
    boxes, confidences = [], []
    for group in groups:
        members = xyxy[group].to(torch.float32)
        boxes.append(
            torch.cat(
                (members[:, :2].min(dim=0).values, members[:, 2:].max(dim=0).values)
            )
        )
        areas = (members[:, 2] - members[:, 0]) * (members[:, 3] - members[:, 1])
        scores = confidence[group].to(torch.float64)
        total = areas.sum().to(torch.float64)
        # Area-weighted mean; the winner's own confidence when areas are zero.
        weighted = (areas.to(torch.float64) * scores).sum() / total
        confidences.append(torch.where(total > 0, weighted, scores[0]))

    changes = {
        "xyxy": torch.stack(boxes).to(xyxy.dtype),
        "confidence": torch.stack(confidences).to(torch.float32).to(confidence.dtype),
        "bboxes_metadata": (
            None
            if merged.bboxes_metadata is None
            else [without_host_mirrors(row) for row in merged.bboxes_metadata]
        ),
    }
    if isinstance(detections, InstanceDetections) and detections.mask is not None:
        changes["mask"] = _union_masks(
            detections.mask, groups=groups, dense_masks=dense_masks
        )

    result = replace(merged, **changes)

    return result


def _union_masks(
    mask: Any, *, groups: List[List[int]], dense_masks: Optional[torch.Tensor]
) -> Any:
    if isinstance(mask, InstancesRLEMasks):
        merged = _union_rle_masks(mask, groups=groups, dense_masks=dense_masks)
        return merged

    unions = torch.stack([mask[group].any(dim=0) for group in groups]).to(mask.dtype)

    return unions


def _union_rle_masks(
    mask: InstancesRLEMasks,
    *,
    groups: List[List[int]],
    dense_masks: Optional[torch.Tensor],
) -> InstancesRLEMasks:
    # Host boundary: RLE masks are validated, decoded, OR-ed per group and
    # encoded. ``dense_masks``, when given, is that validated host decode.
    dense = decode_rle_masks(mask) if dense_masks is None else dense_masks.numpy()
    encoded = [
        mask_utils.encode(np.asfortranarray(dense[group].any(axis=0).astype(np.uint8)))[
            "counts"
        ]
        for group in groups
    ]
    merged = InstancesRLEMasks(image_size=mask.image_size, masks=encoded)

    return merged


def _by_confidence(confidence: torch.Tensor) -> List[int]:
    # Ascending, stable: the most confident row is last; among equals the
    # later row is last, as supervision's argsort orders them.
    order = torch.sort(confidence.cpu(), stable=True).indices.tolist()

    return order


def _same_class(detections: Any) -> torch.Tensor:
    class_id = detections.class_id
    same = class_id[:, None] == class_id[None, :]

    return same


def _overlaps(detections: Any, *, use_masks: bool) -> torch.Tensor:
    if not use_masks:
        boxes = detections.xyxy.to(torch.float32)
        return torchvision.ops.box_iou(boxes, boxes)

    flat = _dense_masks(detections).flatten(1)
    rows, pixels = flat.shape
    areas = flat.sum(dim=1, dtype=torch.float32)
    intersection = torch.zeros((rows, rows), dtype=torch.float32, device=flat.device)
    chunk = max(1, CHUNK_ELEMENTS // max(rows, 1))
    for start in range(0, pixels, chunk):
        part = flat[:, start : start + chunk].to(torch.float32)
        intersection += part @ part.T
    union = areas[:, None] + areas[None, :] - intersection
    iou = intersection / union.clamp(min=1)

    return iou


def _dense_masks(detections: Any) -> Any:
    mask = getattr(detections, "mask", None)
    if mask is None:
        return None
    if isinstance(mask, InstancesRLEMasks):
        # Host boundary: RLE masks are validated and decoded for the overlap test.
        return torch.from_numpy(decode_rle_masks(mask))

    return mask.to(torch.bool)
