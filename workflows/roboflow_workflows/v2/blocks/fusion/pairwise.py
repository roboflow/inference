"""Overlap between two independent sets of boxes.

``geometry/overlap.py`` compares the rows of one prediction with each other.
Fusion blocks compare rows of one prediction with rows of another::

    boxes_a (N, 4) xyxy ─┐
                         ├─ box_iou_between ─> iou (N, M), float32, device of a
    boxes_b (M, 4) xyxy ─┘

``iou[i, j]`` is the IoU of ``boxes_a[i]`` and ``boxes_b[j]``. A pair whose union
is empty (two zero-area boxes) has IoU 0, as in V1, where supervision's NaN is
replaced by 0. Values are float32; V1 compared float64 values, so a pair exactly
at a threshold can decide differently in the last bit.

Mask overlap is not provided: no fusion block compares masks yet.
"""

import torch
import torchvision


def box_iou_between(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Return the IoU of every box in one set with every box in another.

    Args:
        boxes_a: ``(N, 4)`` xyxy boxes.
        boxes_b: ``(M, 4)`` xyxy boxes on the same device as ``boxes_a``.

    Returns:
        ``(N, M)`` float32 IoU on the device of the inputs; empty sets give an
        empty matrix.

    Raises:
        ValueError: If either input is not ``(K, 4)`` or the devices differ.
    """
    for name, boxes in (("boxes_a", boxes_a), ("boxes_b", boxes_b)):
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"{name} must have shape (K, 4), got {tuple(boxes.shape)}."
            )
    if boxes_a.device != boxes_b.device:
        raise ValueError(
            f"boxes_a is on {boxes_a.device} but boxes_b is on {boxes_b.device}."
        )

    iou = torchvision.ops.box_iou(boxes_a.to(torch.float32), boxes_b.to(torch.float32))
    iou = torch.nan_to_num(iou, nan=0.0)

    return iou
