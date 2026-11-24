"""Anchor points of rows, as ``sv.Detections.get_anchors_coordinates`` gives them.

Box anchors use the host boxes' dtype: float64 if the device boxes are
float64, else float32 (``rows.read_rows``), so truncation to integers
matches V1. ``CENTER_OF_MASS`` needs dense masks:
each centre is ``mean(pixel index + 0.5)`` over the mask. The device sums
``2 * index + 1`` and the pixel count as exact int64 (MPS has no float64);
one transfer brings them to the host, which divides in float64 as V1 did.
An empty mask gives ``(0, 0)`` as in V1.
"""

import numpy as np
import torch

CENTER_OF_MASS = "CENTER_OF_MASS"


def box_anchors(xyxy: np.ndarray, position: str) -> np.ndarray:
    """Return ``(N, 2)`` anchor points of boxes for a V1 position name.

    Args:
        xyxy: ``(N, 4)`` float boxes.
        position: ``CENTER``, ``CENTER_LEFT``, ``CENTER_RIGHT``,
            ``TOP_CENTER``, ``TOP_LEFT``, ``TOP_RIGHT``, ``BOTTOM_LEFT``,
            ``BOTTOM_CENTER`` or ``BOTTOM_RIGHT``.

    Returns:
        Float anchors in the boxes' dtype.

    Raises:
        ValueError: On another position.
    """
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
    by_position = {
        "CENTER": (centre_x, centre_y),
        "CENTER_LEFT": (x1, centre_y),
        "CENTER_RIGHT": (x2, centre_y),
        "TOP_CENTER": (centre_x, y1),
        "TOP_LEFT": (x1, y1),
        "TOP_RIGHT": (x2, y1),
        "BOTTOM_LEFT": (x1, y2),
        "BOTTOM_CENTER": (centre_x, y2),
        "BOTTOM_RIGHT": (x2, y2),
    }
    if position not in by_position:
        raise ValueError(f"unknown anchor position {position!r}")

    anchors = np.stack(by_position[position], axis=1)

    return anchors


def mass_centres(masks: torch.Tensor) -> np.ndarray:
    """Return each mask's centre of mass, truncated to integers as V1 does.

    Args:
        masks: ``(N, H, W)`` bool masks on any device.

    Returns:
        ``(N, 2)`` int64 ``(x, y)`` centres on the host.
    """
    count, height, width = masks.shape
    if count == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Doubled half-integer positions: sum(2 * index + 1) / (2 * count) is the
    # mean of index + 0.5, with every sum an exact integer.
    per_column = masks.sum(dim=1, dtype=torch.int64)
    per_row = masks.sum(dim=2, dtype=torch.int64)
    columns = 2 * torch.arange(width, dtype=torch.int64, device=masks.device) + 1
    rows = 2 * torch.arange(height, dtype=torch.int64, device=masks.device) + 1
    doubled_x = (per_column * columns).sum(dim=1)
    doubled_y = (per_row * rows).sum(dim=1)
    total = per_row.sum(dim=1).clamp(min=1)
    sums = torch.stack([doubled_x, doubled_y, total], dim=1).cpu().numpy()
    centres = (sums[:, :2] / (2.0 * sums[:, 2:3])).astype(np.int64)

    return centres
