"""What the shape blocks share: marker anchors and the outline-or-fill thickness.

Dot and triangle markers sit on one anchor point per row::

    position              anchor                         reads
    CENTER ... BOTTOM_*   a point of the row's box       boxes only
    CENTER_OF_MASS        the mean of the row's mask     dense masks (mask_quality)

Only ``CENTER_OF_MASS`` reads masks; every other position leaves a mask view
untouched. Anchors are truncated to integers as V1's ``int(xy)`` does.

Circle and ellipse outlines take OpenCV's thickness: ``>= 1`` pixels, or
``-1`` to fill the shape (``FILLED``).
"""

from typing import Any

import numpy as np
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import (
    CENTER_OF_MASS,
    box_anchors,
    mass_centres,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    DrawRows,
    dense_masks_of,
    has_masks,
    placed_in,
)
from roboflow_workflows.v2.data.image import ImageData

FILLED = -1


def checked_outline_or_fill(value: Any, *, name: str) -> int:
    """Return an outline thickness after checking it is ``>= 1`` or ``FILLED``.

    Raises:
        ValueError: For booleans, non-integers, ``0`` and values below ``-1``.
    """
    valid = isinstance(value, int) and not isinstance(value, bool)
    if not valid or (value < 1 and value != FILLED):
        raise ValueError(
            f"{name} must be an integer >= 1, or -1 to fill the shape, got {value!r}"
        )

    return value


def marker_anchors(
    predictions: Any,
    rows: DrawRows,
    *,
    position: str,
    mask_quality: str,
    image: ImageData,
    block: str,
) -> np.ndarray:
    """Return each row's integer anchor point in the image's pixels.

    Args:
        predictions: The block's predictions (any box carrier).
        rows: Host rows of the same predictions, placed in ``image``.
        position: A ``POSITIONS_WITH_MASS`` name.
        mask_quality: ``accurate`` or ``coarse``; read for ``CENTER_OF_MASS``.
        image: The image drawn on.
        block: Block type for error messages.

    Returns:
        ``(N, 2)`` int64 ``(x, y)`` anchors on the host.

    Raises:
        ValueError: For ``CENTER_OF_MASS`` on predictions without masks.
    """
    if position != CENTER_OF_MASS:
        anchors = box_anchors(rows.xyxy, position).astype(np.int64)
        return anchors

    if not has_masks(predictions):
        raise ValueError(
            f"{block}: position CENTER_OF_MASS needs instance masks; these "
            f"predictions ({type(predictions).__name__}) have none. Use a box "
            "position such as CENTER, or instance segmentation predictions."
        )

    dense = dense_masks_of(predictions, quality=mask_quality, block=block)
    anchors = mass_centres(placed_in(dense, image, block=block).mask)

    return anchors
