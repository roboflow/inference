"""Rectangle type and validation shared by the crop and static crop blocks.

Used by ``roboflow_core/crop@v1`` and ``roboflow_core/static_crop@v1``.
Rectangles are ``[x0, y0, x1, y1]`` pixel coordinates with exclusive upper
bounds.
"""

from typing import Any, Sequence, Tuple


Rectangle = Tuple[int, int, int, int]

def _require_rectangles(regions: Sequence[Any]) -> Sequence[Any]:
    # Shared by literal and selected regions; checks without converting.
    for rectangle in regions:
        corners = list(rectangle) if isinstance(rectangle, (list, tuple)) else None
        if (
            corners is None
            or len(corners) != 4
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in corners
            )
        ):
            raise ValueError(f"rectangle {rectangle!r} must be four integers")
        x0, y0, x1, y1 = corners
        if x1 < x0 or y1 < y0:
            raise ValueError(f"rectangle {corners} must have x0 <= x1 and y0 <= y1")

    return regions
