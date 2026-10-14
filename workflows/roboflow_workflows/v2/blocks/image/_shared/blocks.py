"""Native V2 image blocks.

Each block is one class that declares its identity, parameters and outputs.
A step may set every parameter to a literal, leave its default, or bind it to
a workflow input or an upstream output. ``run`` receives plain values, or a
``Batch`` for a ``Group`` field, and returns a mapping of ordinary results.

* ``roboflow_core/crop@v1``: image and rectangles -> ``crops`` along a new ``regions`` axis,
  and one ``summary`` per image.
* ``roboflow_core/static_crop@v1``: image and configured rectangles -> ``crops`` along a
  new stationary ``regions`` axis: position ``k`` is always rectangle ``k``.
* ``roboflow_core/resize@v1``: image -> ``image`` of a given size.
* ``roboflow_core/invert@v1``: image -> pixel-inverted ``image``.
* ``roboflow_core/mosaic@v1``: group of images -> one ``image`` canvas and its ``count`` per
  parent of the group, timed at the last image of the group.
* ``roboflow_core/has_brightness@v1``: image -> ``keep``, whether the mean pixel value
  reaches a minimum.

Image payloads are ``ImageData``: a ``(channels, height, width)`` ``uint8``
tensor with its provenance; see ``image_data``. Blocks keep tensors on their
device and never modify an input tensor.
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
