"""Centred static crops: one rectangle per image, in pixels or in fractions.

* ``roboflow_core/absolute_static_crop@v1``: centre and size in pixels.
* ``roboflow_core/relative_statoic_crop@v1``: centre and size as fractions of the image.

Both ports keep V1's rectangle arithmetic, including Python ``round`` (half to
even)::

    x_min = round(x_center - width / 2)      x_max = x_min + width
    y_min = round(y_center - height / 2)     y_max = y_min + height

The rectangle is clipped to the image. When nothing remains, the step has no
crop for that image and downstream steps skip it. A crop is a new image whose
provenance places it in its parent and root; source and time are kept.
Neither block adds a nesting axis: one image in, at most one crop out.
"""

from typing import Tuple


Rectangle = Tuple[int, int, int, int]

def centered_rectangle(
    x_center: int, *, y_center: int, width: int, height: int
) -> Rectangle:
    """Return the V1 rectangle ``(x_min, y_min, x_max, y_max)`` around a centre.

    Args:
        x_center: Horizontal centre in pixels.
        y_center: Vertical centre in pixels.
        width: Width in pixels.
        height: Height in pixels.

    Returns:
        Corners in pixels with exclusive upper bounds; not clipped. ``round``
        is Python's half-to-even rounding, as in V1.
    """
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    rectangle = (x_min, y_min, x_min + width, y_min + height)

    return rectangle
