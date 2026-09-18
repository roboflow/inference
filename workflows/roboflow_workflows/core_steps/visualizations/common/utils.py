import copy

import numpy as np
import pycocotools.mask as mask_utils
import supervision as sv

from roboflow_workflows.execution_engine.constants import (
    RLE_MASK_KEY_IN_SV_DETECTIONS,
)

UNKNOWN_CLASS_COLOR = sv.Color.GREY
UNKNOWN_CLASS_COLOR_RGB = UNKNOWN_CLASS_COLOR.as_rgb()


class NegativeSafeColorPalette(sv.ColorPalette):
    """Color palette that paints unknown (negative) ids in a fixed gray.

    Supervision's ``ColorPalette.by_idx`` raises on negative indices. Workflow
    parsers keep unmatched VLM labels as ``class_id == -1`` while preserving
    the model's original string in ``class_name``. This palette renders those
    detections in a neutral gray so they stay visible without colliding with a
    real class color.

    Non-negative indices keep the usual wrap-around palette lookup.
    """

    def by_idx(self, idx: int) -> sv.Color:
        """Return the color for ``idx``, or gray when ``idx`` is negative.

        Args:
            idx: Palette index. Negative values (unknown class or pending
                track) map to ``UNKNOWN_CLASS_COLOR``.

        Returns:
            Palette color for a non-negative index, or gray for a negative
            index.

        Raises:
            ValueError: If the palette is empty and ``idx`` is non-negative.
        """
        if idx < 0:
            return UNKNOWN_CLASS_COLOR
        if not self.colors:
            raise ValueError("A color palette must contain at least one color.")
        return self.colors[idx % len(self.colors)]


def ensure_dense_masks(detections: sv.Detections) -> sv.Detections:
    """Decode RLE side-channel masks so Supervision annotators can read ``mask``.

    RLE-first blocks return ``mask=None`` with COCO RLE in ``data["rle_mask"]``.
    Returns a shallow copy with ``mask`` set; the input is unchanged.
    """
    if (
        detections.mask is not None
        or RLE_MASK_KEY_IN_SV_DETECTIONS not in detections.data
    ):
        return detections
    detections = copy.copy(detections)
    detections.mask = np.array(
        [
            mask_utils.decode(rle).astype(bool)
            for rle in detections.data[RLE_MASK_KEY_IN_SV_DETECTIONS]
        ]
    )
    return detections


def wrap_color_palette(palette: sv.ColorPalette) -> NegativeSafeColorPalette:
    """Wrap a palette so negative ids resolve to gray instead of raising.

    Args:
        palette: Palette produced by a visualization block's ``getPalette``.

    Returns:
        A ``NegativeSafeColorPalette`` with the same colors. The input is
        returned unchanged when it is already a safe palette.
    """
    if isinstance(palette, NegativeSafeColorPalette):
        return palette
    return NegativeSafeColorPalette(colors=list(palette.colors))


def str_to_color(color: str) -> sv.Color:
    if color.startswith("#"):
        return sv.Color.from_hex(color)
    elif color.startswith("rgb"):
        r, g, b = map(int, color[4:-1].split(","))
        return sv.Color.from_rgb_tuple((r, g, b))
    elif color.startswith("bgr"):
        b, g, r = map(int, color[4:-1].split(","))
        return sv.Color.from_bgr_tuple((b, g, r))
    elif hasattr(sv.Color, color.upper()):
        return getattr(sv.Color, color.upper())
    else:
        raise ValueError(
            f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
        )
