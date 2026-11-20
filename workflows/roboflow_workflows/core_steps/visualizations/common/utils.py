import copy
import dataclasses
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils
import supervision as sv
from roboflow_workflows.core_steps.common.keypoints import (
    KEYPOINT_PADDING_CLASS_NAME,
    MAX_KEYPOINT_SLOTS,
    MAX_KEYPOINTS_PADDING_CELLS,
)
from roboflow_workflows.execution_engine.constants import RLE_MASK_KEY_IN_SV_DETECTIONS

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


def pad_key_points_to_edges(key_points: sv.KeyPoints, *, edges: Any) -> sv.KeyPoints:
    """Give every instance a slot for each vertex the 1-based ``edges`` name.

    Keypoint arrays end at the last slot any instance fills, so when no instance
    has a skeleton's last vertices, or any vertex at all, an edge to them points
    past the last slot, and supervision's edge annotator fails the run instead of
    skipping the edge. The added slots sit at (0, 0), which the annotator reads as
    a missing vertex.

    Args:
        key_points (sv.KeyPoints): Keypoints an edge annotator is about to draw.
        edges (Any): The annotator's edges, as pairs of 1-based vertex numbers.

    Returns:
        A copy with the added slots, or ``key_points`` itself when the edges fit,
        are not pairs of vertex numbers, or ask for more slots than the keypoint
        slot and padding limits allow.
    """
    try:
        needed = max((int(vertex) for edge in edges for vertex in edge), default=0)
    except (TypeError, ValueError):
        return key_points

    instances, slots = key_points.xy.shape[:2]
    if (
        instances == 0
        or needed <= slots
        or needed > MAX_KEYPOINT_SLOTS
        or instances * needed > MAX_KEYPOINTS_PADDING_CELLS
    ):
        return key_points

    def pad(values: Any) -> Any:
        if not isinstance(values, np.ndarray) or values.shape[:2] != (instances, slots):
            return values

        padding = [(0, 0), (0, needed - slots)] + [(0, 0)] * (values.ndim - 2)
        fill = KEYPOINT_PADDING_CLASS_NAME if values.dtype.kind in "OSU" else 0
        padded = np.pad(values, padding, constant_values=fill)
        return padded

    padded_key_points = dataclasses.replace(
        key_points,
        **{
            field.name: pad(getattr(key_points, field.name))
            for field in dataclasses.fields(key_points)
            if field.name != "data"
        },
        data={name: pad(values) for name, values in key_points.data.items()},
    )
    return padded_key_points


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
