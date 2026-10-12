"""Where each label goes and its cached sprite: the V1 label geometry.

A label is rendered once on the host with supervision's exact cv2 calls into
a ``_LabelSprite`` (the V1 tensor label path), then pasted. ``LabelPlacement``
picks the sprite and its canvas origin for one detection::

    interior label      one sprite per (text, color), with a margin all round
    label at an edge    a variant whose canvas ends at that frame edge
                        (cv2 anti-aliases clipped strokes differently)
    label outside       nothing is drawn

Sprites are LRU-cached (``SPRITE_CACHE_SIZE``). A cache is not thread-safe;
the engine calls one step instance at a time.
"""

from collections import OrderedDict
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv
import torch
from roboflow_workflows.core_steps.visualizations.label.v1_tensor import (
    _LabelMeasurement,
    _LabelSprite,
    _measure_label,
    _render_label_sprite,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.drawing import (
    PALETTE,
    DetectionDrawing,
)
from supervision.annotators.utils import resolve_text_background_xyxy

# V1 label defaults: white "<class> <confidence>" text on a box-colored patch
# above the box's top-left corner.
TEXT_COLOR_BGR = (255, 255, 255)
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
TEXT_PADDING = 10
BORDER_RADIUS = 0
LABEL_POSITION = sv.Position.TOP_LEFT

# Sprites are keyed by label text, color, clip window, device and frame width.
# A stream has a small vocabulary (class x two-digit confidence).
SPRITE_CACHE_SIZE = 256

# Label size is pure in (text, scale, thickness, padding), so a repeated label
# skips its cv2.getTextSize calls.
MEASUREMENT_CACHE_SIZE = 4096
_cached_measure_label = lru_cache(maxsize=MEASUREMENT_CACHE_SIZE)(_measure_label)

SpriteOrigin = Tuple[int, int]


class LabelPlacement:
    """Label sprites and their canvas origins, with a bounded sprite cache."""

    def __init__(self) -> None:
        self._sprites: "OrderedDict[tuple, _LabelSprite]" = OrderedDict()

    @property
    def cached_sprites(self) -> int:
        """Number of sprites currently cached (at most ``SPRITE_CACHE_SIZE``)."""
        return len(self._sprites)

    def sprites(
        self,
        drawing: DetectionDrawing,
        *,
        frame_hw: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[List[_LabelSprite], List[SpriteOrigin]]:
        """Sprites and canvas origins of every label of one image, in order.

        Args:
            drawing: Prepared detections of the image.
            frame_hw: ``(height, width)`` of the image.
            device: Device of the image; sprites hold their colors there.

        Returns:
            Sprites and their ``(x, y)`` canvas origins; labels entirely
            outside the frame are left out.
        """
        corners = drawing.xyxy[:, :2].astype(int)
        sprites: List[_LabelSprite] = []
        origins: List[SpriteOrigin] = []
        for label, (x, y), class_id in zip(drawing.labels, corners, drawing.class_id):
            placed = self._place(
                label,
                anchor=(int(x), int(y)),
                background_bgr=PALETTE.by_idx(int(class_id)).as_bgr(),
                frame_hw=frame_hw,
                device=device,
            )
            if placed is not None:
                sprites.append(placed[0])
                origins.append(placed[1])

        return sprites, origins

    def _place(
        self,
        label: str,
        *,
        anchor: Tuple[int, int],
        background_bgr: Tuple[int, int, int],
        frame_hw: Tuple[int, int],
        device: torch.device,
    ) -> Optional[Tuple[_LabelSprite, SpriteOrigin]]:
        # Port of LabelVisualizationBlockV1.run's per-label geometry.
        frame_h, frame_w = frame_hw
        measurement = _cached_measure_label(
            label, TEXT_SCALE, TEXT_THICKNESS, TEXT_PADDING
        )
        background_xyxy = resolve_text_background_xyxy(
            center_coordinates=anchor,
            text_wh=(measurement.width_padded, measurement.height_padded),
            position=LABEL_POSITION,
        )
        bx1, by1, bx2, by2 = (
            int(value) for value in np.asarray(background_xyxy, dtype=np.float32)
        )
        margin = measurement.margin

        sprite = self._sprite(
            label,
            measurement=measurement,
            background_bgr=background_bgr,
            device=device,
            frame_width=frame_w,
            box_in_canvas=(
                margin,
                margin,
                margin + measurement.width_padded,
                margin + measurement.height_padded,
            ),
            canvas_hw=(
                measurement.height_padded + 1 + 2 * margin,
                measurement.width_padded + 1 + 2 * margin,
            ),
            frame_edge_sides=(False, False, False, False),
        )
        origin = (bx1 - sprite.offset_x, by1 - sprite.offset_y)
        inside = (
            origin[0] + sprite.col_min >= 0
            and origin[1] + sprite.row_min >= 0
            and origin[0] + sprite.col_max < frame_w
            and origin[1] + sprite.row_max < frame_h
        )
        if inside:
            return sprite, origin

        window_x1, window_y1 = max(bx1 - margin, 0), max(by1 - margin, 0)
        window_x2 = min(bx2 + 1 + margin, frame_w)
        window_y2 = min(by2 + 1 + margin, frame_h)
        if window_x2 <= window_x1 or window_y2 <= window_y1:
            return None

        sprite = self._sprite(
            label,
            measurement=measurement,
            background_bgr=background_bgr,
            device=device,
            frame_width=frame_w,
            box_in_canvas=(
                bx1 - window_x1,
                by1 - window_y1,
                bx2 - window_x1,
                by2 - window_y1,
            ),
            canvas_hw=(window_y2 - window_y1, window_x2 - window_x1),
            frame_edge_sides=(
                bx1 - margin < 0,
                by1 - margin < 0,
                bx2 + 1 + margin > frame_w,
                by2 + 1 + margin > frame_h,
            ),
        )
        origin = (bx1 - sprite.offset_x, by1 - sprite.offset_y)

        return sprite, origin

    def _sprite(
        self,
        label: str,
        *,
        measurement: _LabelMeasurement,
        background_bgr: Tuple[int, int, int],
        device: torch.device,
        frame_width: int,
        box_in_canvas: Tuple[int, int, int, int],
        canvas_hw: Tuple[int, int],
        frame_edge_sides: Tuple[bool, bool, bool, bool],
    ) -> _LabelSprite:
        # The reused sprite also caches flat pixel indices by image width.
        # Keying by width keeps that inner cache to one entry per LRU sprite.
        key = (
            label,
            background_bgr,
            str(device),
            frame_width,
            box_in_canvas,
            canvas_hw,
            frame_edge_sides,
        )
        sprite = self._sprites.get(key)
        if sprite is not None:
            self._sprites.move_to_end(key)
            return sprite

        sprite = _render_label_sprite(
            measurement=measurement,
            text_color_bgr=TEXT_COLOR_BGR,
            background_color_bgr=background_bgr,
            text_scale=TEXT_SCALE,
            text_thickness=TEXT_THICKNESS,
            text_padding=TEXT_PADDING,
            border_radius=BORDER_RADIUS,
            device=device,
            box_in_canvas=box_in_canvas,
            canvas_hw=canvas_hw,
            frame_edge_sides=frame_edge_sides,
        )
        self._sprites[key] = sprite
        if len(self._sprites) > SPRITE_CACHE_SIZE:
            self._sprites.popitem(last=False)

        return sprite
