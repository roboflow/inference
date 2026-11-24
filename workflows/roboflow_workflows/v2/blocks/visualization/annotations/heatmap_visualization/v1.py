"""Accumulated presence: ``roboflow_core/heatmap_visualization@v1``.

Port of V1 ``roboflow_core/heatmap_visualization@v1`` (``sv.HeatMapAnnotator``
plus V1's filter of stationary tracks). Every drawn row adds 1 to a disc of
``radius`` around its anchor in its source's heat; the frame shows all heat
so far::

    state   per source: int32 heat counts (H, W); last moved anchor per track
    host    one transfer of boxes; anchors; cv2.circle discs into the heat;
            hue = low - heat / max(heat) * (low - top); cv2.blur; HSV -> RGB
    device  copy of the image; one upload of the color layer and the region
            heat > 0; blend with opacity (``blend``)

The host part is a naive reference: it reads and writes full-frame host
arrays on every call, whatever the device. Heat colors equal V1; the blend is
within 1 level of ``cv2.addWeighted`` (``blend``).

Stationary tracks, as V1: with ``ignore_stationary`` and a ``tracker_id`` on
every row, a track is drawn only on a frame where its anchor is at least
``motion_threshold`` pixels from the anchor kept for it, which then becomes
the kept anchor. A track's first frame is never drawn. Rows without tracker
ids are all drawn.

Different from V1, on purpose:

* The region drawn is ``heat > 0``. V1 tested ``heat.astype(uint8) > 0``,
  which hides pixels whose heat is a multiple of 256.
* One heat per source, kept when a selected style value changes. V1 started an
  empty heat for every new parameter combination. A new ``position`` forgets
  the kept anchors (they are of the old position), not the heat.
* A kept anchor is forgotten 60 seconds after its track was last seen,
  measured in ``video_metadata.frame_timestamp`` (the process clock for
  images without video metadata). V1 forgot it 60 s of wall-clock time after
  the track last moved, checked every 10 s.
* A source whose image size changes starts a new heat. V1 failed.
* Hues are OpenCV 8-bit hues, 0 to 179 (0 red, 60 green, 120 blue).
* Every call adds one frame of heat to its source, so the image must be the
  whole frame or a resize of it. A crop fails with a "draw before cropping"
  message (``sources``); V1 drew it.

State and its limits are those of ``sources.SourceStates``. Heat costs
``4 x H x W`` bytes per source (8.3 MB at 1920 x 1080) until a processing
reset. ``max_sources`` defaults to 32 sources (265 MB at 1080p); a higher value
is allowed. The 33rd source fails before any heat is allocated for it.
"""

import time
from typing import Any, Dict, Literal, Mapping, Optional, Tuple
import cv2
import numpy as np
import torch
from pydantic import Field, StrictBool, StrictInt, ValidationError, model_validator
from pydantic_core import PydanticCustomError
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import box_anchors
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.blend import blend
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    check_literals,
    checked_choice,
    checked_int,
    checked_opacity,
    reject_copy_image,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    boxes_of,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.sources import SourceStates
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


HEAT_POSITIONS = (
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_CENTER",
    "BOTTOM_LEFT",
    "BOTTOM_RIGHT",
)

MAX_HUE = 179

DEFAULT_MAX_SOURCES = 32

KEPT_ANCHOR_SECONDS = 60.0

_METADATA_MESSAGE = (
    "metadata is not a V2 parameter; delete the field from the step. The "
    "source is the image's own video_metadata."
)

class _Heat:
    """One source's heat counts and kept anchors of tracks."""

    def __init__(self) -> None:
        self.counts: Optional[np.ndarray] = None
        self.position: Optional[str] = None
        # tracker id -> (x, y) of its last drawn anchor, and when it was last seen
        self.anchors: Dict[int, Tuple[Any, Any, float]] = {}

    def fit(self, size_hw: Tuple[int, int], *, position: str) -> None:
        """Start a new heat for a new size; forget anchors of another position."""
        if self.counts is None or self.counts.shape != tuple(size_hw):
            self.counts = np.zeros(size_hw, dtype=np.int32)
            self.anchors.clear()
        if position != self.position:
            self.anchors.clear()
            self.position = position

    def moving(
        self,
        tracker_ids: np.ndarray,
        anchors: np.ndarray,
        *,
        threshold: int,
        now: float,
    ) -> np.ndarray:
        """Return which rows moved; keep each moved or new track's anchor."""
        stale = [
            tracker_id
            for tracker_id, (_, _, seen) in self.anchors.items()
            if now - seen > KEPT_ANCHOR_SECONDS
        ]
        for tracker_id in stale:
            del self.anchors[tracker_id]

        # Distances in the anchors' dtype (float32 for float32 boxes), as V1.
        moved = np.zeros(len(tracker_ids), dtype=bool)
        for row, tracker_id in enumerate(tracker_ids.tolist()):
            x, y = anchors[row]
            kept = self.anchors.get(tracker_id)
            if kept is not None:
                moved[row] = (
                    np.sqrt((x - kept[0]) ** 2 + (y - kept[1]) ** 2) >= threshold
                )
            if kept is None or moved[row]:
                self.anchors[tracker_id] = (x, y, now)
            else:
                self.anchors[tracker_id] = (kept[0], kept[1], now)

        return moved

    def add(self, anchors: np.ndarray, *, radius: int) -> None:
        """Add 1 inside each disc; overlapping discs of one frame add 1 once."""
        if len(anchors) == 0:
            return

        discs = np.zeros(self.counts.shape, dtype=np.uint8)
        for x, y in anchors.tolist():
            cv2.circle(discs, (int(x), int(y)), radius, 1, thickness=-1)
        self.counts += discs

class HeatmapVisualizationBlock(Block):
    """Draw where detections have been, accumulated per video source.

    Every call adds the current rows' discs to the source's heat and returns
    a copy of the image with all its heat so far: hottest areas in
    ``top_hue``, coolest in ``low_hue``, blended with ``opacity``. Give it
    each whole frame once (a resize is fine); crop after it.

    Example: a tracked car crossing a still road leaves a blue trail that
    turns red where it waited (when ``ignore_stationary`` is false) or stays
    absent there (true, the default).
    """

    type = "roboflow_core/heatmap_visualization@v1"
    prunable = False
    metadata = {
        "name": "Heatmap Visualization",
        "port_of": "roboflow_core/heatmap_visualization@v1",
        "section": "visualization",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description=(
                "New annotated image with the input's identity, frame, source, "
                "time and parent row."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "RGB image to draw on; not modified. The output is a new image "
                "with the same identity, source and time."
            ),
            examples=["$inputs.image", "$steps.mask_visualization.image"],
        )
        predictions: Ref(*BOX_KINDS) = Field(
            description=(
                "Detections, instance segmentation (dense, RLE or view) or "
                "keypoint predictions with boxes, of this image or of another "
                "frame of the same workflow input. Masks are never read."
            ),
            examples=["$steps.byte_tracker.tracked_detections"],
        )
        position: Literal[HEAT_POSITIONS] | Ref(STRING_KIND) = Field(
            default="BOTTOM_CENTER",
            description="Point of each detection's box that adds heat.",
            examples=["BOTTOM_CENTER", "CENTER", "$inputs.position"],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.2,
            description="Weight of the heat colors over the image, in [0, 1].",
            examples=[0.2, "$inputs.opacity"],
        )
        radius: StrictInt | Ref(INTEGER_KIND) = Field(
            default=40,
            description="Radius in pixels of the disc each detection adds; >= 0.",
            examples=[40, "$inputs.radius"],
        )
        kernel_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=25,
            description="Side of the box blur of the heat colors; >= 1 (1: none).",
            examples=[25, "$inputs.kernel_size"],
        )
        top_hue: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description="OpenCV hue (0-179) of the hottest pixels; 0 is red.",
            examples=[0, "$inputs.top_hue"],
        )
        low_hue: StrictInt | Ref(INTEGER_KIND) = Field(
            default=125,
            description="OpenCV hue (0-179) of the coolest heat; 120 is blue.",
            examples=[125, "$inputs.low_hue"],
        )
        ignore_stationary: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Add heat for a tracked row only when it moved motion_threshold "
                "pixels since it last added heat; a track's first frame adds "
                "none. Needs tracker_id on every row; otherwise every row adds heat."
            ),
            examples=[True, "$inputs.ignore_stationary"],
        )
        motion_threshold: StrictInt | Ref(INTEGER_KIND) = Field(
            default=25,
            description="Pixels a track must move to add heat again; >= 0.",
            examples=[25, "$inputs.motion_threshold"],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most video sources with a kept heat (4 x H x W bytes each, "
                "8.3 MB at 1080p). The first frame of one more fails with "
                "SourceCapacityError before its heat is allocated."
            ),
            examples=[DEFAULT_MAX_SOURCES, 128],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="before")
        @classmethod
        def _reject_metadata(cls, values: Any) -> Any:
            if isinstance(values, Mapping) and "metadata" in values:
                removed = PydanticCustomError("removed_parameter", _METADATA_MESSAGE)
                raise ValidationError.from_exception_data(
                    cls.__name__,
                    [
                        {
                            "type": removed,
                            "loc": ("metadata",),
                            "input": values["metadata"],
                        }
                    ],
                )
            return values

        @model_validator(mode="after")
        def _check_literals(self) -> "HeatmapVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def __init__(self) -> None:
        self._heats: SourceStates[_Heat] = SourceStates(_Heat)

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        position: str,
        opacity: float,
        radius: int,
        kernel_size: int,
        top_hue: int,
        low_hue: int,
        ignore_stationary: bool,
        motion_threshold: int,
        max_sources: int,
    ) -> Dict[str, Any]:
        """Add this frame's discs to the source's heat and draw the heat.

        Args:
            image: RGB whole frame or a resize of it; not modified.
            predictions: A box carrier (see ``rows``).
            position: Anchor of each row.
            opacity: Weight of the heat colors.
            radius: Disc radius.
            kernel_size: Box blur side.
            top_hue: Hue of the hottest pixels.
            low_hue: Hue of the coolest heat.
            ignore_stationary: Skip tracks that did not move.
            motion_threshold: Pixels a track must move.
            max_sources: Most sources with a kept heat.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, a cropped image or predictions
                of another workflow input.
            SourceCapacityError: On the first frame of one source too many.
        """
        values = dict(
            position=position,
            opacity=opacity,
            radius=radius,
            kernel_size=kernel_size,
            top_hue=top_hue,
            low_hue=low_hue,
            ignore_stationary=ignore_stationary,
            motion_threshold=motion_threshold,
        )
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
        )
        rows = read_rows(boxes)
        anchors = box_anchors(rows.xyxy, position)

        heat = self._heats.of(self, image, max_sources=max_sources)
        heat.fit(image.size_hw, position=position)
        tracker_ids = rows.tracker_ids()
        if ignore_stationary and tracker_ids is not None:
            moved = heat.moving(
                tracker_ids,
                anchors,
                threshold=motion_threshold,
                now=_source_time(image),
            )
            anchors = anchors[moved]
        heat.add(anchors, radius=radius)

        if heat.counts.max() > 0:
            layer = heat_colors(
                heat.counts, top_hue=top_hue, low_hue=low_hue, kernel_size=kernel_size
            )
            canvas = blend(
                canvas,
                torch.from_numpy(layer).permute(2, 0, 1).to(canvas.device),
                opacity=opacity,
                region=torch.from_numpy(heat.counts > 0).to(canvas.device),
            )
        result = annotated(image, canvas)

        return result

    def reset_state(self) -> None:
        """Forget every source's heat and kept anchors."""
        self._heats.clear()

def heat_colors(
    counts: np.ndarray, *, top_hue: int, low_hue: int, kernel_size: int
) -> np.ndarray:
    """Return the RGB color of every pixel's heat, as ``sv.HeatMapAnnotator``.

    Args:
        counts: ``(H, W)`` heat with a positive maximum.
        top_hue: Hue of the maximum.
        low_hue: Hue of zero heat.
        kernel_size: Box blur side applied to the hues.

    Returns:
        ``(H, W, 3)`` uint8 RGB colors (full saturation and value).
    """
    hues = low_hue - counts / counts.max() * (low_hue - top_hue)
    hues = cv2.blur(hues.astype(np.uint8), (kernel_size, kernel_size))
    hsv = np.full(counts.shape + (3,), 255, dtype=np.uint8)
    hsv[..., 0] = hues
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return colors

def _source_time(image: ImageData) -> float:
    if image.video_metadata is None:
        return time.monotonic()

    seconds = image.video_metadata.frame_timestamp.timestamp()

    return seconds

def _checked_hue(value: Any, *, name: str) -> int:
    hue = checked_int(value, name=name, minimum=0)
    if hue > MAX_HUE:
        raise ValueError(
            f"{name} must be an OpenCV hue from 0 to {MAX_HUE}, got {hue!r}"
        )

    return hue

def _checked_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"ignore_stationary must be true or false, got {value!r}")

    return value

_CHECKS = {
    "position": lambda value: checked_choice(
        value, name="position", choices=HEAT_POSITIONS
    ),
    "opacity": checked_opacity,
    "radius": lambda value: checked_int(value, name="radius", minimum=0),
    "kernel_size": lambda value: checked_int(value, name="kernel_size", minimum=1),
    "top_hue": lambda value: _checked_hue(value, name="top_hue"),
    "low_hue": lambda value: _checked_hue(value, name="low_hue"),
    "ignore_stationary": _checked_bool,
    "motion_threshold": lambda value: checked_int(
        value, name="motion_threshold", minimum=0
    ),
}
