"""Track trails: ``v2/trace_visualization@v1``.

Port of V1 ``roboflow_core/trace_visualization@v1`` (``sv.TraceAnnotator``).
Each current row with a ``tracker_id`` gets a line through that track's
anchors of the last ``trace_length`` calls for its source::

    state   per source: (call, tracker ids, anchors) of the last trace_length calls
    host    one transfer of boxes; anchors; cv2.polylines per track (Strokes)
    device  copy of the image; one indexed write of every line

Line pixels equal V1 for one source whose frames all have rows. Kept on
purpose, as V1: rows with ``tracker_id == -1`` (pending) are skipped, a track
missing from the current frame is not drawn but keeps its points until they
age out, and a single point draws nothing.

Different from V1, on purpose:

* Every source has its own history. V1 kept one history per parameter
  combination, shared by all videos.
* Points age by calls of the step for their source, with or without rows.
  A step runs once per processed frame, so this is one age per frame. V1
  dropped old points only once more than ``trace_length`` frames with rows
  were stored, so frames without rows did not age a trace.
* A selected ``trace_length`` or style may change between frames and keeps
  the history (a shorter length drops older points). V1 started an empty
  history for every new combination. A new ``position`` or image size
  restarts the source's traces: anchors of different positions or scales
  are not joined.
* The image must be the whole frame or a resize of it. A crop fails with a
  "draw before cropping" message (``sources``); V1 drew it.

State and its limits (``sources.SourceStates``): the source is
``video_metadata.video_identifier``, else the image's source in the engine,
else ``"default"``. A lower ``frame_number`` than the source's last restarts
it. ``max_sources`` bounds sources; nothing is evicted. ``reset_state()``
forgets every source. Calls of one step run in pulse order, which the trail
relies on, so the block is never pruned.
"""

from collections import deque
from typing import Any, Deque, Dict, Literal, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.anchors import (
    CENTER_OF_MASS,
    box_anchors,
    mass_centres,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    POSITIONS_WITH_MASS,
    BoxColorParams,
    MaskQuality,
    check_literals,
    checked_choice,
    checked_int,
    checked_mask_quality,
    mask_quality_field,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    boxes_of,
    dense_masks_of,
    has_masks,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.sources import SourceStates
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    bounds_scan,
    stroke_reach,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Ref

DEFAULT_MAX_SOURCES = 256
PENDING_TRACK_ID = -1
"""``sv.TraceAnnotator`` skips rows of tracks that are not confirmed yet."""


class _Trail:
    """Anchors of one source's last calls, oldest first."""

    def __init__(self) -> None:
        self.position: Optional[str] = None
        self.size_hw: Optional[Tuple[int, int]] = None
        self.call = 0
        self.frames: Deque[Tuple[int, np.ndarray, np.ndarray]] = deque()

    def record(
        self,
        tracker_ids: np.ndarray,
        anchors: np.ndarray,
        *,
        position: str,
        size_hw: Tuple[int, int],
        trace_length: int,
    ) -> None:
        if position != self.position or size_hw != self.size_hw:
            self.frames.clear()
            self.position = position
            self.size_hw = size_hw
        self.frames.append((self.call, tracker_ids, anchors))
        while self.frames[0][0] <= self.call - trace_length:
            self.frames.popleft()
        self.call += 1

    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Every kept ``(tracker ids, anchors)``, in frame order."""
        tracker_ids = np.concatenate([ids for _, ids, _ in self.frames])
        anchors = np.concatenate([points for _, _, points in self.frames])

        return tracker_ids, anchors


class TraceVisualizationBlock(AnnotationBlock):
    """Draw the recent path of each tracked detection onto a copy of the image.

    Needs ``tracker_id`` on every row (``v2/byte_tracker@v1``). A path joins
    the track's anchors of the last ``trace_length`` calls for the image's
    source. Give it each whole frame once (a resize is fine); crop after it.

    Example: with ``trace_length=3``, a track at x = 10, 20, 30, 40 on four
    frames draws 10-20 on the second frame, 10-20-30 on the third and
    20-30-40 on the fourth.
    """

    type = "v2/trace_visualization@v1"
    prunable = False
    metadata = {
        "name": "Trace Visualization",
        "port_of": "roboflow_core/trace_visualization@v1",
        "section": "visualization",
    }

    class Params(BoxColorParams):
        position: Literal[POSITIONS_WITH_MASS] | Ref(STRING_KIND) = Field(
            default="CENTER",
            description=(
                "Point of each detection the path joins. CENTER_OF_MASS needs "
                "instance masks and builds one full-image mask per detection "
                "(see mask_quality). A new value restarts the source's paths."
            ),
            examples=["CENTER", "BOTTOM_CENTER", "$inputs.position"],
        )
        trace_length: StrictInt | Ref(INTEGER_KIND) = Field(
            default=30,
            description=(
                "Calls for the source (one per frame) a path spans, the current "
                "one included; >= 1."
            ),
            examples=[30, "$inputs.trace_length"],
        )
        thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1,
            description="Line width in pixels; >= 1.",
            examples=[1, "$inputs.thickness"],
        )
        mask_quality: MaskQuality = mask_quality_field(
            reads="position is CENTER_OF_MASS, on instance masks"
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most video sources with a kept history. The first frame of one "
                "more fails with SourceCapacityError; no source is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "TraceVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def __init__(self) -> None:
        self._trails: SourceStates[_Trail] = SourceStates(_Trail)

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        color_palette: str,
        palette_size: int,
        custom_colors: Any,
        color_axis: str,
        position: str,
        trace_length: int,
        thickness: int,
        mask_quality: str,
        max_sources: int,
    ) -> Dict[str, Any]:
        """Add this frame's anchors to the source's history and draw the paths.

        Args:
            image: RGB whole frame or a resize of it; not modified.
            predictions: A box carrier (see ``rows``) with ``tracker_id`` rows.
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            position: Anchor of each row.
            trace_length: Calls a path spans.
            thickness: Line width.
            mask_quality: Mask quality for ``CENTER_OF_MASS``.
            max_sources: Most sources with a kept history.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, a cropped image, rows without
                ``tracker_id``, ``CENTER_OF_MASS`` without masks, or
                predictions of another workflow input.
            SourceCapacityError: On the first frame of one source too many.
        """
        values = dict(
            position=position,
            trace_length=trace_length,
            thickness=thickness,
            mask_quality=mask_quality,
        )
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
        )
        rows = read_rows(boxes)
        tracker_ids = self._tracker_ids(rows)
        anchors = self._anchors(
            predictions,
            image,
            rows=rows,
            position=position,
            mask_quality=mask_quality,
        )

        confirmed = tracker_ids != PENDING_TRACK_ID
        tracker_ids, anchors = tracker_ids[confirmed], anchors[confirmed]
        trail = self._trails.of(self, image, max_sources=max_sources)
        trail.record(
            tracker_ids,
            anchors,
            position=position,
            size_hw=image.size_hw,
            trace_length=trace_length,
        )

        colors = row_colors(
            rows.class_id[confirmed],
            tracker_ids,
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
        )
        kept_ids, kept_anchors = trail.points()
        strokes = Strokes(image.size_hw)
        for tracker_id, color in zip(tracker_ids.tolist(), colors):
            path = kept_anchors[kept_ids == tracker_id].astype(np.int32)
            if len(path) > 1:
                add_path(strokes, path, color=color, thickness=thickness)
        strokes.paint(canvas)
        result = annotated(image, canvas)

        return result

    def reset_state(self) -> None:
        """Forget every source's history."""
        self._trails.clear()

    def _tracker_ids(self, rows: Any) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros(0, dtype=np.int64)

        tracker_ids = rows.tracker_ids()
        if tracker_ids is None:
            raise ValueError(
                f"{self.type} needs tracker_id on every row; track the "
                "predictions first, e.g. with v2/byte_tracker@v1"
            )

        return tracker_ids

    def _anchors(
        self,
        predictions: Any,
        image: ImageData,
        *,
        rows: Any,
        position: str,
        mask_quality: str,
    ) -> np.ndarray:
        if position != CENTER_OF_MASS:
            anchors = box_anchors(rows.xyxy, position)
            return anchors

        if not has_masks(predictions):
            raise ValueError(
                f"{self.type}: position CENTER_OF_MASS needs instance masks; "
                "these predictions have none"
            )
        dense = dense_masks_of(predictions, quality=mask_quality, block=self.type)
        anchors = mass_centres(placed_in(dense, image, block=self.type).mask)

        return anchors


def add_path(strokes: Strokes, path: np.ndarray, *, color: Any, thickness: int) -> None:
    """Add one open polyline through integer points, as ``cv2.polylines`` draws it.

    Args:
        strokes: Strokes of the frame.
        path: ``(K, 2)`` int32 ``(x, y)`` points, oldest first.
        color: RGB color.
        thickness: Line width.
    """
    strokes.add(
        lambda mask: cv2.polylines(mask, [path], False, 1, thickness),
        color=color,
        scan=bounds_scan(path, reach=stroke_reach(thickness)),
    )


_CHECKS = {
    "position": lambda value: checked_choice(
        value, name="position", choices=POSITIONS_WITH_MASS
    ),
    "trace_length": lambda value: checked_int(value, name="trace_length", minimum=1),
    "thickness": lambda value: checked_int(value, name="thickness", minimum=1),
    "mask_quality": checked_mask_quality,
}
