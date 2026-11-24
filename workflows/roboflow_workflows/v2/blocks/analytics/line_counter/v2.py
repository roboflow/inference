"""``roboflow_core/line_counter@v2``: count tracked rows crossing a line, per source.

Port of V1 ``roboflow_core/line_counter@v2``; its tensor variant is the
reference. The V1 ``detections`` input is ``predictions`` here. Per source
and tracker id, on every call::

    side   = zones.line_sides(row's anchors, line)    +1 / -1 / 0
    0      -> nothing recorded (straddling, or beside the segment)
    +1, -1 -> compared with the track's last recorded side, then recorded
              last -1, now +1: crossed in    last +1, now -1: crossed out

``+1`` is the left of ``start -> end`` as drawn on screen: for a line drawn
left to right, an object moving up crosses in. Without ``triggering_anchor``
all four box corners must change side, so a box crosses only once it has
fully passed the line. A track's first recorded side never counts. Rows
are recorded in input order, so a tracker id repeated within one call is
recorded once per row, as in V1. ``count_in`` / ``count_out`` are the source's
totals so far; ``detections_in`` / ``detections_out`` are this call's
crossing rows, in input order, same family, rows unchanged.

State: one line and anchor choice per source, created on its first call,
empty prediction included. A source whose ``line_segment`` or
``triggering_anchor`` changes starts again from zero counts and no history:
recorded sides are only comparable for the same line and box points. V1
kept the first line and anchor forever. Re-sending the same anchor, through
a literal or a selector, keeps the counts. Each source remembers the last
side of at most ``MAX_TRACKS_PER_SOURCE`` tracks. After each call, the
tracks least recently seen are forgotten down to that bound; the call's own
tracks are always kept. A forgotten track needs two recorded frames again.

Capacity fails loudly, never by evicting a source or a track of the current
call. ``LineCounterCapacityError`` is raised, before this call changes any
state, when:

* a new source would exceed the literal ``max_sources`` (default 256), or
* one call has more than ``MAX_TRACKS_PER_SOURCE`` distinct tracker ids.

A processing reset replaces the block and ``reset_state()`` forgets every
source.

An empty prediction returns both counts and passes the empty prediction on
as both crossing outputs. It changes no counts or sides, except that a
changed ``line_segment`` or ``triggering_anchor`` restarts the source, as
on any call. Every row needs a ``tracker_id``.

Source key: ``video_metadata.video_identifier``, else the ``image`` input's
own ``SampleContext.source_id``, else ``"default"`` (also for a plain
``run()`` outside the engine).

CPU boundary: the boxes are copied to the host once per non-empty call as
float64 (V1's precision; line sides are exact for integer coordinates). The
crossing rows are then gathered on the boxes' device; a segmentation view
stays lazy.

Differences from V1:

* A changed ``line_segment`` or ``triggering_anchor`` restarts the source's
  counts (V1 ignored both). Without the restart, a static box straddling
  the line would count a crossing when the anchor moves to its other side.
* Tracks and sources are bounded (V1 kept every track and source).
* Bad lines and anchors fail at compile time when literal; V1 checked in
  ``run``. Booleans and non-finite coordinates are rejected.
* A segmentation view is accepted.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import torch
from pydantic import Field, StrictInt, field_validator
from roboflow_workflows.v2.blocks.analytics import zones
from roboflow_workflows.v2.blocks.query.detections import boxes_of, select_rows
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.masks import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.prediction_ops import RowSelection
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


__all__ = [
    "DEFAULT_SOURCE",
    "LineCounterBlock",
    "LineCounterCapacityError",
    "MAX_TRACKED_SOURCES",
    "MAX_TRACKS_PER_SOURCE",
]

DEFAULT_SOURCE = "default"

MAX_TRACKED_SOURCES = 256

MAX_TRACKS_PER_SOURCE = 4096

TRACKER_ID_KEY = "tracker_id"

COUNTED_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)

Anchor = Literal[zones.ANCHORS]

class LineCounterCapacityError(RuntimeError):
    """A new source or one call's tracks would exceed the line counter's bounds."""

@dataclass
class _SourceLine:
    """Crossing state of one source: its line, anchors, totals and track sides."""

    line: zones.Line
    # Box points tested against the line: one anchor, or the four corners.
    anchors: Tuple[str, ...]
    count_in: int = 0
    count_out: int = 0
    # tracker id -> last recorded side (+1 / -1), least recently seen first.
    sides: "OrderedDict[int, int]" = field(default_factory=OrderedDict)

class LineCounterBlock(Block):
    """Count tracked rows crossing a line segment, per source.

    Example, line ``[[0, 50], [100, 50]]``, ``triggering_anchor="CENTER"``, one
    track moving down then up::

        frame             1     2     3     4
        centre y          40    60    60    45
        side              +1    -1    -1    +1
        count_in          0     0     0     1
        count_out         0     1     1     1
        detections_out    -     row   -     -
    """

    type = "roboflow_core/line_counter@v2"
    metadata = {
        "name": "Line Counter",
        "port_of": "roboflow_core/line_counter@v2",
        "section": "video",
    }
    outputs = {
        "count_in": Output(
            INTEGER_KIND,
            description=(
                "Crossings in so far for this source; a track crossing in "
                "twice counts twice (not unique objects)."
            ),
        ),
        "count_out": Output(
            INTEGER_KIND,
            description=(
                "Crossings out so far for this source; a track crossing out "
                "twice counts twice (not unique objects)."
            ),
        ),
        "detections_in": Output(
            *COUNTED_KINDS,
            source="predictions",
            description="Rows that crossed in on this frame; same family.",
        ),
        "detections_out": Output(
            *COUNTED_KINDS,
            source="predictions",
            description="Rows that crossed out on this frame; same family.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Frame whose video_metadata names the source."
        )
        predictions: Ref(*COUNTED_KINDS) = Field(
            description=(
                "Tracked rows with tracker_id in bboxes_metadata, e.g. a "
                "tracker's tracked_detections (V1 input 'detections')."
            ),
            examples=["$steps.byte_tracker.tracked_detections"],
        )
        line_segment: list | Ref(LIST_OF_VALUES_KIND) = Field(
            description=(
                "Two [x, y] pixel points. Crossing to the left of start -> end "
                "as drawn counts in: for [[0, 100], [500, 100]] moving up is in. "
                "Changing it restarts this source's counts."
            ),
            examples=[[[0, 100], [500, 100]], "$inputs.line"],
        )
        triggering_anchor: Optional[Anchor] | Ref(STRING_KIND) = Field(
            default=None,
            description=(
                "Box point that must cross, e.g. CENTER or BOTTOM_CENTER. "
                "None: all four corners must cross. Changing it restarts this "
                "source's counts."
            ),
            examples=[None, "CENTER", "$inputs.anchor"],
        )
        max_sources: StrictInt = Field(
            default=MAX_TRACKED_SOURCES,
            ge=1,
            description=(
                "Most sources with a line and counts. A new source beyond this "
                "literal bound raises LineCounterCapacityError without evicting "
                "a source. Raise the bound or reset processing to free sources."
            ),
            examples=[256, 512],
        )

        @field_validator("line_segment")
        @classmethod
        def _line_is_valid(cls, value: Any) -> Any:
            # A selector is a string here; its value is checked in run().
            if not isinstance(value, str):
                zones.parse_line(value)

            return value

    def __init__(self) -> None:
        self._sources: Dict[str, _SourceLine] = {}

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        line_segment: Any,
        triggering_anchor: Optional[str] = None,
        max_sources: int = MAX_TRACKED_SOURCES,
    ) -> Dict[str, Any]:
        """Count this frame's crossings and return the source's totals.

        Args:
            image: Frame naming the source.
            predictions: Tracked rows of this frame; not modified.
            line_segment: Two ``[x, y]`` points.
            triggering_anchor: Name from ``zones.ANCHORS``, or ``None`` for
                all four box corners.
            max_sources: Positive literal bound on sources with state.

        Returns:
            ``count_in``, ``count_out``, ``detections_in`` and
            ``detections_out``.

        Raises:
            ValueError: When the line or anchor is invalid, a row has no
                ``tracker_id`` or ``max_sources`` is not a positive integer.
            LineCounterCapacityError: When a new source exceeds
                ``max_sources`` or the call has more than
                ``MAX_TRACKS_PER_SOURCE`` distinct tracker ids. The call
                changes no state; earlier batch rows may already have.
        """
        if type(max_sources) is not int or max_sources < 1:
            raise ValueError("max_sources must be a positive integer")

        line = zones.parse_line(line_segment)
        anchors = (
            zones.DEFAULT_LINE_ANCHORS
            if triggering_anchor is None
            else (zones.parse_anchor(triggering_anchor),)
        )
        boxes = boxes_of(predictions)
        tracker_ids = _tracker_ids(boxes, block_type=self.type)
        _check_track_count(tracker_ids, block_type=self.type)

        state = self._source_state(
            image, line=line, anchors=anchors, max_sources=max_sources
        )
        if not tracker_ids:
            return _result(state, crossed_in=predictions, crossed_out=predictions)

        # Copy, then cast: MPS returns zeros for a combined cpu + float64 .to().
        xyxy = boxes.xyxy.detach().to("cpu").to(torch.float64)
        sides = zones.line_sides(zones.anchor_points(xyxy, anchors), line).tolist()
        crossings = [
            _record(state, tracker_id=tracker_id, side=side)
            for tracker_id, side in zip(tracker_ids, sides)
        ]
        _forget_oldest(state.sides)

        crossed_in = select_rows(
            predictions,
            RowSelection.of(len(crossings), mask=[c == 1 for c in crossings]),
        )
        crossed_out = select_rows(
            predictions,
            RowSelection.of(len(crossings), mask=[c == -1 for c in crossings]),
        )
        result = _result(state, crossed_in=crossed_in, crossed_out=crossed_out)

        return result

    def reset_state(self) -> None:
        """Forget every source's line, anchors, counts and track sides."""
        self._sources.clear()

    def _source_state(
        self,
        image: ImageData,
        *,
        line: zones.Line,
        anchors: Tuple[str, ...],
        max_sources: int,
    ) -> _SourceLine:
        source = _source_key(self, image)
        state = self._sources.get(source)
        if state is None and len(self._sources) >= max_sources:
            raise LineCounterCapacityError(
                f"{self.type} cannot count source {source!r}: "
                f"max_sources={max_sources} is full. The rejected source changes "
                "no state; earlier batch rows may already have updated theirs. "
                "Raise max_sources or reset processing to free sources."
            )

        if state is None or (state.line, state.anchors) != (line, anchors):
            state = _SourceLine(line=line, anchors=anchors)
            self._sources[source] = state

        return state

def _record(state: _SourceLine, *, tracker_id: int, side: int) -> int:
    """Record one row's side; return +1 crossed in, -1 crossed out, else 0."""
    if side == 0:
        # Seen, side unknown: keep its last side and mark it recently seen.
        if tracker_id in state.sides:
            state.sides.move_to_end(tracker_id)

        return 0

    previous = state.sides.pop(tracker_id, None)
    state.sides[tracker_id] = side
    if previous is None or previous == side:
        return 0

    if side == 1:
        state.count_in += 1
    else:
        state.count_out += 1

    return side

def _forget_oldest(sides: "OrderedDict[int, int]") -> None:
    # After the whole call: this call's tracks are the most recently seen,
    # and _check_track_count keeps them within the bound.
    while len(sides) > MAX_TRACKS_PER_SOURCE:
        sides.popitem(last=False)

def _result(state: _SourceLine, *, crossed_in: Any, crossed_out: Any) -> dict:
    result = {
        "count_in": state.count_in,
        "count_out": state.count_out,
        "detections_in": crossed_in,
        "detections_out": crossed_out,
    }

    return result

def _check_track_count(tracker_ids: List[int], *, block_type: str) -> None:
    # Above the bound, recording this call would evict its own tracks.
    distinct = len(set(tracker_ids))
    if distinct > MAX_TRACKS_PER_SOURCE:
        raise LineCounterCapacityError(
            f"{block_type} keeps at most MAX_TRACKS_PER_SOURCE="
            f"{MAX_TRACKS_PER_SOURCE} tracks per source; this frame has "
            f"{distinct} distinct tracker ids. No state was changed."
        )

def _tracker_ids(boxes: Any, *, block_type: str) -> List[int]:
    rows = boxes.bboxes_metadata or []
    tracker_ids = [(row or {}).get(TRACKER_ID_KEY) for row in rows]
    tracker_ids += [None] * (len(boxes) - len(tracker_ids))
    if any(tracker_id is None for tracker_id in tracker_ids):
        raise ValueError(
            f"{block_type} needs tracked predictions: every row needs a "
            "tracker_id in bboxes_metadata; connect a tracker first"
        )

    integer_ids = [int(tracker_id) for tracker_id in tracker_ids]

    return integer_ids

def _source_key(block: Block, image: ImageData) -> str:
    metadata = image.video_metadata
    if metadata is not None and metadata.video_identifier:
        return metadata.video_identifier

    # The image's own context; a plain run() has no context.
    try:
        context = block.execution_context
    except NoExecutionContextError:
        return DEFAULT_SOURCE

    sample = context.sample_at(context.indices[0], field="image")
    key = sample.source_id if sample is not None else DEFAULT_SOURCE

    return key
