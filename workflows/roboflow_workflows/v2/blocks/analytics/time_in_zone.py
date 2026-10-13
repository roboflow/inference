"""``v2/time_in_zone@v3``: how long each tracked row has been in a zone.

Port of V1 ``roboflow_core/time_in_zone@v3``; its tensor variant is the
reference. The V1 ``detections`` input is ``predictions`` here. Per source
and tracker id, on every call::

    t        = frame_number / fps         video file with finite fps > 0
             = frame_timestamp (seconds)  otherwise (streams)
    in zone  = every triggering anchor inside ANY polygon of ``zone``
    in zone:      entry = entry or t;  time_in_zone = t - entry
    out of zone:  time_in_zone = 0;  entry forgotten if reset_out_of_zone_detections

So a row entering on frame 10 of a 10 fps file and staying reports 0.0,
0.1, 0.2, ... With ``reset_out_of_zone_detections=False`` a track keeps its
first entry: after leaving and coming back it reports the time since that
first entry, gap included.

``timed_detections`` holds the zone rows (all rows when
``remove_out_of_zone_detections`` is False), in input order, each with
``time_in_zone`` in seconds as a Python float in ``bboxes_metadata``.
``count_in_zone`` is the number of this frame's rows in a zone. Boxes,
masks (a segmentation view stays lazy) and row identity are unchanged.

``zone`` is one polygon, a list of polygons, or ``[]`` for none (no row is
in a zone). It and ``triggering_anchor`` may change per frame: entries are
kept per track, not per polygon or anchor, and each call tests the current
anchor. Polygon masks are cached, the ``ZONE_CACHE_SIZE`` most recently
added shared by all sources.

State: each source, created on its first call (empty prediction included),
remembers the entry time of at most ``MAX_TRACKS_PER_SOURCE`` tracks. A
track absent from a frame keeps its entry. After each call, the entries of
the tracks least recently seen are forgotten down to that bound; the call's
own tracks are always kept.

Capacity fails loudly, never by evicting a source or a track of the current
call. ``TimeInZoneCapacityError`` is raised, before this call changes any
entry, when:

* a new source would exceed the literal ``max_sources`` (default 256), or
* one call has more than ``MAX_TRACKS_PER_SOURCE`` distinct tracker ids.

A processing reset replaces the block and ``reset_state()`` forgets every
source. An empty prediction is returned as given with ``count_in_zone`` 0 and
changes no entry. Every row needs a
``tracker_id``; the image needs ``video_metadata``.

Source key: ``video_metadata.video_identifier``, else the ``image`` input's
own ``SampleContext.source_id``, else ``"default"``.

CPU boundary: the boxes are copied to the host once per non-empty call as
float64 and tested against CPU polygon masks (V1's raster). The kept rows are
then gathered on the boxes' device.

Differences from V1:

* ``reset_out_of_zone_detections=False`` keeps the entry also when
  ``remove_out_of_zone_detections`` is False; V1 then reset it anyway.
* A frame earlier than a track's entry (a restarted file) restarts the
  entry at that frame; V1 reported a negative time.
* A video file without a finite positive ``fps`` uses ``frame_timestamp``;
  V1 failed for ``fps=None``.
* A changed ``triggering_anchor`` applies from that call on, keeping
  entries; V1 kept testing the source's first anchor for that zone.
* Every polygon needs 3 points; V1 checked only the first. Bad literal zones
  and anchors fail at compile time.
* Adds ``count_in_zone``. Tracks and sources are bounded. A segmentation view
  is accepted.
"""

import math
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, List, Literal

import torch
from pydantic import Field, StrictBool, StrictInt, field_validator
from roboflow_workflows.v2.blocks.analytics import zones
from roboflow_workflows.v2.blocks.query.detections import (
    boxes_of,
    replace_boxes,
    select_rows,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
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
    "MAX_TRACKED_SOURCES",
    "MAX_TRACKS_PER_SOURCE",
    "TimeInZoneBlock",
    "TimeInZoneCapacityError",
    "ZONE_CACHE_SIZE",
]

DEFAULT_SOURCE = "default"
"""Source key of an image with neither a video identifier nor a source."""

# Default source ceiling; state is allocated only when a source arrives.
MAX_TRACKED_SOURCES = 256
MAX_TRACKS_PER_SOURCE = 4096
ZONE_CACHE_SIZE = 100

TRACKER_ID_KEY = "tracker_id"
TIME_IN_ZONE_KEY = "time_in_zone"
TIMED_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)
Anchor = Literal[zones.ANCHORS]


class TimeInZoneCapacityError(RuntimeError):
    """A new source or one call's tracks would exceed time in zone's bounds."""


class TimeInZoneBlock(Block):
    """Attach each tracked row's time in any zone, per source.

    Example, 10 fps video file, one zone, one track inside on frames 3-5 and
    7, outside on 6, ``reset_out_of_zone_detections=True``::

        frame           3     4     5     6     7
        in zone         yes   yes   yes   no    yes
        time_in_zone    0.0   0.1   0.2   -     0.0
    """

    type = "v2/time_in_zone@v3"
    metadata = {
        "name": "Time in Zone",
        "port_of": "roboflow_core/time_in_zone@v3",
        "section": "video",
    }
    outputs = {
        "timed_detections": Output(
            *TIMED_KINDS,
            source="predictions",
            description=(
                "Rows in a zone (all rows if remove_out_of_zone_detections is "
                "False) with time_in_zone in seconds in bboxes_metadata."
            ),
        ),
        "count_in_zone": Output(
            INTEGER_KIND,
            description="Rows of this frame in any zone.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata names the source and gives the "
                "time: frame_number / fps for video files, else frame_timestamp."
            )
        )
        predictions: Ref(*TIMED_KINDS) = Field(
            description=(
                "Tracked rows with tracker_id in bboxes_metadata, e.g. a "
                "tracker's tracked_detections (V1 input 'detections')."
            ),
            examples=["$steps.byte_tracker.tracked_detections"],
        )
        zone: list | Ref(LIST_OF_VALUES_KIND) = Field(
            description=(
                "One polygon [[x, y], ...] of at least 3 pixel points, a list "
                "of polygons (a row in any of them is in zone), or []."
            ),
            examples=[[[100, 100], [100, 200], [300, 200], [300, 100]], "$inputs.zone"],
        )
        triggering_anchor: Anchor | Ref(STRING_KIND) = Field(
            default="CENTER",
            description="Box point that must be inside a zone, e.g. BOTTOM_CENTER.",
            examples=["CENTER", "$inputs.anchor"],
        )
        remove_out_of_zone_detections: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "True: output only rows in a zone. False: output every row; "
                "rows outside have time_in_zone 0."
            ),
            examples=[True, False],
        )
        reset_out_of_zone_detections: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "True: leaving every zone forgets the entry, so time restarts "
                "at 0 on return. False: time counts from the first entry."
            ),
            examples=[True, False],
        )
        max_sources: StrictInt = Field(
            default=MAX_TRACKED_SOURCES,
            ge=1,
            description=(
                "Most sources with kept entries. A new source beyond this "
                "literal bound raises TimeInZoneCapacityError without evicting "
                "a source. Raise the bound or reset processing to free sources."
            ),
            examples=[256, 512],
        )

        @field_validator("zone")
        @classmethod
        def _zone_is_valid(cls, value: Any) -> Any:
            # A selector is a string here; its value is checked in run().
            if not isinstance(value, str):
                zones.parse_polygons(value)

            return value

    def __init__(self) -> None:
        # source -> tracker id -> entry time, least recently seen first.
        self._entries: Dict[str, "OrderedDict[int, float]"] = {}
        self._masks: "OrderedDict[zones.Polygon, torch.Tensor]" = OrderedDict()

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        zone: Any,
        triggering_anchor: str = "CENTER",
        remove_out_of_zone_detections: bool = True,
        reset_out_of_zone_detections: bool = True,
        max_sources: int = MAX_TRACKED_SOURCES,
    ) -> Dict[str, Any]:
        """Time this frame's tracked rows against the source's entries.

        Args:
            image: Frame naming the source and giving the time.
            predictions: Tracked rows of this frame; not modified.
            zone: One polygon, a list of polygons, or ``[]``.
            triggering_anchor: Name from ``zones.ANCHORS``.
            remove_out_of_zone_detections: Drop rows outside every zone.
            reset_out_of_zone_detections: Forget a track's entry when it is
                outside every zone.
            max_sources: Positive literal bound on sources with state.

        Returns:
            ``timed_detections`` and ``count_in_zone``.

        Raises:
            ValueError: When the zone, anchor or a flag is invalid, a row has
                no ``tracker_id``, the image has no ``video_metadata`` or
                ``max_sources`` is not a positive integer.
            TimeInZoneCapacityError: When a new source exceeds
                ``max_sources`` or the call has more than
                ``MAX_TRACKS_PER_SOURCE`` distinct tracker ids. The call
                changes no entry; earlier batch rows may already have.
        """
        if type(max_sources) is not int or max_sources < 1:
            raise ValueError("max_sources must be a positive integer")

        polygons = zones.parse_polygons(zone)
        anchor = zones.parse_anchor(triggering_anchor)
        _check_flag(remove_out_of_zone_detections, "remove_out_of_zone_detections")
        _check_flag(reset_out_of_zone_detections, "reset_out_of_zone_detections")
        now = _time_of(image)
        boxes = boxes_of(predictions)
        tracker_ids = _tracker_ids(boxes, block_type=self.type)
        _check_track_count(tracker_ids, block_type=self.type)

        entries = self._source_entries(image, max_sources=max_sources)
        if not tracker_ids:
            return {"timed_detections": predictions, "count_in_zone": 0}

        in_zone = self._in_any_zone(boxes.xyxy, polygons=polygons, anchor=anchor)
        times = []
        for tracker_id, inside in zip(tracker_ids, in_zone):
            if inside:
                elapsed = _record_inside_and_get_elapsed(
                    entries, tracker_id=tracker_id, now=now
                )
                times.append(elapsed)
            else:
                _mark_outside(
                    entries, tracker_id=tracker_id, reset=reset_out_of_zone_detections
                )
                times.append(0.0)
        _forget_oldest(entries)

        kept = in_zone if remove_out_of_zone_detections else [True] * len(in_zone)
        kept_rows = select_rows(predictions, RowSelection.of(len(kept), mask=kept))
        kept_boxes = boxes_of(kept_rows)
        rows_metadata = [dict(row or {}) for row in kept_boxes.bboxes_metadata or []]
        rows_metadata += [{} for _ in range(len(kept_boxes) - len(rows_metadata))]
        kept_times = [time for time, keep in zip(times, kept) if keep]
        for row, time in zip(rows_metadata, kept_times):
            row[TIME_IN_ZONE_KEY] = time
        timed = replace_boxes(
            kept_rows, replace(kept_boxes, bboxes_metadata=rows_metadata)
        )
        result = {"timed_detections": timed, "count_in_zone": sum(in_zone)}

        return result

    def reset_state(self) -> None:
        """Forget every source's entries; cached polygon masks are kept."""
        self._entries.clear()

    def _source_entries(
        self, image: ImageData, *, max_sources: int
    ) -> "OrderedDict[int, float]":
        source = _source_key(self, image)
        entries = self._entries.get(source)
        if entries is None:
            if len(self._entries) >= max_sources:
                raise TimeInZoneCapacityError(
                    f"{self.type} cannot time source {source!r}: "
                    f"max_sources={max_sources} is full. The rejected source "
                    "changes no entry; earlier batch rows may already have "
                    "updated theirs. Raise max_sources or reset processing to "
                    "free sources."
                )

            entries = OrderedDict()
            self._entries[source] = entries

        return entries

    def _in_any_zone(
        self,
        xyxy: torch.Tensor,
        *,
        polygons: tuple,
        anchor: str,
    ) -> List[bool]:
        # Copy, then cast: MPS returns zeros for a combined cpu + float64 .to().
        host_xyxy = xyxy.detach().to("cpu").to(torch.float64)
        points = zones.anchor_points(host_xyxy, (anchor,))
        in_zone = torch.zeros(len(host_xyxy), dtype=torch.bool)
        for polygon in polygons:
            in_zone |= zones.points_in_mask(points, self._mask_of(polygon))

        rows_in_zone = in_zone.tolist()

        return rows_in_zone

    def _mask_of(self, polygon: zones.Polygon) -> torch.Tensor:
        mask = self._masks.get(polygon)
        if mask is None:
            mask = zones.polygon_mask(polygon)
            self._masks[polygon] = mask
            if len(self._masks) > ZONE_CACHE_SIZE:
                self._masks.popitem(last=False)

        return mask


def _record_inside_and_get_elapsed(
    entries: "OrderedDict[int, float]", *, tracker_id: int, now: float
) -> float:
    entry = entries.pop(tracker_id, now)
    if now < entry:
        entry = now
    entries[tracker_id] = entry

    elapsed = now - entry

    return elapsed


def _mark_outside(
    entries: "OrderedDict[int, float]", *, tracker_id: int, reset: bool
) -> None:
    if reset:
        entries.pop(tracker_id, None)
    elif tracker_id in entries:
        # Kept entry of a track seen this call: mark it recently seen.
        entries.move_to_end(tracker_id)


def _forget_oldest(entries: "OrderedDict[int, float]") -> None:
    # After the whole call: this call's tracks are the most recently seen,
    # and _check_track_count keeps them within the bound.
    while len(entries) > MAX_TRACKS_PER_SOURCE:
        entries.popitem(last=False)


def _check_flag(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false; got {value!r}")


def _check_track_count(tracker_ids: List[int], *, block_type: str) -> None:
    # Above the bound, recording this call would evict its own tracks.
    distinct = len(set(tracker_ids))
    if distinct > MAX_TRACKS_PER_SOURCE:
        raise TimeInZoneCapacityError(
            f"{block_type} keeps at most MAX_TRACKS_PER_SOURCE="
            f"{MAX_TRACKS_PER_SOURCE} tracks per source; this frame has "
            f"{distinct} distinct tracker ids. No entry was changed."
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


def _time_of(image: ImageData) -> float:
    metadata = image.video_metadata
    if metadata is None:
        raise ValueError(
            f"v2/time_in_zone@v3 needs video_metadata for the frame time; image "
            f"{image.image_id!r} has none"
        )

    fps = metadata.fps
    has_rate = isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0
    if metadata.comes_from_video_file and has_rate:
        return metadata.frame_number / fps

    timestamp = metadata.frame_timestamp.timestamp()

    return timestamp


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
