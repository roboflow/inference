"""``v2/detection_event_log@v1``: when each track appeared and how long it stayed.

Port of V1 ``roboflow_core/detection_event_log@v1``; its tensor variant is
the reference. The V1 ``detections`` input is ``predictions`` here. Per
source, every call is one frame of an internal counter (1, 2, ...), and each
tracker id has one event::

    event = tracker_id, class_name (of its first row),
            first/last_seen_frame        internal frame counter
            first/last_seen_timestamp    frame_timestamp, epoch seconds
            first/last_seen_relative     seconds since the source's first frame
            frame_count                  frames the track was seen in

    frame_count >= frame_threshold  -> event_log["logged"]
    otherwise                       -> event_log["pending"]

Every ``flush_interval`` frames, before this frame's rows are added, events
not seen for more than ``stale_frames`` frames are removed; the removed
logged ones are returned once in ``complete_events``. Both mappings are keyed
by the tracker id as a string, in first-seen order. ``total_logged`` and
``total_pending`` count the current ``event_log``.

Times: with ``video_metadata``, absolute times are its ``frame_timestamp``
and relative times its distance from the source's first timestamp. An image
without ``video_metadata`` uses ``time.time()`` and ``(frame - 1) /
fallback_fps``.

Rows: a row without ``tracker_id`` is ignored. A row's class is its own
``class`` name, else ``image_metadata["class_names"]`` of its class id, else
``class_<id>``. ``detections`` returns the input prediction unchanged. An
empty prediction is a frame without rows: it ages events. A filtered or
missing prediction skips the call and is not a frame.

Source key: ``video_metadata.video_identifier``, else the ``image`` input's
own ``SampleContext.source_id``, else ``"default"`` (also for a plain
``run()`` outside the engine). ``max_sources`` bounds the sources; the first
frame of one source too many raises ``TrackerCapacityError`` before any
state changes. Sources are never evicted; a processing reset replaces the
block and drops them all.
State is block-instance state, one instance per step and session. It relies
on the engine calling one step one call at a time, in run order (a pipelined
run serializes each step's calls in pulse order).

CPU boundary: one device-to-host copy of the class ids per non-empty call;
the log is host data.

Differences from V1:

* ``max_sources`` replaces V1's silent eviction of the least recently seen
  of 100 videos.
* A tracker id with several rows in one frame counts that frame once; V1
  counted each row.
* The deprecated, unused ``reference_timestamp`` input is not offered.
* V1 required ``video_metadata``, so ``fallback_fps`` was never reached.
* A segmentation view is accepted and stays lazy.
"""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.query.detections import boxes_of
from roboflow_workflows.v2.blocks.tracking.byte_track import TrackerCapacityError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import DICTIONARY_KIND, FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.masks import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.prediction_ops import class_names_by_id, row_class_names
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

__all__ = [
    "DEFAULT_SOURCE",
    "DetectionEventLogBlock",
]

DEFAULT_SOURCE = "default"
"""Source key of an image with neither a video identifier nor a source."""

DEFAULT_MAX_SOURCES = 256
TRACKER_ID_KEY = "tracker_id"

TRACKED_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)


@dataclass
class _Event:
    """One track's event; ``asdict`` gives the V1 output fields in V1 order."""

    tracker_id: int
    class_name: str
    first_seen_frame: int
    first_seen_timestamp: float
    last_seen_frame: int
    last_seen_timestamp: float
    first_seen_relative: float
    last_seen_relative: float
    frame_count: int = 1


@dataclass
class _SourceLog:
    events: Dict[int, _Event] = field(default_factory=dict)
    frame: int = 0
    last_flush: Optional[int] = None
    first_timestamp: Optional[float] = None


class DetectionEventLogBlock(Block):
    """Log first and last sightings of each track, per source.

    Example, ``frame_threshold=2``, ``flush_interval=1``, ``stale_frames=1``,
    track 7 seen in frames 1 and 2 only::

        frame   pending   logged   complete_events
        1       {"7"}     {}       {}
        2       {}        {"7"}    {}
        3       {}        {"7"}    {}
        4       {}        {}       {"7": frames 1-2, frame_count 2}
    """

    type = "v2/detection_event_log@v1"
    metadata = {
        "name": "Detection Event Log",
        "port_of": "roboflow_core/detection_event_log@v1",
        "section": "analytics",
    }
    outputs = {
        "event_log": Output(
            DICTIONARY_KIND,
            description=(
                "{'logged': {...}, 'pending': {...}}: current events by tracker "
                "id string, split by frame_threshold."
            ),
        ),
        "detections": Output(
            *TRACKED_KINDS,
            source="predictions",
            description="The input prediction, unchanged.",
        ),
        "total_logged": Output(
            INTEGER_KIND, description="Number of logged events in event_log."
        ),
        "total_pending": Output(
            INTEGER_KIND, description="Number of pending events in event_log."
        ),
        "complete_events": Output(
            DICTIONARY_KIND,
            description=(
                "Logged events removed as stale in this call, by tracker id "
                "string; empty in most calls."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata names the source and gives the "
                "frame_timestamp."
            ),
            examples=["$inputs.image"],
        )
        predictions: Ref(*TRACKED_KINDS) = Field(
            description=(
                "Tracked rows with tracker_id in bboxes_metadata, e.g. a "
                "tracker's tracked_detections (V1 input 'detections')."
            ),
            examples=["$steps.byte_tracker.tracked_detections"],
        )
        frame_threshold: StrictInt | Ref(INTEGER_KIND) = Field(
            default=30,
            ge=1,
            description="Frames a track must be seen in before it is logged.",
            examples=[5, "$inputs.frame_threshold"],
        )
        flush_interval: StrictInt | Ref(INTEGER_KIND) = Field(
            default=30,
            ge=1,
            description="Frames between removals of stale events.",
            examples=[30, 60],
        )
        stale_frames: StrictInt | Ref(INTEGER_KIND) = Field(
            default=300,
            ge=1,
            description=(
                "An event not seen for more than this many frames is removed "
                "at the next removal."
            ),
            examples=[150, 300],
        )
        fallback_fps: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=1.0,
            gt=0,
            description=(
                "Frame rate for relative times of images without " "video_metadata."
            ),
            examples=[1.0, 30.0],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with an event log. A frame of one more source "
                "fails with TrackerCapacityError; no source is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._sources: Dict[str, _SourceLog] = {}

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        frame_threshold: int = 30,
        flush_interval: int = 30,
        stale_frames: int = 300,
        fallback_fps: float = 1.0,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> Dict[str, Any]:
        """Record this frame's tracks and return the source's log.

        Args:
            image: Frame naming the source and giving the time.
            predictions: Tracked rows of this frame; not modified.
            frame_threshold: Frames before an event is logged, at least 1.
            flush_interval: Frames between stale removals, at least 1.
            stale_frames: Unseen frames before removal, at least 1.
            fallback_fps: Positive rate for images without video metadata.
            max_sources: Most sources with state.

        Returns:
            ``event_log``, ``detections``, ``total_logged``,
            ``total_pending`` and ``complete_events``.

        Raises:
            ValueError: When a direct call passes an out-of-range parameter.
            TrackerCapacityError: When this is the first frame of a source
                and ``max_sources`` sources already have state.
        """
        _check_parameters(
            frame_threshold=frame_threshold,
            flush_interval=flush_interval,
            stale_frames=stale_frames,
            fallback_fps=fallback_fps,
        )
        rows = _tracked_rows(predictions)
        log = self._source_log(image, max_sources=max_sources)
        log.frame += 1
        now, relative = _times(log, image, fallback_fps=fallback_fps)

        if log.last_flush is None:
            log.last_flush = log.frame
        complete: List[_Event] = []
        if log.frame - log.last_flush >= flush_interval:
            complete = _remove_stale(log, stale_frames, frame_threshold)
            log.last_flush = log.frame

        for tracker_id, class_name in rows:
            event = log.events.get(tracker_id)
            if event is None:
                log.events[tracker_id] = _Event(
                    tracker_id=tracker_id,
                    class_name=class_name,
                    first_seen_frame=log.frame,
                    first_seen_timestamp=now,
                    last_seen_frame=log.frame,
                    last_seen_timestamp=now,
                    first_seen_relative=relative,
                    last_seen_relative=relative,
                )
                continue
            event.last_seen_frame = log.frame
            event.last_seen_timestamp = now
            event.last_seen_relative = relative
            event.frame_count += 1

        logged = {
            str(event.tracker_id): asdict(event)
            for event in log.events.values()
            if event.frame_count >= frame_threshold
        }
        pending = {
            str(event.tracker_id): asdict(event)
            for event in log.events.values()
            if event.frame_count < frame_threshold
        }
        result = {
            "event_log": {"logged": logged, "pending": pending},
            "detections": predictions,
            "total_logged": len(logged),
            "total_pending": len(pending),
            "complete_events": {
                str(event.tracker_id): asdict(event) for event in complete
            },
        }

        return result

    def _source_log(self, image: ImageData, *, max_sources: int) -> _SourceLog:
        source = _source_key(self, image)
        log = self._sources.get(source)
        if log is not None:
            return log

        if len(self._sources) >= max_sources:
            raise TrackerCapacityError(
                f"{self.type}: source {source!r} would be source "
                f"{len(self._sources) + 1} of max_sources={max_sources}; "
                "raise max_sources or reset processing to drop ended sources"
            )

        log = _SourceLog()
        self._sources[source] = log

        return log


def _remove_stale(
    log: _SourceLog, stale_frames: int, frame_threshold: int
) -> List[_Event]:
    """Drop events unseen for more than ``stale_frames``; return logged ones."""
    stale = [
        event
        for event in log.events.values()
        if log.frame - event.last_seen_frame > stale_frames
    ]
    for event in stale:
        del log.events[event.tracker_id]
    complete = [event for event in stale if event.frame_count >= frame_threshold]

    return complete


def _tracked_rows(predictions: Any) -> List[Tuple[int, str]]:
    """Return ``(tracker_id, class_name)`` of rows with a tracker id, once each."""
    boxes = boxes_of(predictions)
    rows_metadata = list(boxes.bboxes_metadata or [])
    rows_metadata += [{} for _ in range(len(boxes) - len(rows_metadata))]
    rows_metadata = [row or {} for row in rows_metadata]
    if not any(row.get(TRACKER_ID_KEY) is not None for row in rows_metadata):
        return []

    class_ids = [int(class_id) for class_id in boxes.class_id.tolist()]
    names = row_class_names(
        class_ids,
        row_metadata=rows_metadata,
        names_by_id=class_names_by_id(boxes.image_metadata),
    )
    tracked: Dict[int, str] = {}
    for row, class_id, name in zip(rows_metadata, class_ids, names):
        tracker_id = row.get(TRACKER_ID_KEY)
        if tracker_id is None or int(tracker_id) in tracked:
            continue
        tracked[int(tracker_id)] = (
            str(name) if name is not None else f"class_{class_id}"
        )

    return list(tracked.items())


def _times(
    log: _SourceLog, image: ImageData, *, fallback_fps: float
) -> Tuple[float, float]:
    """Return this frame's absolute and relative time in seconds."""
    metadata = image.video_metadata
    if metadata is None or metadata.frame_timestamp is None:
        return time.time(), (log.frame - 1) / fallback_fps

    timestamp = metadata.frame_timestamp.timestamp()
    if log.first_timestamp is None:
        log.first_timestamp = timestamp

    return timestamp, timestamp - log.first_timestamp


def _check_parameters(
    *, frame_threshold: Any, flush_interval: Any, stale_frames: Any, fallback_fps: Any
) -> None:
    # Params checks literals and selected values; direct calls get the same.
    counts = {
        "frame_threshold": frame_threshold,
        "flush_interval": flush_interval,
        "stale_frames": stale_frames,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    is_rate = isinstance(fallback_fps, (int, float)) and not isinstance(
        fallback_fps, bool
    )
    if not (is_rate and math.isfinite(fallback_fps) and fallback_fps > 0):
        raise ValueError(
            f"fallback_fps must be a finite number > 0, got {fallback_fps!r}"
        )


def _source_key(block: Block, image: ImageData) -> str:
    metadata = image.video_metadata
    if metadata is not None and metadata.video_identifier:
        return metadata.video_identifier

    # The image's own context: another input without one must not drop it.
    # A plain run() has no context; errors in an engine call still propagate.
    try:
        context = block.execution_context
    except NoExecutionContextError:
        return DEFAULT_SOURCE

    sample = context.sample_at(context.indices[0], field="image")
    key = sample.source_id if sample is not None else DEFAULT_SOURCE

    return key
