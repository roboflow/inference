"""``v2/velocity@v1``: velocity and speed of tracked box centres.

Port of V1 ``roboflow_core/velocity@v1``; its tensor variant is the
reference. The V1 ``detections`` input is ``predictions`` here. Per source
and tracker id::

    t         = frame_number / fps            video file with fps > 0
              = frame_timestamp (seconds)     otherwise (streams)
    velocity  = (centre - previous centre) / (t - previous t)     px/s
              = (0, 0)  on a track's first frame, or when t did not grow
    smoothed  = a * velocity + (1 - a) * previous smoothed   a = smoothing_alpha
              = velocity on a track's first frame
    speed     = |velocity|, smoothed_speed = |smoothed|
    outputs   = each of the four / pixels_per_meter            m/s

So a centre moving ``d`` px per frame of a 30 fps file reports
``30 * d / pixels_per_meter`` from its second frame, and its smoothed value
approaches that as ``1 - (1 - a) ** (k - 1)`` on frame ``k``.

Each row gets ``velocity`` ``[vx, vy]``, ``speed``, ``smoothed_velocity`` and
``smoothed_speed`` as Python floats in ``bboxes_metadata``. Rows, boxes,
masks (a segmentation view stays low resolution), keypoints and row identity
are unchanged. State holds unconverted px/s, so a ``pixels_per_meter`` change
rescales outputs at once; ``smoothing_alpha`` applies from the call that
passes it. A track missing from frames keeps its last centre and time and is
measured across the gap when it returns.

An empty prediction is returned as given and changes no track state (its
source is still registered); a filtered or missing prediction skips the
call. Every row needs a ``tracker_id``.

Source key: ``video_metadata.video_identifier``, else the ``image`` input's
own ``SampleContext.source_id``, else ``"default"`` (also for a plain
``run()`` outside the engine). ``max_sources`` (default 256) bounds the
sources; the first frame of one source too many raises
``TrackerCapacityError`` before any state changes. Sources are never
evicted; a processing reset replaces the block and drops them all. Each
source remembers its ``MAX_TRACKS_PER_SOURCE`` most recently seen tracks;
within one frame a later row is more recent. A frame with more rows than
that still measures every row, then remembers only its last
``MAX_TRACKS_PER_SOURCE``.

Device and CPU boundary: centres, displacement, smoothing and unit
conversion run on the boxes' device in float32, and positions and smoothed
velocities stay there between frames. Tracker ids and absolute times stay on
the host in float64 (epoch seconds do not fit float32). Per non-empty call:
three small host-to-device copies (previous rows, known flags, time deltas)
and one device-to-host copy of the ``(N, 6)`` result.

Differences from V1:

* A video file without a positive finite ``fps`` uses ``frame_timestamp``;
  V1 failed for ``fps=None``.
* Out-of-range literal ``smoothing_alpha`` and ``pixels_per_meter`` fail at
  compile time, selected ones when resolved; V1 checked both in ``run``.
* V1 kept every track ever seen; V2 keeps the 4096 most recently seen per
  source, as V1's track class lock does, and has no 256-video eviction.
* A segmentation view is accepted and stays lazy.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import torch
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.query.detections import replace_boxes
from roboflow_workflows.v2.blocks.tracking.base import (
    DEFAULT_MAX_SOURCES,
    TRACKED_KINDS,
    TrackedPrediction,
    TrackerCapacityError,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

__all__ = [
    "DEFAULT_SOURCE",
    "MAX_TRACKS_PER_SOURCE",
    "VelocityBlock",
]

DEFAULT_SOURCE = "default"
"""Source key of an image with neither a video identifier nor a source."""

MAX_TRACKS_PER_SOURCE = 4096

# Row keys; values match the V1 execution engine constants.
TRACKER_ID_KEY = "tracker_id"
VELOCITY_KEY = "velocity"
SPEED_KEY = "speed"
SMOOTHED_VELOCITY_KEY = "smoothed_velocity"
SMOOTHED_SPEED_KEY = "smoothed_speed"


@dataclass
class _SourceTracks:
    """Velocity state of one source.

    ``rows`` maps a tracker id to its row in ``positions`` and ``smoothed``,
    ordered from least to most recently seen.
    """

    rows: Dict[int, int] = field(default_factory=dict)
    times: Dict[int, float] = field(default_factory=dict)
    positions: Optional[torch.Tensor] = None  # (K, 2) px
    smoothed: Optional[torch.Tensor] = None  # (K, 2) px/s


class VelocityBlock(Block):
    """Attach velocity and speed of each tracked box centre, per source.

    Example, 10 fps video file, ``pixels_per_meter=10``, ``smoothing_alpha=0.5``,
    a centre moving 3 px right per frame::

        frame               1      2      3      4
        velocity [m/s]      0      3.0    3.0    3.0
        smoothed [m/s]      0      1.5    2.25   2.625
    """

    type = "v2/velocity@v1"
    metadata = {
        "name": "Velocity",
        "port_of": "roboflow_core/velocity@v1",
        "section": "video",
    }
    outputs = {
        "velocity_detections": Output(
            *TRACKED_KINDS,
            source="predictions",
            description=(
                "The input rows with velocity, speed, smoothed_velocity and "
                "smoothed_speed in bboxes_metadata, in px/s divided by "
                "pixels_per_meter (m/s when calibrated; px/s at the default 1.0)."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata names the source and gives the "
                "time: frame_number / fps for video files, else frame_timestamp."
            )
        )
        predictions: Ref(*TRACKED_KINDS) = Field(
            description=(
                "Tracked rows with tracker_id in bboxes_metadata, e.g. a "
                "tracker's tracked_detections (V1 input 'detections')."
            ),
            examples=["$steps.byte_tracker.tracked_detections"],
        )
        smoothing_alpha: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=0.5,
            gt=0,
            le=1,
            description=(
                "Weight of the newest velocity in the moving average; lower "
                "is smoother and slower."
            ),
            examples=[0.5, "$inputs.smoothing_alpha"],
        )
        pixels_per_meter: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=1.0,
            gt=0,
            description=(
                "Pixels per meter; outputs are px/s divided by it. 1.0 reports "
                "px/s; 100.0 means 1 px is 1 cm."
            ),
            examples=[1.0, 100.0, "$inputs.pixels_per_meter"],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with velocity state. A frame of one more source "
                "fails with TrackerCapacityError; no source is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._sources: Dict[str, _SourceTracks] = {}

    def run(
        self,
        *,
        image: ImageData,
        predictions: TrackedPrediction,
        smoothing_alpha: float,
        pixels_per_meter: float,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> Dict[str, Any]:
        """Measure this frame's tracked centres against the source's last ones.

        Args:
            image: Frame naming the source and giving the time.
            predictions: Tracked rows of this frame; not modified.
            smoothing_alpha: Moving-average weight, in (0, 1].
            pixels_per_meter: Positive px-to-m divisor.
            max_sources: Most sources with state.

        Returns:
            ``velocity_detections``: the input family with four values per row.

        Raises:
            ValueError: When a row has no ``tracker_id``, a keypoint
                prediction has no boxes, the image has no ``video_metadata``
                or a direct call passes an out-of-range parameter.
            TrackerCapacityError: When this is the first frame of a source
                and ``max_sources`` sources already have state.
        """
        if not 0 < smoothing_alpha <= 1:
            raise ValueError(
                f"smoothing_alpha must be in (0, 1], got {smoothing_alpha}"
            )
        if not pixels_per_meter > 0:
            raise ValueError(f"pixels_per_meter must be > 0, got {pixels_per_meter}")

        boxes = _boxes_of(predictions)
        rows_metadata = [dict(row or {}) for row in boxes.bboxes_metadata or []]
        rows_metadata += [{} for _ in range(len(boxes) - len(rows_metadata))]
        tracker_ids = [row.get(TRACKER_ID_KEY) for row in rows_metadata]
        if any(tracker_id is None for tracker_id in tracker_ids):
            raise ValueError(
                f"{self.type} needs tracked predictions: every row needs a "
                "tracker_id in bboxes_metadata; connect a tracker first"
            )

        now = _time_of(image)
        source = self._source_state(image, max_sources=max_sources)
        if not tracker_ids:
            return {"velocity_detections": predictions}

        measured = _measure(
            source,
            xyxy=boxes.xyxy.detach(),
            tracker_ids=[int(tracker_id) for tracker_id in tracker_ids],
            now=now,
            alpha=float(smoothing_alpha),
        )
        for row, values in zip(rows_metadata, (measured / pixels_per_meter).tolist()):
            row[VELOCITY_KEY] = values[0:2]
            row[SPEED_KEY] = values[2]
            row[SMOOTHED_VELOCITY_KEY] = values[3:5]
            row[SMOOTHED_SPEED_KEY] = values[5]

        if isinstance(predictions, SegmentationView):
            measured_rows = predictions.with_row_metadata(rows_metadata)
        else:
            measured_boxes = replace(boxes, bboxes_metadata=rows_metadata)
            measured_rows = replace_boxes(predictions, measured_boxes)
        result = {"velocity_detections": measured_rows}

        return result

    def _source_state(self, image: ImageData, *, max_sources: int) -> _SourceTracks:
        source = _source_key(self, image)
        state = self._sources.get(source)
        if state is not None:
            return state

        if len(self._sources) >= max_sources:
            raise TrackerCapacityError(
                f"{self.type}: source {source!r} would be source "
                f"{len(self._sources) + 1} of max_sources={max_sources}; "
                "raise max_sources or reset processing to drop ended sources"
            )

        state = _SourceTracks()
        self._sources[source] = state

        return state


def _measure(
    source: _SourceTracks,
    *,
    xyxy: torch.Tensor,
    tracker_ids: List[int],
    now: float,
    alpha: float,
) -> torch.Tensor:
    """Update the source with this frame and return ``(N, 6)`` px/s values.

    Columns: vx, vy, speed, smoothed vx, smoothed vy, smoothed speed; on the
    host, in float32.
    """
    device = xyxy.device
    centres = (xyxy[:, :2].float() + xyxy[:, 2:].float()) * 0.5
    if source.positions is not None and source.positions.device != device:
        source.positions = source.positions.to(device)
        source.smoothed = source.smoothed.to(device)

    known_flags = [tracker_id in source.rows for tracker_id in tracker_ids]
    deltas = [
        now - source.times[tracker_id] if known else 0.0
        for tracker_id, known in zip(tracker_ids, known_flags)
    ]
    if any(known_flags):
        previous_rows = [source.rows.get(tracker_id, 0) for tracker_id in tracker_ids]
        index = torch.as_tensor(previous_rows, dtype=torch.long, device=device)
        previous_centres = source.positions[index]
        previous_smoothed = source.smoothed[index]
    else:
        previous_centres = centres
        previous_smoothed = torch.zeros_like(centres)

    known = torch.as_tensor(known_flags, dtype=torch.bool, device=device)
    elapsed = torch.as_tensor(deltas, dtype=torch.float32, device=device)
    moved = (known & (elapsed > 0)).unsqueeze(1)
    safe_elapsed = torch.where(elapsed > 0, elapsed, torch.ones_like(elapsed))
    velocity = torch.where(
        moved,
        (centres - previous_centres) / safe_elapsed.unsqueeze(1),
        torch.zeros_like(centres),
    )
    smoothed = torch.where(
        known.unsqueeze(1),
        alpha * velocity + (1 - alpha) * previous_smoothed,
        velocity,
    )
    values = torch.cat(
        [
            velocity,
            torch.linalg.vector_norm(velocity, dim=1, keepdim=True),
            smoothed,
            torch.linalg.vector_norm(smoothed, dim=1, keepdim=True),
        ],
        dim=1,
    )

    _remember(
        source, tracker_ids=tracker_ids, centres=centres, smoothed=smoothed, now=now
    )
    host_values = values.cpu()

    return host_values


def _remember(
    source: _SourceTracks,
    *,
    tracker_ids: List[int],
    centres: torch.Tensor,
    smoothed: torch.Tensor,
    now: float,
) -> None:
    # Recency: this frame's rows, last row newest, then absent tracks, least
    # recently seen first. The oldest beyond the cap are dropped, so a frame
    # with more rows than the cap keeps only its last ones; a measured id
    # outside them is dropped too, not left with its previous state.
    first = max(len(tracker_ids) - MAX_TRACKS_PER_SOURCE, 0)
    kept_ids = tracker_ids[first:]
    if first:
        # Copies, so the dropped rows' storage is not retained.
        centres, smoothed = centres[first:].clone(), smoothed[first:].clone()

    measured = set(tracker_ids)
    survivors = [
        (tracker_id, row)
        for tracker_id, row in source.rows.items()
        if tracker_id not in measured
    ]
    room = MAX_TRACKS_PER_SOURCE - len(kept_ids)
    survivors = survivors[max(len(survivors) - room, 0) :]
    if survivors:
        index = torch.as_tensor(
            [row for _, row in survivors], dtype=torch.long, device=centres.device
        )
        source.positions = torch.cat([source.positions[index], centres])
        source.smoothed = torch.cat([source.smoothed[index], smoothed])
    else:
        source.positions, source.smoothed = centres, smoothed

    rows = {tracker_id: row for row, (tracker_id, _) in enumerate(survivors)}
    times = {tracker_id: source.times[tracker_id] for tracker_id, _ in survivors}
    for offset, tracker_id in enumerate(kept_ids):
        rows[tracker_id] = len(survivors) + offset
        times[tracker_id] = now
    source.rows, source.times = rows, times


def _time_of(image: ImageData) -> float:
    metadata = image.video_metadata
    if metadata is None:
        raise ValueError(
            f"v2/velocity@v1 needs video_metadata for the frame time; image "
            f"{image.image_id!r} has none"
        )

    fps = metadata.fps
    has_rate = isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0
    if metadata.comes_from_video_file and has_rate:
        return metadata.frame_number / fps

    timestamp = metadata.frame_timestamp.timestamp()

    return timestamp


def _boxes_of(prediction: TrackedPrediction) -> Detections | InstanceDetections:
    if isinstance(prediction, SegmentationView):
        return prediction.detections

    if not isinstance(prediction, tuple):
        return prediction

    if prediction[1] is None:
        raise ValueError(
            "Velocity needs boxes: this keypoint prediction has KeyPoints "
            "without Detections"
        )

    return prediction[1]


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
