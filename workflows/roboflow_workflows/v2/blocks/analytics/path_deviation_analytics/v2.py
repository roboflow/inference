"""``roboflow_core/path_deviation_analytics@v2``: Fréchet distance of tracks from a path.

Port of V1 ``roboflow_core/path_deviation_analytics@v2``; its tensor variant
is the reference. The V1 ``detections`` input is ``predictions`` here. Per
source and tracker id, each frame appends the row's anchor point to the
track's path and reports the discrete Fréchet distance between that path
(``P``, ``T`` points so far) and ``reference_path`` (``R``, ``M`` points)::

    d[i][j] = |P[i] - R[j]|                         Euclidean, pixels
    F[0][0] = d[0][0]
    F[i][j] = max(d[i][j], min(F[i-1][j], F[i-1][j-1], F[i][j-1]))
    path_deviation = F[T-1][M-1]

So a track moving exactly along the reference reports 0 once it reached
the path's end, and a track shifted by ``k`` px everywhere reports ``k``.

The recurrence needs only the previous row ``F[T-2]``, so each track keeps
its last row: one new frame costs ``O(M)``, not V1's ``O(T * M)``
recursive recomputation, which also exceeds Python's recursion limit for
tracks of about a thousand frames. The value is the same. A changed
``reference_path`` recomputes a track's row from its kept path once.

Each row gets ``path_deviation`` (a Python float) in ``bboxes_metadata``.
Rows, boxes, masks (a segmentation view stays low resolution) and row
identity are unchanged. Every row needs a ``tracker_id``. A track missing
from frames adds no point while it is missing. An empty prediction is
returned as given and changes no track state (its source is still
registered); a filtered or missing prediction skips the call.

Source key: ``video_metadata.video_identifier``, else the ``image`` input's
own ``SampleContext.source_id``, else ``"default"`` (also for a plain
``run()`` outside the engine). ``max_sources`` bounds the sources; the
first frame of one source too many raises ``TrackerCapacityError`` before
any state changes. Sources are never evicted; a processing reset replaces
the block and drops them all. Each source remembers its
``MAX_TRACKS_PER_SOURCE`` most recently seen tracks.
State is block-instance state, one instance per step and session. It relies
on the engine calling one step one call at a time, in run order (a pipelined
run serializes each step's calls in pulse order).

CPU boundary: the recurrence is a sequential scan over small host arrays.
Per non-empty call, the ``(N, 4)`` boxes are copied to the host once and
anchors and distances are computed there in float64, as V1 does.

Differences from V1:

* ``CENTER_OF_MASS`` is not offered: the V1 tensor block failed on it, as
  it never passed masks to the anchor helper.
* Literal anchors and reference paths fail at compile time and selected
  ones in the call; V1 failed inside the computation, or not at all for a
  one-point path.
* Long tracks no longer fail with ``RecursionError``.
* V1 kept every track and video ever seen; V2 bounds both (see above).
* A segmentation view is accepted and stays lazy.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
from pydantic import Field, StrictInt
from roboflow_workflows.v2.blocks.query.detections import boxes_of, replace_boxes
from roboflow_workflows.v2.blocks.tracking.byte_tracker.v3 import TrackerCapacityError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND, STRING_KIND
from roboflow_workflows.v2.data.masks import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


__all__ = [
    "ANCHORS",
    "DEFAULT_SOURCE",
    "MAX_TRACKS_PER_SOURCE",
    "PathDeviationBlock",
]

DEFAULT_SOURCE = "default"

DEFAULT_MAX_SOURCES = 256

MAX_TRACKS_PER_SOURCE = 4096

TRACKER_ID_KEY = "tracker_id"

PATH_DEVIATION_KEY = "path_deviation"

TRACKED_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)

ANCHORS: Dict[str, Tuple[str, str]] = {
    "CENTER": ("centre", "centre"),
    "CENTER_LEFT": ("min", "centre"),
    "CENTER_RIGHT": ("max", "centre"),
    "TOP_CENTER": ("centre", "min"),
    "TOP_LEFT": ("min", "min"),
    "TOP_RIGHT": ("max", "min"),
    "BOTTOM_LEFT": ("min", "max"),
    "BOTTOM_CENTER": ("centre", "max"),
    "BOTTOM_RIGHT": ("max", "max"),
}

Anchor = Literal[tuple(ANCHORS)]  # type: ignore[valid-type]

@dataclass
class _Track:
    """Kept path of one track and the last Fréchet row against ``reference``."""

    path: List[Tuple[float, float]] = field(default_factory=list)
    row: Optional[np.ndarray] = None  # (M,) float64
    reference: Optional[bytes] = None  # reference points the row belongs to

class PathDeviationBlock(Block):
    """Attach each track's Fréchet distance from a reference path, per source.

    Example, ``reference_path=[[0, 0], [10, 0], [20, 0]]``, ``CENTER`` anchor,
    one track whose centre is 3 px below the path::

        frame              1       2       3
        centre           (0,3)  (10,3)  (20,3)
        path_deviation   20.2    10.4     3.0
    """

    type = "roboflow_core/path_deviation_analytics@v2"
    metadata = {
        "name": "Path Deviation",
        "port_of": "roboflow_core/path_deviation_analytics@v2",
        "section": "video",
    }
    outputs = {
        "path_deviation_detections": Output(
            *TRACKED_KINDS,
            source="predictions",
            description=(
                "The input rows with path_deviation (Fréchet distance in px) "
                "in bboxes_metadata."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata.video_identifier names the source "
                "of the tracks."
            )
        )
        predictions: Ref(*TRACKED_KINDS) = Field(
            description=(
                "Tracked rows with tracker_id in bboxes_metadata, e.g. a "
                "tracker's tracked_detections (V1 input 'detections')."
            ),
            examples=["$steps.byte_tracker.tracked_detections"],
        )
        reference_path: List[Tuple[float, float]] | Ref(LIST_OF_VALUES_KIND) = Field(
            min_length=2,
            description=(
                "Expected path: at least 2 [x, y] points in image pixels, in "
                "the order an object should travel them."
            ),
            examples=[[[100, 200], [200, 300], [300, 400]], "$inputs.path"],
        )
        triggering_anchor: Anchor | Ref(STRING_KIND) = Field(
            default="CENTER",
            description=("Box point that is followed: " + ", ".join(ANCHORS) + "."),
            examples=["CENTER", "BOTTOM_CENTER"],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with path state. A frame of one more source "
                "fails with TrackerCapacityError; no source is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._sources: Dict[str, "OrderedDict[int, _Track]"] = {}

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        reference_path: Sequence[Sequence[float]],
        triggering_anchor: str = "CENTER",
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> Dict[str, Any]:
        """Extend this frame's tracks and measure them against the path.

        Args:
            image: Frame naming the source.
            predictions: Tracked rows of this frame; not modified.
            reference_path: At least two ``[x, y]`` points.
            triggering_anchor: One of ``ANCHORS``.
            max_sources: Most sources with state.

        Returns:
            ``path_deviation_detections``: the input family with one value
            per row.

        Raises:
            ValueError: When a row has no ``tracker_id``, or the anchor or
                path (selected, or passed to a direct call) is invalid.
            TrackerCapacityError: When this is the first frame of a source
                and ``max_sources`` sources already have state.
        """
        reference = _reference_points(reference_path)
        edges = _anchor_edges(triggering_anchor)
        boxes = boxes_of(predictions)
        rows_metadata = [dict(row or {}) for row in boxes.bboxes_metadata or []]
        rows_metadata += [{} for _ in range(len(boxes) - len(rows_metadata))]
        tracker_ids = [row.get(TRACKER_ID_KEY) for row in rows_metadata]
        if any(tracker_id is None for tracker_id in tracker_ids):
            raise ValueError(
                f"{self.type} needs tracked predictions: every row needs a "
                "tracker_id in bboxes_metadata; connect a tracker first"
            )

        tracks = self._source_tracks(image, max_sources=max_sources)
        if not tracker_ids:
            return {"path_deviation_detections": predictions}

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
        anchors = _anchors(xyxy, edges)
        for row, tracker_id, point in zip(rows_metadata, tracker_ids, anchors):
            track = _recent_track(tracks, int(tracker_id))
            row[PATH_DEVIATION_KEY] = _extend(track, point, reference)

        measured = replace(boxes, bboxes_metadata=rows_metadata)
        result = {"path_deviation_detections": replace_boxes(predictions, measured)}

        return result

    def _source_tracks(
        self, image: ImageData, *, max_sources: int
    ) -> "OrderedDict[int, _Track]":
        source = _source_key(self, image)
        tracks = self._sources.get(source)
        if tracks is not None:
            return tracks

        if len(self._sources) >= max_sources:
            raise TrackerCapacityError(
                f"{self.type}: source {source!r} would be source "
                f"{len(self._sources) + 1} of max_sources={max_sources}; "
                "raise max_sources or reset processing to drop ended sources"
            )

        tracks = OrderedDict()
        self._sources[source] = tracks

        return tracks

def _recent_track(tracks: "OrderedDict[int, _Track]", tracker_id: int) -> _Track:
    # Most recently seen last; the least recent beyond the cap is dropped.
    track = tracks.pop(tracker_id, None) or _Track()
    tracks[tracker_id] = track
    if len(tracks) > MAX_TRACKS_PER_SOURCE:
        tracks.popitem(last=False)

    return track

def _extend(track: _Track, point: np.ndarray, reference: np.ndarray) -> float:
    """Append ``point`` to the track and return its Fréchet distance."""
    key = reference.tobytes()
    if track.reference != key:
        track.row = None
        for kept in track.path:
            track.row = _next_row(track.row, np.asarray(kept), reference)
        track.reference = key

    track.path.append((float(point[0]), float(point[1])))
    track.row = _next_row(track.row, point, reference)
    distance = float(track.row[-1])

    return distance

def _next_row(
    previous: Optional[np.ndarray], point: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Return Fréchet row ``F[i]`` from ``F[i-1]`` (``None`` for ``i = 0``)."""
    distances = np.sqrt(np.sum((reference - point) ** 2, axis=1))
    row = np.empty_like(distances)
    if previous is None:
        # F[0][j] = max(F[0][j-1], d[0][j]): the running maximum.
        np.maximum.accumulate(distances, out=row)
        return row

    # F[i][j] = max(d, min(F[i-1][j], F[i-1][j-1], F[i][j-1])); the first two
    # terms are vectorized, the last makes it a scan.
    diagonal = np.minimum(previous[1:], previous[:-1])
    row[0] = max(previous[0], distances[0])
    for j in range(1, len(row)):
        row[j] = max(distances[j], min(diagonal[j - 1], row[j - 1]))

    return row

def _anchors(xyxy: np.ndarray, edges: Tuple[str, str]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    xs = {"min": x1, "centre": (x1 + x2) / 2, "max": x2}[edges[0]]
    ys = {"min": y1, "centre": (y1 + y2) / 2, "max": y2}[edges[1]]
    anchors = np.stack([xs, ys], axis=1)

    return anchors

def _anchor_edges(anchor: Any) -> Tuple[str, str]:
    # Literals are checked at compile time; selected values and direct calls
    # are checked here.
    if not isinstance(anchor, str) or anchor not in ANCHORS:
        raise ValueError(
            f"triggering_anchor must be one of {', '.join(ANCHORS)}; got {anchor!r}"
        )

    return ANCHORS[anchor]

def _reference_points(path: Any) -> np.ndarray:
    """Return the path as ``(M, 2)`` float64, or raise a ``ValueError``."""
    is_list = isinstance(path, (list, tuple))
    points = list(path) if is_list else []
    valid = len(points) >= 2 and all(_is_point(point) for point in points)
    if not valid:
        raise ValueError(
            "reference_path must be a list of at least 2 [x, y] points with "
            f"finite numbers; got {path!r}"
        )

    reference = np.asarray(points, dtype=np.float64)

    return reference

def _is_point(point: Any) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False

    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in point
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
