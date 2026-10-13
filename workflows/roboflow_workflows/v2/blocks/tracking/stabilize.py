"""``v2/stabilize_detections@v1``: smooth tracked boxes and fill short gaps.

Port of V1 ``roboflow_core/stabilize_detections@v1``; its tensor variant is
the reference. The V1 ``detections`` input is ``predictions`` here. Per source
and tracker id, one call is one frame::

    shift     = corner(box now) - corner(cached box)    tracks cached before
    velocity  = Kalman(mean of the last smoothing_window_size shifts)
    seen row  : box = a * box now + (1 - a) * cached box      a = coefficient
    gap row   : box = cached box + a * (vx, vy, vx, vy)   cached, not seen now
    cache     = box now for seen rows, the gap box for gap rows

``corner`` is ``(x1 + |x2 - x1|, y1 + |y2 - y1|)``, the far corner, which V1
names the centre; for a rigid shift it moves exactly as the centre does. A
track missing from a frame loses its oldest shift; once none are left it
leaves the filter, and the frame after its last gap row it leaves the cache.
A track seen once has no shift yet, so it is never gap-filled.

Output rows: this frame's rows in input order (a tracker id seen twice keeps
its first position and its last row, as V1), then one gap row per missing
track in filter order. A gap row is the cached row of an earlier frame, mask
and row metadata included, with the predicted box. ``image_metadata`` is this
frame's. The rows are a new result, so the output gets a fresh row identity;
host mirrors of the changed boxes are dropped.

Frames are counted per source. An empty prediction is a frame and produces
gap rows; a filtered or missing prediction skips the call and is not counted.
``smoothing_window_size`` is read once per source, at its first frame (V1);
``bbox_smoothing_coefficient`` applies from the call that passes it.

Source key: ``video_metadata.video_identifier``, else the ``image`` input's
own ``SampleContext.source_id``, else ``"default"`` (also for a plain
``run()`` outside the engine). ``max_sources`` (default 256) bounds the
sources; the first frame of one source too many raises
``TrackerCapacityError`` before any state changes. Sources are never
evicted; a processing reset replaces the block and drops them all.

CPU boundary: the filter and box arithmetic run in float64 NumPy on one
device-to-host copy of ``xyxy``; the output boxes return in one host-to-device
copy, in the input's box dtype. Rows are kept by reference to their frame's
prediction, not copied; gap rows are gathered from it only when output.

Differences from V1:

* ``smoothing_window_size`` must be an integer of at least 1 and the
  coefficient a number from 0 to 1; V1 allowed ``None``.
* V1 shared one filter dictionary across videos keyed by video and had no
  source bound; V2 raises ``TrackerCapacityError`` past ``max_sources``.
* Gap rows need this frame's family and mask layout (dense size or RLE
  image size); a change within a source is an error, not a mixed batch.
* The output carries a fresh row identity instead of the input rows'.
"""

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.query.detections import without_host_mirror
from roboflow_workflows.v2.blocks.tracking.base import (
    DEFAULT_MAX_SOURCES,
    TrackerCapacityError,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    with_row_identity,
    without_row_record,
)
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

__all__ = [
    "DEFAULT_SOURCE",
    "STABILIZED_KINDS",
    "StabilizeDetectionsBlock",
]

DEFAULT_SOURCE = "default"
"""Source key of an image with neither a video identifier nor a source."""

STABILIZED_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)

TRACKER_ID_KEY = "tracker_id"

Boxes = Union[Detections, InstanceDetections]


@dataclass
class _CachedRow:
    """Last box of one track and where its row lives."""

    prediction: Boxes
    row: int
    xyxy: np.ndarray


@dataclass
class _SourceRows:
    """Cached rows and velocity filter of one source."""

    rows: Dict[int, _CachedRow]
    velocities: "_VelocityKalmanFilter"


class StabilizeDetectionsBlock(Block):
    """Smooth tracked boxes and fill short detection gaps per source.

    Example with ``bbox_smoothing_coefficient=0.5`` and a box moving 10 px
    right per frame (x1 shown; the other corners move with it)::

        frame     1     2     3     4 (missing)   5 (missing)   6 (missing)
        model     0     10    20    -             -             -
        output    0     5     15    25            30            no row

    Velocity is 10 px per frame from frame 2. Gap rows move by ``a * 10``.
    The default window of 3 holds two shifts after frame 3, so the track
    leaves the filter after frame 5.
    """

    type = "v2/stabilize_detections@v1"
    metadata = {
        "name": "Detections Stabilizer",
        "port_of": "roboflow_core/stabilize_detections@v1",
        "section": "video",
    }
    outputs = {
        "tracked_detections": Output(
            *STABILIZED_KINDS,
            source="predictions",
            description=(
                "This frame's rows with smoothed boxes, then predicted rows "
                "of tracks missing from it; same family as the input."
            ),
        ),
    }
    retains_inputs = ("predictions",)

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Frame whose video_metadata names the source."
        )
        predictions: Ref(*STABILIZED_KINDS) = Field(
            description=(
                "Tracked detections or instance segmentations with tracker_id "
                "in bboxes_metadata (V1 input 'detections')."
            ),
            examples=["$steps.byte_tracker.tracked_detections"],
        )
        smoothing_window_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=3,
            ge=1,
            description=(
                "Shifts averaged by the velocity filter, and frames a missing "
                "track is gap-filled at most. Read once per source, at its "
                "first frame."
            ),
            examples=[3, "$inputs.smoothing_window_size"],
        )
        bbox_smoothing_coefficient: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=0.2,
            ge=0,
            le=1,
            description=(
                "Weight a of the current box: a * current + (1 - a) * "
                "previous. Lower is smoother and slower."
            ),
            examples=[0.2, "$inputs.bbox_smoothing_coefficient"],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with cached rows. A frame of one more source "
                "fails with TrackerCapacityError; no source is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._sources: Dict[str, _SourceRows] = {}

    def run(
        self,
        *,
        image: ImageData,
        predictions: Boxes,
        smoothing_window_size: int,
        bbox_smoothing_coefficient: float,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> Dict[str, Any]:
        """Smooth this frame's tracked boxes and predict missing tracks.

        Args:
            image: Frame naming the source.
            predictions: Tracked rows of this frame; not modified.
            smoothing_window_size: Filter window of a new source.
            bbox_smoothing_coefficient: Weight of the current box.
            max_sources: Most sources with state.

        Returns:
            ``tracked_detections``: smoothed and gap rows, input family.

        Raises:
            ValueError: When a row has no ``tracker_id``, or a gap row's
                family or mask layout differs from this frame's.
            TrackerCapacityError: When this is the first frame of a source
                and ``max_sources`` sources already have state.
        """
        rows_metadata = predictions.bboxes_metadata or [{}] * len(predictions)
        tracker_ids = [(row or {}).get(TRACKER_ID_KEY) for row in rows_metadata]
        if any(tracker_id is None for tracker_id in tracker_ids):
            raise ValueError(
                f"{self.type} needs tracked predictions: every row needs a "
                "tracker_id in bboxes_metadata; connect a tracker first"
            )

        source = self._source_state(
            image, max_sources=max_sources, window=smoothing_window_size
        )
        alpha = bbox_smoothing_coefficient
        xyxy = predictions.xyxy.detach().cpu().double().numpy()
        cached = source.rows
        shifts = {
            tracker_id: _corner(xyxy[row]) - _corner(cached[tracker_id].xyxy)
            for row, tracker_id in enumerate(tracker_ids)
            if tracker_id in cached
        }
        velocities = source.velocities.update(shifts)

        output: Dict[int, Tuple[Boxes, int, np.ndarray]] = {}
        for row, tracker_id in enumerate(tracker_ids):
            box = xyxy[row]
            if tracker_id in cached:
                box = _smooth(cached[tracker_id].xyxy, box, alpha=alpha)
            output[tracker_id] = (predictions, row, box)
            # The raw box, not the smoothed one, is next frame's reference.
            cached[tracker_id] = _CachedRow(predictions, row, xyxy[row])

        for tracker_id, velocity in velocities.items():
            if tracker_id in output:
                continue

            entry = cached[tracker_id]
            entry.xyxy = _smooth(
                entry.xyxy, entry.xyxy + np.tile(velocity, 2), alpha=alpha
            )
            output[tracker_id] = (entry.prediction, entry.row, entry.xyxy)

        for tracker_id in list(cached):
            if tracker_id not in source.velocities.tracks and tracker_id not in output:
                del cached[tracker_id]

        stabilized = _stabilized_rows(predictions, list(output.values()))
        result = {"tracked_detections": stabilized}

        return result

    def _source_state(
        self, image: ImageData, *, max_sources: int, window: int
    ) -> _SourceRows:
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

        state = _SourceRows(rows={}, velocities=_VelocityKalmanFilter(window))
        self._sources[source] = state

        return state


class _VelocityKalmanFilter:
    """Per-track 2-D velocity filter of V1, unchanged.

    Identity transition; process noise 0.001 I, measurement noise 0.01 I,
    initial covariance I. A measured track is corrected toward the mean of
    its last ``window`` shifts; an unmeasured one loses its oldest shift and
    leaves when none are left.
    """

    _PROCESS_NOISE = np.eye(2) * 0.001
    _MEASUREMENT_NOISE = np.eye(2) * 0.01

    def __init__(self, window: int) -> None:
        self.window = window
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, shifts: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Correct measured tracks, then predict every track's velocity.

        Args:
            shifts: Measured ``(dx, dy)`` per tracker id.

        Returns:
            ``(dx, dy)`` velocity per tracker id kept before this frame's
            ageing, in filter order.
        """
        for tracker_id, shift in shifts.items():
            measurement = np.asarray(shift, dtype=float).reshape(2, 1)
            track = self.tracks.get(tracker_id)
            if track is None:
                self.tracks[tracker_id] = {
                    "velocity": measurement,
                    "covariance": np.eye(2),
                    "history": deque([measurement], maxlen=self.window),
                }
                continue

            history: Deque[np.ndarray] = track["history"]
            history.appendleft(measurement)
            residual = np.mean(history, axis=0) - track["velocity"]
            gain = track["covariance"] @ np.linalg.inv(
                track["covariance"] + self._MEASUREMENT_NOISE
            )
            track["velocity"] = track["velocity"] + gain @ residual
            track["covariance"] = track["covariance"] - gain @ track["covariance"]

        velocities = {}
        for tracker_id, track in self.tracks.items():
            track["covariance"] = track["covariance"] + self._PROCESS_NOISE
            velocities[tracker_id] = track["velocity"].reshape(2)

        for tracker_id in set(self.tracks) - set(shifts):
            history = self.tracks[tracker_id]["history"]
            if history:
                history.popleft()
            if not history:
                del self.tracks[tracker_id]

        return velocities


def _corner(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    corner = np.array([x1 + abs(x2 - x1), y1 + abs(y2 - y1)])

    return corner


def _smooth(previous: np.ndarray, current: np.ndarray, *, alpha: float) -> np.ndarray:
    smoothed = alpha * current + (1 - alpha) * previous

    return smoothed


def _stabilized_rows(
    current: Boxes, output: List[Tuple[Boxes, int, np.ndarray]]
) -> Boxes:
    # Consecutive rows of one frame's prediction are gathered together.
    parts: List[Boxes] = []
    start = 0
    for end in range(1, len(output) + 1):
        if end == len(output) or output[end][0] is not output[start][0]:
            rows = [row for _, row, _ in output[start:end]]
            parts.append(select_predictions(output[start][0], indices=rows))
            start = end
    if not parts:
        parts.append(select_predictions(current, indices=[]))
    _check_layouts(current, parts)

    boxes = np.stack([box for _, _, box in output]) if output else np.empty((0, 4))
    xyxy = torch.as_tensor(boxes, dtype=current.xyxy.dtype, device=current.xyxy.device)
    rows_metadata = without_host_mirror(
        [dict(row or {}) for part in parts for row in _rows_of(part)]
    )
    joined = replace(
        parts[0],
        xyxy=xyxy,
        class_id=torch.cat([part.class_id for part in parts]),
        confidence=torch.cat([part.confidence for part in parts]),
        image_metadata=without_row_record(current.image_metadata),
        bboxes_metadata=rows_metadata or None,
    )
    if isinstance(current, InstanceDetections):
        joined = replace(joined, mask=_joined_masks(current, parts))
    stabilized = with_row_identity(joined)

    return stabilized


def _rows_of(part: Boxes) -> List[Optional[dict]]:
    rows = part.bboxes_metadata or [None] * len(part)

    return rows


def _check_layouts(current: Boxes, parts: List[Boxes]) -> None:
    expected = _layout(current)
    others = {_layout(part) for part in parts} - {expected}
    if others:
        raise ValueError(
            f"stabilize_detections cannot add gap rows of layout {sorted(others)} "
            f"to a frame of layout {expected}; reset processing after changing "
            "the source resolution or model"
        )


def _layout(prediction: Boxes) -> Tuple[Any, ...]:
    # Rows of one output need one family and one mask size.
    if not isinstance(prediction, InstanceDetections):
        return ("boxes",)

    if isinstance(prediction.mask, InstancesRLEMasks):
        return ("rle", tuple(prediction.mask.image_size))

    layout = ("dense", tuple(prediction.mask.shape[1:]))

    return layout


def _joined_masks(
    current: InstanceDetections, parts: List[InstanceDetections]
) -> Union[torch.Tensor, InstancesRLEMasks]:
    if isinstance(current.mask, InstancesRLEMasks):
        masks = InstancesRLEMasks(
            image_size=current.mask.image_size,
            masks=[mask for part in parts for mask in part.mask.masks],
        )
        return masks

    masks = torch.cat([part.mask for part in parts])

    return masks


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
