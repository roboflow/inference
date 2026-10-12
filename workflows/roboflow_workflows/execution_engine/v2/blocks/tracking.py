"""ByteTrack over native predictions, with one tracker per video source.

Data flow of one call::

    image.video_metadata.video_identifier ── source key ── this source's sv.ByteTrack
    predictions (Detections | InstanceDetections, any device)
      ├─ xyxy / class_id / confidence -> host (three device-to-host copies)
      ├─ sv.ByteTrack.update_with_detections -> kept input rows, tracker ids
      └─ select_predictions(rows) on the ORIGINAL tensors (masks included)
           └─ tracked_detections, tracker_id in copied bboxes_metadata

This is the ByteTrack path of the V1 tensor block (``byte_tracker/v3_tensor``)
and example 26. The index recovery and tracker-id writing are restated here:
importing the original helpers also loads the V1 engine and block prototypes.

State and its limits:

* Sources never share a tracker. A source's tracker is created on its first
  frame, with that call's parameters and frame rate; later values do not
  reconfigure it.
* The frame rate only scales ``lost_track_buffer``, which is written for
  30 fps: a lost track is dropped after ``int(rate / 30 * lost_track_buffer)``
  processed frames. It is the source's nominal rate, not the measured
  processing throughput. ``frame_rate`` set: used for every source.
  ``frame_rate`` null: the frame's ``video_metadata.fps``, which must be finite
  and positive. A source that cannot report its rate (a live RTSP stream may
  report 0) needs ``frame_rate``; the block never guesses one.
* A block has no callback when a source ends, so trackers are kept until a
  processing reset replaces the block. ``max_sources`` bounds them: the first
  frame of one source too many raises ``TrackerCapacityError`` before any
  tracker is created or updated. Live trackers are never evicted.
* Tracking advances once per processed frame. A frame dropped before the step,
  or a call skipped because its predictions are missing, is never seen: the
  tracker does not learn that time passed. An empty prediction is a frame
  with no detections and does age tracks.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.prediction_ops import (
    select_predictions,
)
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

# V1 ByteTrackerBlockV3 defaults.
DEFAULT_TRACK_ACTIVATION_THRESHOLD = 0.25
DEFAULT_LOST_TRACK_BUFFER = 30
DEFAULT_MINIMUM_MATCHING_THRESHOLD = 0.8
DEFAULT_MINIMUM_CONSECUTIVE_FRAMES = 1
DEFAULT_MAX_SOURCES = 256

TRACKED_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)

_INPUT_INDEX_KEY = "input_row"

TrackedPrediction = Union[Detections, InstanceDetections]


class TrackerCapacityError(RuntimeError):
    """A new source would exceed the tracker block's ``max_sources``."""


class ByteTrackerBlock(Block):
    """Track detections with one ``sv.ByteTrack`` per video source.

    The source is ``image.video_metadata.video_identifier``. The nominal frame
    rate is ``frame_rate`` when set, otherwise ``video_metadata.fps``.
    ``tracked_detections`` has the input's family and image metadata: the rows ByteTrack kept, in ByteTrack order, each with its
    ``tracker_id`` in ``bboxes_metadata``. The input prediction is not changed.
    """

    type = "v2/byte_tracker@v1"
    outputs = {
        "tracked_detections": Output(
            *TRACKED_KINDS,
            source="image",
            description=(
                "Rows ByteTrack kept, in ByteTrack order, same family as the "
                "input, with tracker_id in bboxes_metadata."
            ),
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata names the source and, unless "
                "frame_rate is set, gives its nominal fps."
            )
        )
        predictions: Ref(*TRACKED_KINDS) = Field(
            description="Detections or instance segmentations of this frame."
        )
        track_activation_threshold: StrictFloat = Field(
            default=DEFAULT_TRACK_ACTIVATION_THRESHOLD,
            ge=0,
            le=1,
            description="Minimum confidence that starts a new track.",
            examples=[DEFAULT_TRACK_ACTIVATION_THRESHOLD],
        )
        lost_track_buffer: StrictInt = Field(
            default=DEFAULT_LOST_TRACK_BUFFER,
            ge=1,
            description=(
                "Lost-track buffer at 30 fps; ByteTrack scales it by the "
                "frame rate. Only processed frames advance the tracker."
            ),
            examples=[DEFAULT_LOST_TRACK_BUFFER],
        )
        minimum_matching_threshold: StrictFloat = Field(
            default=DEFAULT_MINIMUM_MATCHING_THRESHOLD,
            ge=0,
            le=1,
            description="ByteTrack matching threshold.",
            examples=[DEFAULT_MINIMUM_MATCHING_THRESHOLD],
        )
        minimum_consecutive_frames: StrictInt = Field(
            default=DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
            ge=1,
            description="Frames a track must persist before it is reported.",
            examples=[DEFAULT_MINIMUM_CONSECUTIVE_FRAMES],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources tracked at once. A frame of one more source "
                "fails with TrackerCapacityError; no tracker is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )
        frame_rate: Optional[StrictFloat] = Field(
            default=None,
            gt=0,
            allow_inf_nan=False,
            description=(
                "Nominal source fps that scales lost_track_buffer, for every "
                "source. Null uses video_metadata.fps and fails when a source "
                "does not report a finite positive fps (e.g. live RTSP). Not "
                "the processing rate; dropped frames do not age tracks."
            ),
            examples=[None, 30.0],
        )

    def __init__(self):
        self._trackers: Dict[str, sv.ByteTrack] = {}

    def run(
        self,
        *,
        image: ImageData,
        predictions: TrackedPrediction,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_matching_threshold: float,
        minimum_consecutive_frames: int,
        max_sources: int,
        frame_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update this source's tracker with one frame.

        Args:
            image: Frame naming the source in ``video_metadata``.
            predictions: Detections of this frame; not modified.
            track_activation_threshold: ByteTrack activation threshold.
            lost_track_buffer: ByteTrack lost track buffer.
            minimum_matching_threshold: ByteTrack matching threshold.
            minimum_consecutive_frames: ByteTrack minimum consecutive frames.
            max_sources: Most sources this block keeps a tracker for.
            frame_rate: Nominal fps for every source's tracker; None uses
                ``video_metadata.fps``. Read only when a source's tracker is
                created.

        Returns:
            ``tracked_detections``: a new prediction of the input's family.

        Raises:
            ValueError: When the image has no video identifier, when
                ``frame_rate`` is not a finite positive number, or when it is
                None and ``video_metadata.fps`` is not one.
            TrackerCapacityError: When this is the first frame of a source and
                ``max_sources`` sources already have trackers.
        """
        source = self._source_of(image)
        tracker = self._trackers.get(source)
        is_new_source = tracker is None
        if is_new_source:
            if len(self._trackers) >= max_sources:
                raise TrackerCapacityError(
                    f"{self.type}: source {source!r} would be source "
                    f"{len(self._trackers) + 1} of max_sources={max_sources}; "
                    "raise max_sources or reset processing to drop ended sources"
                )
            rate = self._frame_rate_of(image, source=source, frame_rate=frame_rate)
            tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                frame_rate=rate,
            )

        # A tracker is kept only once its first frame was copied to the host.
        host_detections = _host_detections(predictions)
        if is_new_source:
            self._trackers[source] = tracker

        kept_rows, tracker_ids = _track(tracker, host_detections)
        tracked = select_predictions(predictions, indices=kept_rows)
        _write_tracker_ids(tracked, tracker_ids)
        result = {"tracked_detections": tracked}

        return result

    def _source_of(self, image: ImageData) -> str:
        metadata = image.video_metadata
        if metadata is None or not metadata.video_identifier:
            raise ValueError(f"{self.type} needs video_metadata.video_identifier")

        return metadata.video_identifier

    def _frame_rate_of(
        self, image: ImageData, *, source: str, frame_rate: Optional[float]
    ) -> float:
        # Direct run() calls bypass Params validation, so check here too.
        if frame_rate is not None:
            if not _is_positive_finite(frame_rate):
                raise ValueError(
                    f"{self.type}: frame_rate must be a finite positive number "
                    f"or null, got {frame_rate!r}"
                )

            return float(frame_rate)
        fps = image.video_metadata.fps
        if not _is_positive_finite(fps):
            raise ValueError(
                f"{self.type}: source {source!r} reports video_metadata.fps="
                f"{fps!r}, not a finite positive nominal rate (a live stream "
                "may not report one); set the step's frame_rate to the "
                "source's nominal fps"
            )

        return float(fps)


def _is_positive_finite(value: Any) -> bool:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    return is_number and math.isfinite(value) and value > 0


def _host_detections(predictions: TrackedPrediction) -> sv.Detections:
    # Three device-to-host copies; each row is tagged with its input position
    # because ByteTrack filters and reorders rows.
    rows = int(predictions.xyxy.shape[0])
    host_detections = sv.Detections(
        xyxy=predictions.xyxy.detach().to("cpu").numpy(),
        class_id=predictions.class_id.detach().to("cpu").numpy(),
        confidence=predictions.confidence.detach().to("cpu").numpy(),
        data={_INPUT_INDEX_KEY: np.arange(rows, dtype=np.int64)},
    )

    return host_detections


def _track(
    tracker: sv.ByteTrack, host_detections: sv.Detections
) -> Tuple[List[int], np.ndarray]:
    # Kept input rows and their tracker ids, both in ByteTrack output order.
    tracked = tracker.update_with_detections(host_detections)
    kept_positions = tracked.data.get(_INPUT_INDEX_KEY, np.empty((0,), np.int64))
    kept_rows = [int(row) for row in np.asarray(kept_positions).tolist()]
    tracker_ids = tracked.tracker_id
    if tracker_ids is None:
        tracker_ids = np.full(len(tracked), -1, dtype=np.int64)

    return kept_rows, tracker_ids


def _write_tracker_ids(tracked: TrackedPrediction, tracker_ids: np.ndarray) -> None:
    # ``tracked`` is a new selection; its row dicts are copied again so no
    # dict of the input prediction can change.
    existing = tracked.bboxes_metadata
    rows_metadata = []
    for row, tracker_id in enumerate(tracker_ids.tolist()):
        row_metadata = dict(existing[row]) if existing and existing[row] else {}
        row_metadata["tracker_id"] = int(tracker_id)
        rows_metadata.append(row_metadata)
    tracked.bboxes_metadata = rows_metadata or None
