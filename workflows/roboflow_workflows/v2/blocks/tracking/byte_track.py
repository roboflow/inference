"""ByteTrack over native predictions, with one tracker per video source.

This is the V1 ``roboflow_core/byte_tracker@v3`` capability. The V1
``detections`` input is ``predictions`` here. Unlike V1, keypoints of a
keypoint prediction stay on every output, aligned with the kept rows, and an
instance segmentation model's ``segmentation`` view can be tracked without a
full-resolution mask.

Shared source, cache, capacity and frame-rate policies live in ``base``.
ByteTrack scales its 30-fps ``lost_track_buffer`` to
``int(rate / 30 * lost_track_buffer)`` processed frames at the nominal source
rate. Processing throughput does not change that scaling.
"""

from typing import Any, Dict, Optional

import supervision as sv
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.tracking.base import (
    DEFAULT_INSTANCES_CACHE_SIZE,
    DEFAULT_MAX_SOURCES,
    TRACKED_KINDS,
    TRACKER_OUTPUTS,
    SourceTrackers,
    TrackedPrediction,
    TrackerCapacityError,
    host_boxes,
    kept_rows,
    tracking_result,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine import NoExecutionContextError, get_execution_context
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref

# V1 ByteTrackerBlockV3 defaults.
DEFAULT_TRACK_ACTIVATION_THRESHOLD = 0.25
DEFAULT_LOST_TRACK_BUFFER = 30
DEFAULT_MINIMUM_MATCHING_THRESHOLD = 0.8
DEFAULT_MINIMUM_CONSECUTIVE_FRAMES = 1

_FIRST_FRAME_NOTE = (
    "Read once per source, at its first frame; a later value, selected or "
    "edited, does not retune it until a processing reset."
)

__all__ = [
    "DEFAULT_INSTANCES_CACHE_SIZE",
    "DEFAULT_LOST_TRACK_BUFFER",
    "DEFAULT_MAX_SOURCES",
    "DEFAULT_MINIMUM_CONSECUTIVE_FRAMES",
    "DEFAULT_MINIMUM_MATCHING_THRESHOLD",
    "DEFAULT_TRACK_ACTIVATION_THRESHOLD",
    "TRACKED_KINDS",
    "ByteTrackerBlock",
    "TrackerCapacityError",
]


class ByteTrackerBlock(Block):
    """Track detections with one ``sv.ByteTrack`` per video source.

    The source is ``image.video_metadata.video_identifier``. The nominal frame
    rate is ``frame_rate`` when set, otherwise ``video_metadata.fps``. Every
    output has the input's family and image metadata: rows ByteTrack kept, in
    ByteTrack order, each with its ``tracker_id`` in ``bboxes_metadata``.
    ``new_instances`` and ``already_seen_instances`` split those rows by
    whether the source reported the id before. The input is not changed.
    """

    type = "v2/byte_tracker@v1"
    outputs = TRACKER_OUTPUTS

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame whose video_metadata names the source and, unless "
                "frame_rate is set, gives its nominal fps."
            )
        )
        predictions: Ref(*TRACKED_KINDS) = Field(
            description=(
                "Detections, instance segmentations (dense, RLE or a "
                "segmentation view) or keypoints with boxes of this frame "
                "(V1 input 'detections')."
            )
        )
        track_activation_threshold: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=DEFAULT_TRACK_ACTIVATION_THRESHOLD,
            ge=0,
            le=1,
            description=(
                f"Minimum confidence that starts a new track. {_FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_TRACK_ACTIVATION_THRESHOLD, "$inputs.activation"],
        )
        lost_track_buffer: StrictInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_LOST_TRACK_BUFFER,
            ge=1,
            description=(
                "Lost-track buffer at 30 fps; ByteTrack scales it by the "
                "frame rate. Only processed frames advance the tracker. "
                f"{_FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_LOST_TRACK_BUFFER, "$inputs.lost_track_buffer"],
        )
        minimum_matching_threshold: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=DEFAULT_MINIMUM_MATCHING_THRESHOLD,
            ge=0,
            le=1,
            description=f"ByteTrack matching threshold. {_FIRST_FRAME_NOTE}",
            examples=[DEFAULT_MINIMUM_MATCHING_THRESHOLD, "$inputs.matching"],
        )
        minimum_consecutive_frames: StrictInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
            ge=1,
            description=(
                "Frames a track must persist before it is reported. "
                f"{_FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_MINIMUM_CONSECUTIVE_FRAMES, "$inputs.min_frames"],
        )
        instances_cache_size: StrictInt = Field(
            default=DEFAULT_INSTANCES_CACHE_SIZE,
            ge=1,
            description=(
                "Tracker ids each source remembers to tell new_instances from "
                "already_seen_instances; the oldest is dropped first, and a "
                f"dropped id counts as new again. {_FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_INSTANCES_CACHE_SIZE],
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
        self._trackers = SourceTrackers(block_type=self.type)

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
        instances_cache_size: int = DEFAULT_INSTANCES_CACHE_SIZE,
    ) -> Dict[str, Any]:
        """Update this source's tracker and instance cache with one frame.

        Args:
            image: Frame naming the source in ``video_metadata``.
            predictions: Detections of this frame; not modified.
            track_activation_threshold: ByteTrack activation threshold.
            lost_track_buffer: ByteTrack lost track buffer.
            minimum_matching_threshold: ByteTrack matching threshold.
            minimum_consecutive_frames: ByteTrack minimum consecutive frames.
            max_sources: Most sources this block keeps a tracker for.
            frame_rate: Nominal fps for a new source's tracker; None uses
                ``video_metadata.fps``.
            instances_cache_size: Tracker ids a new source's cache keeps.

        Returns:
            Demanded outputs among ``tracked_detections``, ``new_instances``
            and ``already_seen_instances``, as predictions of the input's
            family. Direct calls return all three.

        Notes:
            The four ByteTrack values, ``frame_rate`` and ``instances_cache_size``
            are read only when a source's state is created.

        Raises:
            ValueError: When the image has no video identifier, when a
                keypoint prediction has no boxes, when ``frame_rate`` is not a
                finite positive number, or when it is None and
                ``video_metadata.fps`` is not one.
            TrackerCapacityError: When this is the first frame of a source and
                ``max_sources`` sources already have trackers.
        """
        # Copied first: a malformed prediction leaves no source state behind.
        boxes = host_boxes(predictions)

        def create_tracker(rate: float) -> sv.ByteTrack:
            tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                frame_rate=rate,
            )

            return tracker

        state = self._trackers.state_for(
            image,
            max_sources=max_sources,
            frame_rate=frame_rate,
            instances_cache_size=instances_cache_size,
            create_tracker=create_tracker,
        )

        tracked = state.tracker.update_with_detections(boxes)
        rows, tracker_ids = kept_rows(tracked)
        new_positions, seen_positions = state.instances.split(tracker_ids.tolist())

        result = tracking_result(
            predictions,
            rows=rows,
            tracker_ids=tracker_ids,
            new_positions=new_positions,
            seen_positions=seen_positions,
            wanted_outputs=self._wanted_outputs(),
        )

        return result

    def _wanted_outputs(self) -> set[str]:
        # wants() requires a running call. Detect a plain run() through the
        # public context API; errors in an engine call must still propagate.
        try:
            get_execution_context()
        except NoExecutionContextError:
            return set(self.outputs)

        wanted = {name for name in self.outputs if self.wants(name)}

        return wanted
