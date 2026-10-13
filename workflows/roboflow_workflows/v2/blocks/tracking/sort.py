"""SORT from the ``trackers`` package, with one tracker per video source.

This is the V1 ``roboflow_core/trackers_sort@v1`` capability. Shared behavior
is described in ``package``.
"""

from typing import Any, Dict, Optional

from pydantic import StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.tracking.base import (
    DEFAULT_INSTANCES_CACHE_SIZE,
    TrackedPrediction,
)
from roboflow_workflows.v2.blocks.tracking.package import (
    TrackersPackageBlock,
    fraction_knob,
    frame_count_knob,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Ref

# V1 SORTBlockV1 defaults.
DEFAULT_LOST_TRACK_BUFFER = 30
DEFAULT_TRACK_ACTIVATION_THRESHOLD = 0.25
DEFAULT_MINIMUM_CONSECUTIVE_FRAMES = 3
DEFAULT_MINIMUM_IOU_THRESHOLD = 0.3

__all__ = [
    "DEFAULT_LOST_TRACK_BUFFER",
    "DEFAULT_MINIMUM_CONSECUTIVE_FRAMES",
    "DEFAULT_MINIMUM_IOU_THRESHOLD",
    "DEFAULT_TRACK_ACTIVATION_THRESHOLD",
    "SORTTrackerBlock",
]


class SORTTrackerBlock(TrackersPackageBlock):
    """Track detections with one ``trackers.SORTTracker`` per video source.

    SORT matches detections to Kalman-predicted tracks by IoU in a single
    stage. Only detections of at least ``track_activation_threshold`` start
    tracks, and a track is reported once it has ``minimum_consecutive_frames``
    matches.
    """

    type = "v2/trackers_sort@v1"
    library_class = "SORTTracker"

    class Params(TrackersPackageBlock.Params):
        minimum_iou_threshold: StrictFloat | Ref(FLOAT_KIND) = fraction_knob(
            DEFAULT_MINIMUM_IOU_THRESHOLD,
            name="minimum_iou_threshold",
            description=(
                "Minimum IoU that associates a detection with an existing track."
            ),
        )
        minimum_consecutive_frames: StrictInt | Ref(INTEGER_KIND) = frame_count_knob(
            DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
            name="minimum_consecutive_frames",
            description=(
                "Frames a track must be detected in, counting its first, before "
                "it is reported; a missed frame does not restart the count."
            ),
        )
        lost_track_buffer: StrictInt | Ref(INTEGER_KIND) = frame_count_knob(
            DEFAULT_LOST_TRACK_BUFFER,
            name="lost_track_buffer",
            description=(
                "Frames at 30 fps a track survives unmatched; scaled by the "
                "frame rate. Only processed frames advance the tracker."
            ),
        )
        track_activation_threshold: StrictFloat | Ref(FLOAT_KIND) = fraction_knob(
            DEFAULT_TRACK_ACTIVATION_THRESHOLD,
            name="track_activation_threshold",
            description="Minimum confidence that starts a new track.",
        )

    def run(
        self,
        *,
        image: ImageData,
        predictions: TrackedPrediction,
        minimum_iou_threshold: float,
        minimum_consecutive_frames: int,
        lost_track_buffer: int,
        track_activation_threshold: float,
        max_sources: int,
        frame_rate: Optional[float] = None,
        instances_cache_size: int = DEFAULT_INSTANCES_CACHE_SIZE,
    ) -> Dict[str, Any]:
        """Track one frame of one source; see ``TrackersPackageBlock.track``.

        Args:
            image: Frame naming the source in ``video_metadata``.
            predictions: Detections of this frame; not modified.
            minimum_iou_threshold: SORT association IoU.
            minimum_consecutive_frames: Detected frames, counting the first,
                before a track is reported; misses do not restart the count.
            lost_track_buffer: Unmatched frames at 30 fps before deletion.
            track_activation_threshold: Confidence that starts a track.
            max_sources: Most sources this block keeps a tracker for.
            frame_rate: Nominal fps for a new source; None uses
                ``video_metadata.fps``.
            instances_cache_size: Tracker ids a new source's cache keeps.

        Returns:
            Demanded tracker outputs; direct calls return all three.
        """
        result = self.track(
            image=image,
            predictions=predictions,
            max_sources=max_sources,
            frame_rate=frame_rate,
            instances_cache_size=instances_cache_size,
            tracker_settings=dict(
                minimum_iou_threshold=minimum_iou_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                lost_track_buffer=lost_track_buffer,
                track_activation_threshold=track_activation_threshold,
            ),
        )

        return result
