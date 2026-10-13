"""OC-SORT from the ``trackers`` package, with one tracker per video source.

This is the V1 ``roboflow_core/trackers_ocsort@v1`` capability. Shared
behavior is described in ``package``.
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

# V1 OCSORTBlockV1 defaults.
DEFAULT_LOST_TRACK_BUFFER = 30
DEFAULT_MINIMUM_CONSECUTIVE_FRAMES = 3
DEFAULT_MINIMUM_IOU_THRESHOLD = 0.3
DEFAULT_HIGH_CONF_DET_THRESHOLD = 0.6
DEFAULT_DIRECTION_CONSISTENCY_WEIGHT = 0.2
DEFAULT_DELTA_T = 3

__all__ = [
    "DEFAULT_DELTA_T",
    "DEFAULT_DIRECTION_CONSISTENCY_WEIGHT",
    "DEFAULT_HIGH_CONF_DET_THRESHOLD",
    "DEFAULT_LOST_TRACK_BUFFER",
    "DEFAULT_MINIMUM_CONSECUTIVE_FRAMES",
    "DEFAULT_MINIMUM_IOU_THRESHOLD",
    "OCSORTTrackerBlock",
]


class OCSORTTrackerBlock(TrackersPackageBlock):
    """Track detections with one ``trackers.OCSORTTracker`` per video source.

    OC-SORT adds a motion-direction term to IoU association and re-anchors a
    track on its last observation after an occlusion, which suits non-linear
    motion. It has no activation threshold: detections below
    ``high_conf_det_threshold`` are ignored, and the rest may start tracks.
    While the tracker has seen at most ``minimum_consecutive_frames`` frames,
    a track matched in the current frame is reported early (warm-up rule).
    """

    type = "v2/trackers_ocsort@v1"
    library_class = "OCSORTTracker"

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
                "Consecutive matches a track needs after its first detection, "
                "which does not count, before it is reported. A missed frame "
                "restarts the count, even for a reported track: it is hidden "
                "until it has them again."
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
        high_conf_det_threshold: StrictFloat | Ref(FLOAT_KIND) = fraction_knob(
            DEFAULT_HIGH_CONF_DET_THRESHOLD,
            name="high_conf_det_threshold",
            description=(
                "Minimum confidence of a detection OC-SORT uses at all; lower "
                "detections are ignored and never reported."
            ),
        )
        direction_consistency_weight: StrictFloat | Ref(FLOAT_KIND) = fraction_knob(
            DEFAULT_DIRECTION_CONSISTENCY_WEIGHT,
            name="direction_consistency_weight",
            description=(
                "Weight of agreement between a track's motion direction and "
                "the direction to a candidate detection."
            ),
        )
        delta_t: StrictInt | Ref(INTEGER_KIND) = frame_count_knob(
            DEFAULT_DELTA_T,
            name="delta_t",
            description="Past frames used to estimate a track's direction.",
        )

    def run(
        self,
        *,
        image: ImageData,
        predictions: TrackedPrediction,
        minimum_iou_threshold: float,
        minimum_consecutive_frames: int,
        lost_track_buffer: int,
        high_conf_det_threshold: float,
        direction_consistency_weight: float,
        delta_t: int,
        max_sources: int,
        frame_rate: Optional[float] = None,
        instances_cache_size: int = DEFAULT_INSTANCES_CACHE_SIZE,
    ) -> Dict[str, Any]:
        """Track one frame of one source; see ``TrackersPackageBlock.track``.

        Args:
            image: Frame naming the source in ``video_metadata``.
            predictions: Detections of this frame; not modified.
            minimum_iou_threshold: Association IoU.
            minimum_consecutive_frames: Consecutive matches after the first
                detection before a track is reported; a miss restarts the
                count.
            lost_track_buffer: Unmatched frames at 30 fps before deletion.
            high_conf_det_threshold: Confidence below which detections are
                ignored.
            direction_consistency_weight: Weight of the direction term.
            delta_t: Frames used for a track's direction.
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
                high_conf_det_threshold=high_conf_det_threshold,
                direction_consistency_weight=direction_consistency_weight,
                delta_t=delta_t,
            ),
        )

        return result
