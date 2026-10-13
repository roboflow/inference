"""BoT-SORT from the ``trackers`` package, with one tracker per video source.

This is the V1 ``roboflow_core/trackers_botsort@v1`` capability. Shared
behavior is described in ``package``. BoT-SORT uses no appearance (ReID) model
here, as in V1.

Camera motion compensation (CMC) is the one place a tracker block reads
pixels. With ``enable_cmc`` the frame is copied to the host as a BGR ``uint8``
array once per processed frame, and OpenCV estimates the camera motion on the
CPU::

    enable_cmc false:  boxes ──► tracker.update(boxes)
    enable_cmc true:   boxes ─┬─► tracker.update(boxes, frame=bgr)
                       image ─┴─► host BGR copy (CPU boundary)

Without CMC no pixel leaves the image's device.
"""

from typing import Any, Dict, Literal, Optional

import cv2
import numpy as np
import supervision as sv
from pydantic import Field, StrictBool, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.tracking.base import (
    DEFAULT_INSTANCES_CACHE_SIZE,
    TrackedPrediction,
)
from roboflow_workflows.v2.blocks.tracking.package import (
    FIRST_FRAME_NOTE,
    TrackersPackageBlock,
    fraction_knob,
    frame_count_knob,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Ref

CMCMethod = Literal["orb", "sift", "sparseOptFlow", "ecc"]

# V1 BoTSORTBlockV1 defaults; enable_cmc and instant_first_frame_activation
# differ from the library defaults on purpose.
DEFAULT_LOST_TRACK_BUFFER = 30
DEFAULT_TRACK_ACTIVATION_THRESHOLD = 0.7
DEFAULT_MINIMUM_CONSECUTIVE_FRAMES = 2
DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC = 0.2
DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC = 0.5
DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC = 0.3
DEFAULT_HIGH_CONF_DET_THRESHOLD = 0.6
DEFAULT_ENABLE_CMC = False
DEFAULT_CMC_METHOD: CMCMethod = "sparseOptFlow"
DEFAULT_CMC_DOWNSCALE = 2
DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION = False

__all__ = [
    "DEFAULT_CMC_DOWNSCALE",
    "DEFAULT_CMC_METHOD",
    "DEFAULT_ENABLE_CMC",
    "DEFAULT_HIGH_CONF_DET_THRESHOLD",
    "DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION",
    "DEFAULT_LOST_TRACK_BUFFER",
    "DEFAULT_MINIMUM_CONSECUTIVE_FRAMES",
    "DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC",
    "DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC",
    "DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC",
    "DEFAULT_TRACK_ACTIVATION_THRESHOLD",
    "BoTSORTTrackerBlock",
    "CMCMethod",
    "host_bgr_frame",
]


class BoTSORTTrackerBlock(TrackersPackageBlock):
    """Track detections with one ``trackers.BoTSORTTracker`` per video source.

    BoT-SORT associates in three steps: confident detections (at least
    ``high_conf_det_threshold``) against confirmed and lost tracks, then
    low-confidence detections against the remaining tracks, then leftover
    confident detections against unconfirmed tracks. With ``enable_cmc`` it
    first shifts predicted tracks by the estimated camera motion. Whether a
    source uses CMC is fixed by its first frame, like every other knob.
    """

    type = "v2/trackers_botsort@v1"
    library_class = "BoTSORTTracker"

    class Params(TrackersPackageBlock.Params):
        minimum_iou_threshold_first_assoc: StrictFloat | Ref(FLOAT_KIND) = (
            fraction_knob(
                DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
                name="minimum_iou_threshold_first_assoc",
                description=(
                    "Minimum fused similarity (IoU x confidence) in the first, "
                    "high-confidence association step."
                ),
            )
        )
        minimum_iou_threshold_second_assoc: StrictFloat | Ref(FLOAT_KIND) = (
            fraction_knob(
                DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC,
                name="minimum_iou_threshold_second_assoc",
                description=(
                    "Minimum IoU in the second, low-confidence association step."
                ),
            )
        )
        minimum_iou_threshold_unconfirmed_assoc: StrictFloat | Ref(FLOAT_KIND) = (
            fraction_knob(
                DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC,
                name="minimum_iou_threshold_unconfirmed_assoc",
                description=(
                    "Minimum fused similarity matching unconfirmed tracks to "
                    "the remaining high-confidence detections."
                ),
            )
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
        high_conf_det_threshold: StrictFloat | Ref(FLOAT_KIND) = fraction_knob(
            DEFAULT_HIGH_CONF_DET_THRESHOLD,
            name="high_conf_det_threshold",
            description=(
                "Confidence splitting the high- from the low-confidence "
                "association steps."
            ),
        )
        enable_cmc: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=DEFAULT_ENABLE_CMC,
            description=(
                "Compensate camera motion from the image's pixels, for moving "
                "cameras. Copies each processed frame to the host and runs "
                f"OpenCV on the CPU. {FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_ENABLE_CMC, "$inputs.enable_cmc"],
        )
        cmc_method: CMCMethod = Field(
            default=DEFAULT_CMC_METHOD,
            description=(
                "Camera motion estimator: orb, sift, sparseOptFlow or ecc. "
                f"Literal only, as in V1. {FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_CMC_METHOD, "orb"],
        )
        cmc_downscale: StrictInt | Ref(INTEGER_KIND) = frame_count_knob(
            DEFAULT_CMC_DOWNSCALE,
            name="cmc_downscale",
            description=(
                "Integer factor the frame is shrunk by before camera motion "
                "estimation."
            ),
        )
        instant_first_frame_activation: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION,
            description=(
                "Report tracks of a source's first frame immediately, as in "
                "the BoT-SORT paper. False (V1 default) makes them wait "
                "minimum_consecutive_frames like other tracks. "
                f"{FIRST_FRAME_NOTE}"
            ),
            examples=[
                DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION,
                "$inputs.instant_first_frame_activation",
            ],
        )

    def run(
        self,
        *,
        image: ImageData,
        predictions: TrackedPrediction,
        minimum_iou_threshold_first_assoc: float,
        minimum_iou_threshold_second_assoc: float,
        minimum_iou_threshold_unconfirmed_assoc: float,
        minimum_consecutive_frames: int,
        lost_track_buffer: int,
        track_activation_threshold: float,
        high_conf_det_threshold: float,
        enable_cmc: bool,
        cmc_method: CMCMethod,
        cmc_downscale: int,
        instant_first_frame_activation: bool,
        max_sources: int,
        frame_rate: Optional[float] = None,
        instances_cache_size: int = DEFAULT_INSTANCES_CACHE_SIZE,
    ) -> Dict[str, Any]:
        """Track one frame of one source; see ``TrackersPackageBlock.track``.

        Args:
            image: Frame naming the source; its pixels feed CMC when enabled.
            predictions: Detections of this frame; not modified.
            minimum_iou_threshold_first_assoc: First-step fused similarity.
            minimum_iou_threshold_second_assoc: Second-step IoU.
            minimum_iou_threshold_unconfirmed_assoc: Unconfirmed-track step.
            minimum_consecutive_frames: Detected frames, counting the first,
                before a track is reported; misses do not restart the count.
            lost_track_buffer: Unmatched frames at 30 fps before deletion.
            track_activation_threshold: Confidence that starts a track.
            high_conf_det_threshold: High/low confidence split.
            enable_cmc: Whether a new source compensates camera motion.
            cmc_method: Camera motion estimator.
            cmc_downscale: Frame shrink factor for CMC.
            instant_first_frame_activation: Report first-frame tracks at once.
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
                minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
                minimum_iou_threshold_second_assoc=minimum_iou_threshold_second_assoc,
                minimum_iou_threshold_unconfirmed_assoc=(
                    minimum_iou_threshold_unconfirmed_assoc
                ),
                minimum_consecutive_frames=minimum_consecutive_frames,
                lost_track_buffer=lost_track_buffer,
                track_activation_threshold=track_activation_threshold,
                high_conf_det_threshold=high_conf_det_threshold,
                enable_cmc=enable_cmc,
                cmc_method=cmc_method,
                cmc_downscale=cmc_downscale,
                instant_first_frame_activation=instant_first_frame_activation,
            ),
        )

        return result

    def update_tracker(
        self, tracker: Any, boxes: sv.Detections, *, image: ImageData
    ) -> sv.Detections:
        """Advance a BoT-SORT tracker, giving it host pixels only for CMC.

        Args:
            tracker: This source's ``BoTSORTTracker``.
            boxes: Host boxes from ``host_boxes``.
            image: The frame the boxes refer to.

        Returns:
            The library's tracked detections.
        """
        # The source's own setting, fixed at its first frame.
        if not tracker.enable_cmc:
            tracked = tracker.update(boxes)
            return tracked

        frame = host_bgr_frame(image)
        tracked = tracker.update(boxes, frame=frame)

        return tracked


def host_bgr_frame(image: ImageData) -> np.ndarray:
    """Copy an image to the host in OpenCV's BGR ``uint8`` layout.

    CPU boundary for camera motion compensation: one device-to-host copy and
    one color conversion. A grayscale image is expanded to three channels.

    Args:
        image: RGB or grayscale image on any device; not modified.

    Returns:
        A new ``(height, width, 3)`` BGR ``uint8`` array.
    """
    hwc = image.tensor_image.detach().permute(1, 2, 0).cpu().numpy()
    if image.channels == 1:
        frame = cv2.cvtColor(np.ascontiguousarray(hwc[:, :, 0]), cv2.COLOR_GRAY2BGR)
        return frame

    frame = cv2.cvtColor(np.ascontiguousarray(hwc), cv2.COLOR_RGB2BGR)

    return frame
