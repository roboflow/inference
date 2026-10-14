"""BoT-SORT from the ``trackers`` package, with one tracker per video source.

This is the V1 ``roboflow_core/trackers_botsort@v1`` capability. The V1
``detections`` input is ``predictions`` here. Unlike V1, a source without a
finite positive ``video_metadata.fps`` needs ``frame_rate``; V1 assumed 30.
BoT-SORT uses no appearance (ReID) model here, as in V1.

Shared source, cache, capacity and frame-rate policies live in ``base``::

    run(knobs)
      ├─ host_boxes(predictions)           boxes to host, once
      ├─ SourceTrackers.state_for(image)   first frame: tracker from knobs + rate
      ├─ tracker.update(boxes[, frame])    library call (CPU)
      ├─ kept_rows + InstanceCache.split   state advances here
      └─ tracking_result(wanted outputs)   no state change

The library tracker runs on the host CPU over a box-only ``sv.Detections``.
Masks, keypoints and row metadata stay on the input's device and are selected,
not copied to the host. ``trackers`` is imported when the block is created, so
the catalogue imports without it.

Camera motion compensation (CMC) is the one place a tracker block reads
pixels. With ``enable_cmc`` the frame is copied to the host as a BGR ``uint8``
array once per processed frame, and OpenCV estimates the camera motion on the
CPU::

    enable_cmc false:  boxes ──► tracker.update(boxes)
    enable_cmc true:   boxes ─┬─► tracker.update(boxes, frame=bgr)
                       image ─┴─► host BGR copy (CPU boundary)

Without CMC no pixel leaves the image's device.

Tracker ids come from a counter that the ``trackers`` package shares between
all BoT-SORT trackers in the process, as in V1. Ids are therefore unique
across sources and blocks, but a source's first id is not necessarily 0. The
block never calls the library's ``reset``, which would rewind that shared
counter for every source.
"""

from typing import Any, Dict, Literal, Optional
import cv2
import numpy as np
from pydantic import Field, StrictBool, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.tracking.base import (
    DEFAULT_INSTANCES_CACHE_SIZE,
    DEFAULT_MAX_SOURCES,
    TRACKED_KINDS,
    TRACKER_OUTPUTS,
    SourceTrackers,
    TrackedPrediction,
    host_boxes,
    kept_rows,
    tracking_result,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine import NoExecutionContextError, get_execution_context
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref


CMCMethod = Literal["orb", "sift", "sparseOptFlow", "ecc"]

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

_FIRST_FRAME_NOTE = (
    "Read once per source, at its first frame; a later value, selected or "
    "edited, does not retune it until a processing reset."
)

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


class BoTSORTTrackerBlock(Block):
    """Track detections with one ``trackers.BoTSORTTracker`` per video source.

    BoT-SORT associates in three steps: confident detections (at least
    ``high_conf_det_threshold``) against confirmed and lost tracks, then
    low-confidence detections against the remaining tracks, then leftover
    confident detections against unconfirmed tracks. With ``enable_cmc`` it
    first shifts predicted tracks by the estimated camera motion. Whether a
    source uses CMC is fixed by its first frame, like every other knob.
    """

    type = "roboflow_core/trackers_botsort@v1"
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
                "does not report a finite positive fps (e.g. live RTSP); V1 "
                "assumed 30. Not the processing rate; dropped frames do not "
                "age tracks."
            ),
            examples=[None, 30.0],
        )
        minimum_iou_threshold_first_assoc: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
            ge=0,
            le=1,
            description=(
                "Minimum fused similarity (IoU x confidence) in the first, "
                f"high-confidence association step. {_FIRST_FRAME_NOTE}"
            ),
            examples=[
                DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
                "$inputs.minimum_iou_threshold_first_assoc",
            ],
        )
        minimum_iou_threshold_second_assoc: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC,
            ge=0,
            le=1,
            description=(
                "Minimum IoU in the second, low-confidence association step. "
                f"{_FIRST_FRAME_NOTE}"
            ),
            examples=[
                DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC,
                "$inputs.minimum_iou_threshold_second_assoc",
            ],
        )
        minimum_iou_threshold_unconfirmed_assoc: StrictFloat | Ref(FLOAT_KIND) = (
            Field(
                default=DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC,
                ge=0,
                le=1,
                description=(
                    "Minimum fused similarity matching unconfirmed tracks to "
                    "the remaining high-confidence detections. "
                    f"{_FIRST_FRAME_NOTE}"
                ),
                examples=[
                    DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC,
                    "$inputs.minimum_iou_threshold_unconfirmed_assoc",
                ],
            )
        )
        minimum_consecutive_frames: StrictInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
            ge=1,
            description=(
                "Frames a track must be detected in, counting its first, before "
                "it is reported; a missed frame does not restart the count. "
                f"{_FIRST_FRAME_NOTE}"
            ),
            examples=[
                DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
                "$inputs.minimum_consecutive_frames",
            ],
        )
        lost_track_buffer: StrictInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_LOST_TRACK_BUFFER,
            ge=1,
            description=(
                "Frames at 30 fps a track survives unmatched; scaled by the "
                "frame rate. Only processed frames advance the tracker. "
                f"{_FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_LOST_TRACK_BUFFER, "$inputs.lost_track_buffer"],
        )
        track_activation_threshold: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=DEFAULT_TRACK_ACTIVATION_THRESHOLD,
            ge=0,
            le=1,
            description=(
                f"Minimum confidence that starts a new track. {_FIRST_FRAME_NOTE}"
            ),
            examples=[
                DEFAULT_TRACK_ACTIVATION_THRESHOLD,
                "$inputs.track_activation_threshold",
            ],
        )
        high_conf_det_threshold: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=DEFAULT_HIGH_CONF_DET_THRESHOLD,
            ge=0,
            le=1,
            description=(
                "Confidence splitting the high- from the low-confidence "
                f"association steps. {_FIRST_FRAME_NOTE}"
            ),
            examples=[
                DEFAULT_HIGH_CONF_DET_THRESHOLD,
                "$inputs.high_conf_det_threshold",
            ],
        )
        enable_cmc: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=DEFAULT_ENABLE_CMC,
            description=(
                "Compensate camera motion from the image's pixels, for moving "
                "cameras. Copies each processed frame to the host and runs "
                f"OpenCV on the CPU. {_FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_ENABLE_CMC, "$inputs.enable_cmc"],
        )
        cmc_method: CMCMethod = Field(
            default=DEFAULT_CMC_METHOD,
            description=(
                "Camera motion estimator: orb, sift, sparseOptFlow or ecc. "
                f"Literal only, as in V1. {_FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_CMC_METHOD, "orb"],
        )
        cmc_downscale: StrictInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_CMC_DOWNSCALE,
            ge=1,
            description=(
                "Integer factor the frame is shrunk by before camera motion "
                f"estimation. {_FIRST_FRAME_NOTE}"
            ),
            examples=[DEFAULT_CMC_DOWNSCALE, "$inputs.cmc_downscale"],
        )
        instant_first_frame_activation: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION,
            description=(
                "Report tracks of a source's first frame immediately, as in "
                "the BoT-SORT paper. False (V1 default) makes them wait "
                "minimum_consecutive_frames like other tracks. "
                f"{_FIRST_FRAME_NOTE}"
            ),
            examples=[
                DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION,
                "$inputs.instant_first_frame_activation",
            ],
        )

    def __init__(self):
        # Imported here, not at module level, so the catalogue imports
        # without the optional ``trackers`` package.
        try:
            import trackers
        except ImportError as error:
            raise ImportError(
                f"{self.type} needs the 'trackers' package, a roboflow-workflows "
                "dependency: pip install 'trackers~=2.4.0'"
            ) from error

        self._tracker_class = trackers.BoTSORTTracker
        self._trackers = SourceTrackers(block_type=self.type)

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
        """Update this source's tracker and instance cache with one frame.

        Both updates happen before outputs are selected, so state advances
        even when no output is demanded. Every output has the input's family
        and image metadata: rows the tracker kept, in tracker order, each with
        its ``tracker_id`` in ``bboxes_metadata``. Rows the library reports
        with id ``-1`` (not yet confirmed) are dropped.

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
            frame_rate: Nominal fps for a new source's tracker; None uses
                ``video_metadata.fps``.
            instances_cache_size: Tracker ids a new source's cache keeps.

        Returns:
            Demanded outputs among ``tracked_detections``, ``new_instances``
            and ``already_seen_instances``. Direct calls return all three.

        Notes:
            The eleven BoT-SORT values, ``frame_rate`` and
            ``instances_cache_size`` are read only when a source's state is
            created.

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

        def create_tracker(rate: float) -> Any:
            tracker = self._tracker_class(
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

        # The source's own CMC setting, fixed at its first frame; this call's
        # enable_cmc may differ. Without CMC no pixel is copied to the host.
        if state.tracker.enable_cmc:
            frame = host_bgr_frame(image)
            tracked = state.tracker.update(boxes, frame=frame)
        else:
            tracked = state.tracker.update(boxes)

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
