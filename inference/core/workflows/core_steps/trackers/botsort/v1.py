from typing import Any, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field
from trackers import BoTSORTTracker

from inference.core import logger
from inference.core.workflows.core_steps.trackers._base import (
    TRACKER_PREDICTION_KINDS,
    TrackerBlockBase,
    tracker_describe_outputs,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

#: Camera motion compensation (CMC) backend for BoT-SORT. Valid string values:
#:
#: - ``orb``: ORB keypoints/descriptors with RANSAC affine estimation.
#: - ``sift``: SIFT keypoints/descriptors with RANSAC affine estimation (typically slower than ORB).
#: - ``sparseOptFlow``: ``goodFeaturesToTrack`` + Lucas-Kanade optical flow with RANSAC affine estimation.
#: - ``ecc``: Enhanced Correlation Coefficient alignment on image intensities.
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
DEFAULT_INSTANCES_CACHE_SIZE = 16384

SHORT_DESCRIPTION = (
    "ByteTrack-style association with optional camera motion compensation (BoT-SORT)."
)
LONG_DESCRIPTION = """
Track objects across video frames using the **BoT-SORT** algorithm from the
roboflow/trackers package.

BoT-SORT follows a ByteTrack-style association pipeline (high- and low-confidence
detections, Kalman track states) and can apply **camera motion compensation (CMC)**
before association when enabled. CMC estimates a global affine motion between
frames so predicted boxes align better when the camera moves.

**When to use BoT-SORT:**
- Scenes with **moving or shaking cameras** (enable **Camera motion compensation**).
- Dense detection noise where ByteTrack-style two-stage matching helps.
- When you want ByteTrack-like behaviour with an optional motion-compensation stage.

**When to consider alternatives:**
- Fixed camera and you only need speed: **ByteTrack** or **SORT** may be simpler.
- Heavy occlusion and erratic object motion without camera motion: **OC-SORT**.
- Low-texture backgrounds where sparse-feature CMC is unreliable.

**Camera motion compensation:** When enabled, the block passes the workflow image
pixels to the tracker each frame. If the image cannot be decoded to a numpy array,
the tracker runs without CMC for that frame (a warning is logged).

**Instant first-frame activation** defaults to off so behaviour aligns with other
core tracker blocks for ``new_instances`` / ``already_seen_instances``. Enable it
if you want tracks on frame 1 to receive stable IDs immediately (original BoT-SORT
paper-style).

Outputs three detection sets:
- **tracked_detections**: All confirmed tracked detections with assigned track IDs.
- **new_instances**: Detections whose track ID appears for the first time.
- **already_seen_instances**: Detections whose track ID has been seen in a prior frame.

The block maintains separate tracker state and instance cache per `video_identifier`,
enabling multi-stream tracking within a single workflow.
"""


class BoTSORTManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "BoT-SORT Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-location-crosshairs",
                "blockPriority": 0,
                "trackers": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/trackers_botsort@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Input image with embedded video metadata (fps and video_identifier). "
        "Used to initialise and retrieve per-video tracker state. When camera motion "
        "compensation is enabled, frame pixels are read from this image.",
    )
    detections: Selector(
        kind=TRACKER_PREDICTION_KINDS,
    ) = Field(
        description="Detection predictions for the current frame to track.",
        examples=["$steps.object_detection_model.predictions"],
    )
    minimum_iou_threshold_first_assoc: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
        description="Minimum fused similarity (IoU × confidence) for the first "
        f"(high-confidence) association step. Default: {DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC}.",
        examples=[
            DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
            "$inputs.minimum_iou_threshold_first_assoc",
        ],
        json_schema_extra={
            "always_visible": True,
        },
    )
    minimum_iou_threshold_second_assoc: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC,
        description="Minimum IoU for the second (low-confidence) association step. "
        f"Default: {DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC}.",
        examples=[
            DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC,
            "$inputs.minimum_iou_threshold_second_assoc",
        ],
        json_schema_extra={
            "always_visible": True,
        },
    )
    minimum_iou_threshold_unconfirmed_assoc: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC,
        description="Minimum fused similarity for matching unconfirmed tracks to "
        f"remaining high-confidence detections. Default: {DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC}.",
        examples=[
            DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC,
            "$inputs.minimum_iou_threshold_unconfirmed_assoc",
        ],
        json_schema_extra={
            "always_visible": True,
        },
    )
    minimum_consecutive_frames: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = (
        Field(
            default=DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
            description="Number of consecutive frames a track must be matched before it is "
            f"emitted as a confirmed track (tracker_id != -1). Default: {DEFAULT_MINIMUM_CONSECUTIVE_FRAMES}.",
            examples=[
                DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
                "$inputs.minimum_consecutive_frames",
            ],
            json_schema_extra={
                "always_visible": True,
            },
        )
    )
    lost_track_buffer: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=DEFAULT_LOST_TRACK_BUFFER,
        description="Number of frames to keep a track alive after it loses its matched "
        f"detection. Higher values improve occlusion recovery. Default: {DEFAULT_LOST_TRACK_BUFFER}.",
        examples=[DEFAULT_LOST_TRACK_BUFFER, "$inputs.lost_track_buffer"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    track_activation_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=DEFAULT_TRACK_ACTIVATION_THRESHOLD,
        description="Minimum detection confidence required to spawn a new track. "
        f"Detections below this threshold are not used to create new tracks. Default: {DEFAULT_TRACK_ACTIVATION_THRESHOLD}.",
        examples=[
            DEFAULT_TRACK_ACTIVATION_THRESHOLD,
            "$inputs.track_activation_threshold",
        ],
    )
    high_conf_det_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=DEFAULT_HIGH_CONF_DET_THRESHOLD,
        description="Confidence threshold for high-confidence detections used in "
        f"association. Default: {DEFAULT_HIGH_CONF_DET_THRESHOLD}.",
        examples=[DEFAULT_HIGH_CONF_DET_THRESHOLD, "$inputs.high_conf_det_threshold"],
    )
    enable_cmc: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore[arg-type]
        default=DEFAULT_ENABLE_CMC,
        description="Enable camera motion compensation (uses per-frame image pixels). "
        "Recommended for moving cameras.",
        examples=[DEFAULT_ENABLE_CMC, "$inputs.enable_cmc"],
    )
    cmc_method: CMCMethod = Field(
        default=DEFAULT_CMC_METHOD,
        description="Camera motion estimator. One of: orb, "
        "sift, sparseOptFlow, ecc. Default: {DEFAULT_CMC_METHOD!r}.",
        examples=[DEFAULT_CMC_METHOD, "$inputs.cmc_method"],
    )
    cmc_downscale: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=DEFAULT_CMC_DOWNSCALE,
        description="Downscale factor applied inside CMC for speed and robustness. "
        f"Default: {DEFAULT_CMC_DOWNSCALE}.",
        examples=[DEFAULT_CMC_DOWNSCALE, "$inputs.cmc_downscale"],
    )
    instant_first_frame_activation: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore[arg-type]
        default=DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION,
        description="If true, tracks on the first frame receive IDs immediately (paper-style). "
        "Default false so new/already-seen outputs match other core trackers.",
        examples=[
            DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION,
            "$inputs.instant_first_frame_activation",
        ],
    )
    instances_cache_size: int = Field(
        default=DEFAULT_INSTANCES_CACHE_SIZE,
        description="Maximum number of track IDs retained in the instance cache for "
        f"new/already-seen categorisation. Uses FIFO eviction. Default: {DEFAULT_INSTANCES_CACHE_SIZE}.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return tracker_describe_outputs()

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BoTSORTBlockV1(TrackerBlockBase):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoTSORTManifest

    def _create_tracker(self, fps: int, **kwargs: Any) -> Any:
        return BoTSORTTracker(
            lost_track_buffer=kwargs["lost_track_buffer"],
            frame_rate=fps,
            track_activation_threshold=kwargs["track_activation_threshold"],
            minimum_consecutive_frames=kwargs["minimum_consecutive_frames"],
            minimum_iou_threshold_first_assoc=kwargs[
                "minimum_iou_threshold_first_assoc"
            ],
            minimum_iou_threshold_second_assoc=kwargs[
                "minimum_iou_threshold_second_assoc"
            ],
            minimum_iou_threshold_unconfirmed_assoc=kwargs[
                "minimum_iou_threshold_unconfirmed_assoc"
            ],
            high_conf_det_threshold=kwargs["high_conf_det_threshold"],
            enable_cmc=kwargs["enable_cmc"],
            cmc_method=kwargs["cmc_method"],
            cmc_downscale=kwargs["cmc_downscale"],
            instant_first_frame_activation=kwargs["instant_first_frame_activation"],
        )

    def _tracker_update(
        self,
        tracker: Any,
        detections: sv.Detections,
        image: WorkflowImageData,
    ) -> sv.Detections:
        if not getattr(tracker, "enable_cmc", False):
            return tracker.update(detections)
        try:
            frame = image.numpy_image
        except Exception as exc:
            logger.warning(
                "%s: enable_cmc=True but frame unavailable (%s); running without CMC.",
                self.__class__.__name__,
                exc,
            )
            return tracker.update(detections)
        return tracker.update(detections, frame=frame)

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        lost_track_buffer: int = DEFAULT_LOST_TRACK_BUFFER,
        minimum_iou_threshold_first_assoc: float = DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
        minimum_iou_threshold_second_assoc: float = DEFAULT_MINIMUM_IOU_THRESHOLD_SECOND_ASSOC,
        minimum_iou_threshold_unconfirmed_assoc: float = (
            DEFAULT_MINIMUM_IOU_THRESHOLD_UNCONFIRMED_ASSOC
        ),
        minimum_consecutive_frames: int = DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
        instances_cache_size: int = DEFAULT_INSTANCES_CACHE_SIZE,
        track_activation_threshold: float = DEFAULT_TRACK_ACTIVATION_THRESHOLD,
        high_conf_det_threshold: float = DEFAULT_HIGH_CONF_DET_THRESHOLD,
        enable_cmc: bool = DEFAULT_ENABLE_CMC,
        cmc_method: CMCMethod = DEFAULT_CMC_METHOD,
        cmc_downscale: int = DEFAULT_CMC_DOWNSCALE,
        instant_first_frame_activation: bool = DEFAULT_INSTANT_FIRST_FRAME_ACTIVATION,
    ) -> BlockResult:
        return self._run_tracker(
            image=image,
            detections=detections,
            instances_cache_size=instances_cache_size,
            lost_track_buffer=lost_track_buffer,
            minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=minimum_iou_threshold_unconfirmed_assoc,
            minimum_consecutive_frames=minimum_consecutive_frames,
            track_activation_threshold=track_activation_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
            enable_cmc=enable_cmc,
            cmc_method=cmc_method,
            cmc_downscale=cmc_downscale,
            instant_first_frame_activation=instant_first_frame_activation,
        )
