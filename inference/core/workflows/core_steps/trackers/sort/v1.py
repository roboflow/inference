from typing import Any, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field
from trackers import SORTTracker

from inference.core.workflows.core_steps.trackers._base import (
    TrackerBlockBase,
    tracker_describe_outputs,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

DEFAULT_LOST_TRACK_BUFFER = 30
DEFAULT_TRACK_ACTIVATION_THRESHOLD = 0.25
DEFAULT_MINIMUM_CONSECUTIVE_FRAMES = 3
DEFAULT_MINIMUM_IOU_THRESHOLD = 0.3
DEFAULT_INSTANCES_CACHE_SIZE = 16384

SHORT_DESCRIPTION = (
    "Fast, lightweight object tracking. Works best when objects are clearly visible."
)
LONG_DESCRIPTION = """
Track objects across video frames using the **SORT** algorithm from the
roboflow/trackers package.

SORT pairs a Kalman filter motion model with single-stage IoU-based Hungarian
assignment.  It has the fewest parameters and lowest overhead, processing
hundreds of frames per second.  However, it lacks re-identification and
occlusion-recovery mechanisms, so tracks may fragment or switch IDs when objects
are temporarily hidden.

**When to use SORT:**
- Controlled environments with reliable, high-confidence detections.
- Real-time pipelines where maximum throughput is critical.
- Simple scenes with minimal occlusion and predictable linear motion.

**When to consider alternatives:**
- If you see fragmented tracks or missed weak detections, try **ByteTrack**.
- If objects undergo heavy occlusion or non-linear motion, try **OC-SORT**.

Outputs three detection sets:
- **tracked_detections**: All confirmed tracked detections with assigned track IDs.
- **new_instances**: Detections whose track ID appears for the first time.
- **already_seen_instances**: Detections whose track ID has been seen in a prior frame.

The block maintains separate tracker state and instance cache per `video_identifier`,
enabling multi-stream tracking within a single workflow.
"""


class SORTManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SORT Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-location-crosshairs",
                "blockPriority": 1,
                "trackers": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/trackers_sort@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Input image with embedded video metadata (fps and video_identifier). "
        "Used to initialise and retrieve per-video tracker state.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detection predictions for the current frame to track.",
        examples=["$steps.object_detection_model.predictions"],
    )
    minimum_iou_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=DEFAULT_MINIMUM_IOU_THRESHOLD,
        description="Minimum IoU required to associate a detection with an existing track. "
        f"Default: {DEFAULT_MINIMUM_IOU_THRESHOLD}.",
        examples=[DEFAULT_MINIMUM_IOU_THRESHOLD, "$inputs.minimum_iou_threshold"],
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


class SORTBlockV1(TrackerBlockBase):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SORTManifest

    def _create_tracker(self, fps: int, **kwargs: Any) -> Any:
        return SORTTracker(
            lost_track_buffer=kwargs["lost_track_buffer"],
            frame_rate=fps,
            track_activation_threshold=kwargs["track_activation_threshold"],
            minimum_consecutive_frames=kwargs["minimum_consecutive_frames"],
            minimum_iou_threshold=kwargs["minimum_iou_threshold"],
        )

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        lost_track_buffer: int = DEFAULT_LOST_TRACK_BUFFER,
        minimum_iou_threshold: float = DEFAULT_MINIMUM_IOU_THRESHOLD,
        minimum_consecutive_frames: int = DEFAULT_MINIMUM_CONSECUTIVE_FRAMES,
        instances_cache_size: int = DEFAULT_INSTANCES_CACHE_SIZE,
        track_activation_threshold: float = DEFAULT_TRACK_ACTIVATION_THRESHOLD,
    ) -> BlockResult:
        return self._run_tracker(
            image=image,
            detections=detections,
            instances_cache_size=instances_cache_size,
            lost_track_buffer=lost_track_buffer,
            minimum_iou_threshold=minimum_iou_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            track_activation_threshold=track_activation_threshold,
        )
