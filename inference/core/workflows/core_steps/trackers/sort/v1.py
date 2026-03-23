import inspect
from typing import Any, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field
from trackers import SORTTracker

from inference.core.workflows.core_steps.trackers._base import (
    DEFAULT_INSTANCES_CACHE_SIZE,
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

_TRACKER_DEFAULTS = {
    p.name: p.default
    for p in inspect.signature(SORTTracker.__init__).parameters.values()
    if p.default is not inspect.Parameter.empty
}
DEFAULT_MINIMUM_IOU_THRESHOLD = _TRACKER_DEFAULTS["minimum_iou_threshold"]
DEFAULT_MINIMUM_CONSECUTIVE_FRAMES = _TRACKER_DEFAULTS["minimum_consecutive_frames"]
DEFAULT_LOST_TRACK_BUFFER = _TRACKER_DEFAULTS["lost_track_buffer"]
DEFAULT_TRACK_ACTIVATION_THRESHOLD = _TRACKER_DEFAULTS["track_activation_threshold"]

SHORT_DESCRIPTION = "Track objects across video frames using SORT."
LONG_DESCRIPTION = """
Track objects across video frames using the **SORT** algorithm from the
roboflow/trackers package.

SORT is a lightweight IoU-based Hungarian assignment tracker with Kalman filtering.
It is fast and well suited to controlled environments with reliable detections.

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
