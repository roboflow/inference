from typing import Any, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field
from trackers import OCSORTTracker

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

SHORT_DESCRIPTION = "Track objects across video frames using OC-SORT."
LONG_DESCRIPTION = """
Track objects across video frames using the **OC-SORT** algorithm from the
roboflow/trackers package.

OC-SORT is an observation-centric variant of SORT that corrects Kalman drift during
occlusions and adds direction-consistency momentum for non-linear motion.

Outputs three detection sets:
- **tracked_detections**: All confirmed tracked detections with assigned track IDs.
- **new_instances**: Detections whose track ID appears for the first time.
- **already_seen_instances**: Detections whose track ID has been seen in a prior frame.

The block maintains separate tracker state and instance cache per `video_identifier`,
enabling multi-stream tracking within a single workflow.
"""


class OCSORTManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OC-SORT Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-location-crosshairs",
                "blockPriority": 2,
                "trackers": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/trackers_ocsort@v1"]
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
        default=0.3,
        description="Minimum IoU required to associate a detection with an existing track. "
        "Default: 0.3.",
        examples=[0.3, "$inputs.minimum_iou_threshold"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    minimum_consecutive_frames: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = (
        Field(
            default=3,
            description="Number of consecutive frames a track must be matched before it is "
            "emitted as a confirmed track (tracker_id != -1). Default: 3.",
            examples=[3, "$inputs.minimum_consecutive_frames"],
            json_schema_extra={
                "always_visible": True,
            },
        )
    )
    lost_track_buffer: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="Number of frames to keep a track alive after it loses its matched "
        "detection. Higher values improve occlusion recovery. Default: 30.",
        examples=[30, "$inputs.lost_track_buffer"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    high_conf_det_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.6,
        description="Confidence threshold for high-confidence detections used in "
        "association. Default: 0.6.",
        examples=[0.6, "$inputs.high_conf_det_threshold"],
    )
    direction_consistency_weight: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.2,
        description="Weight for the direction consistency term in the OC-SORT association "
        "cost. Higher values prioritise alignment between historical motion direction and "
        "the direction to the candidate detection. Default: 0.2.",
        examples=[0.2, "$inputs.direction_consistency_weight"],
    )
    delta_t: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=3,
        description="Number of past frames used by OC-SORT to estimate per-track velocity "
        "for direction consistency momentum. Default: 3.",
        examples=[3, "$inputs.delta_t"],
    )
    instances_cache_size: int = Field(
        default=16384,
        description="Maximum number of track IDs retained in the instance cache for "
        "new/already-seen categorisation. Uses FIFO eviction. Default: 16384.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return tracker_describe_outputs()

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OCSORTBlockV1(TrackerBlockBase):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OCSORTManifest

    def _create_tracker(self, fps: int, **kwargs: Any) -> Any:
        return OCSORTTracker(
            lost_track_buffer=kwargs.get("lost_track_buffer", 30),
            frame_rate=fps,
            minimum_consecutive_frames=kwargs.get("minimum_consecutive_frames", 3),
            minimum_iou_threshold=kwargs.get("minimum_iou_threshold", 0.3),
            high_conf_det_threshold=kwargs.get("high_conf_det_threshold", 0.6),
            direction_consistency_weight=kwargs.get(
                "direction_consistency_weight", 0.2
            ),
            delta_t=kwargs.get("delta_t", 3),
        )

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        lost_track_buffer: int = 30,
        minimum_iou_threshold: float = 0.3,
        minimum_consecutive_frames: int = 3,
        instances_cache_size: int = 16384,
        high_conf_det_threshold: float = 0.6,
        direction_consistency_weight: float = 0.2,
        delta_t: int = 3,
    ) -> BlockResult:
        return self._run_tracker(
            image=image,
            detections=detections,
            instances_cache_size=instances_cache_size,
            lost_track_buffer=lost_track_buffer,
            minimum_iou_threshold=minimum_iou_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            high_conf_det_threshold=high_conf_det_threshold,
            direction_consistency_weight=direction_consistency_weight,
            delta_t=delta_t,
        )
