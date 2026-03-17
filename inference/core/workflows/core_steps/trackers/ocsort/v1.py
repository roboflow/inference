from typing import Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field
from trackers import OCSORTTracker

from inference.core import logger
from inference.core.workflows.core_steps.trackers._utils import InstanceCache
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
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "tracked_detections"
SHORT_DESCRIPTION = (
    "Track and update object positions across video frames using OC-SORT."
)
LONG_DESCRIPTION = """
Track objects across video frames using the OC-SORT (Observation-Centric SORT) algorithm
from the roboflow/trackers package.

OC-SORT enhances SORT by introducing an observation-centric paradigm: detections are used
to correct Kalman filter errors accumulated during occlusions. It adds Observation-Centric
Re-Update (ORU) to generate virtual trajectories on track reactivation, and Observation-Centric
Momentum (OCM) to blend IoU with direction consistency for more robust association in crowded
and non-linear-motion scenes.

Outputs three detection sets:
- **tracked_detections**: All confirmed tracked detections with assigned track IDs.
- **new_instances**: Detections whose track ID appears for the first time (first appearance).
- **already_seen_instances**: Detections whose track ID has been seen in a prior frame.

The block maintains separate tracker state and instance cache per `video_identifier`, enabling
multi-stream tracking within a single workflow.
"""


class OCSORTTrackerBlockManifest(WorkflowBlockManifest):
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
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/ocsort@v1"]
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
    high_conf_det_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.6,
        description="Confidence threshold for high-confidence detections. Detections below "
        "this threshold are excluded from association entirely. Default: 0.6.",
        examples=[0.6, "$inputs.high_conf_det_threshold"],
    )
    lost_track_buffer: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="Number of frames to keep a track alive after it loses its matched "
        "detection. Higher values improve occlusion recovery. Default: 30.",
        examples=[30, "$inputs.lost_track_buffer"],
    )
    minimum_iou_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.3,
        description="Minimum IoU required to associate a detection with an existing track. "
        "Default: 0.3.",
        examples=[0.3, "$inputs.minimum_iou_threshold"],
    )
    minimum_consecutive_frames: Union[
        Optional[int], Selector(kind=[INTEGER_KIND])
    ] = Field(
        default=3,
        description="Number of consecutive frames a track must be matched before it is "
        "emitted as a confirmed track (tracker_id != -1). Default: 3.",
        examples=[3, "$inputs.minimum_consecutive_frames"],
    )
    direction_consistency_weight: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.2,
        description="Weight for the direction consistency term in the association cost. "
        "Higher values prioritise alignment between historical motion direction and the "
        "direction to the candidate detection. Default: 0.2.",
        examples=[0.2, "$inputs.direction_consistency_weight"],
    )
    delta_t: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=3,
        description="Number of past frames used to estimate per-track velocity for the "
        "direction consistency momentum. Higher values give more stable estimates. Default: 3.",
        examples=[3, "$inputs.delta_t"],
    )
    instances_cache_size: int = Field(
        default=16384,
        description="Maximum number of track IDs retained in the instance cache for "
        "new/already-seen categorisation. Uses FIFO eviction. Default: 16384.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
            OutputDefinition(
                name="new_instances", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name="already_seen_instances",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OCSORTTrackerBlockV1(WorkflowBlock):
    def __init__(self) -> None:
        self._trackers: Dict[str, OCSORTTracker] = {}
        self._per_video_cache: Dict[str, InstanceCache] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OCSORTTrackerBlockManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        high_conf_det_threshold: float = 0.6,
        lost_track_buffer: int = 30,
        minimum_iou_threshold: float = 0.3,
        minimum_consecutive_frames: int = 3,
        direction_consistency_weight: float = 0.2,
        delta_t: int = 3,
        instances_cache_size: int = 16384,
    ) -> BlockResult:
        metadata = image.video_metadata
        fps = metadata.fps
        if not fps:
            fps = 0
            logger.warning(
                f"Malformed fps in VideoMetadata, {self.__class__.__name__} requires "
                "fps in order to initialise OCSORTTracker"
            )
        video_id = metadata.video_identifier
        if video_id not in self._trackers:
            self._trackers[video_id] = OCSORTTracker(
                lost_track_buffer=lost_track_buffer,
                frame_rate=fps,
                minimum_consecutive_frames=minimum_consecutive_frames,
                minimum_iou_threshold=minimum_iou_threshold,
                direction_consistency_weight=direction_consistency_weight,
                high_conf_det_threshold=high_conf_det_threshold,
                delta_t=delta_t,
            )
        tracker = self._trackers[video_id]
        merged = sv.Detections.merge(detections[i] for i in range(len(detections)))
        tracked_detections = tracker.update(merged)

        # Filter out immature / unmatched tracks (tracker_id == -1)
        if tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
            valid_mask = tracked_detections.tracker_id != -1
            tracked_detections = tracked_detections[valid_mask]

        if video_id not in self._per_video_cache:
            self._per_video_cache[video_id] = InstanceCache(size=instances_cache_size)
        cache = self._per_video_cache[video_id]

        not_seen_mask, seen_mask = [], []
        for tracker_id in tracked_detections.tracker_id.tolist():
            already_seen = cache.record_instance(tracker_id=tracker_id)
            not_seen_mask.append(not already_seen)
            seen_mask.append(already_seen)

        return {
            OUTPUT_KEY: tracked_detections,
            "new_instances": tracked_detections[not_seen_mask],
            "already_seen_instances": tracked_detections[seen_mask],
        }
