from typing import Any, Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field
from trackers import ByteTrackTracker, OCSORTTracker, SORTTracker

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
SHORT_DESCRIPTION = "Track and update object positions across video frames."
LONG_DESCRIPTION = """
Track objects across video frames using a selectable tracking algorithm from the
roboflow/trackers package.

Select the tracking algorithm via the **tracker_type** dropdown:

- **bytetrack**: ByteTrack — two-stage association using high- and low-confidence
  detection pools. Robust in crowded scenes and during occlusions.
- **sort**: SORT — lightweight IoU-based Hungarian assignment with Kalman filtering.
  Fast and well suited to controlled environments with reliable detections.
- **ocsort**: OC-SORT — observation-centric variant of SORT that corrects Kalman
  drift during occlusions and adds direction-consistency momentum for non-linear motion.

Outputs three detection sets:
- **tracked_detections**: All confirmed tracked detections with assigned track IDs.
- **new_instances**: Detections whose track ID appears for the first time.
- **already_seen_instances**: Detections whose track ID has been seen in a prior frame.

The block maintains separate tracker state and instance cache per `video_identifier`,
enabling multi-stream tracking within a single workflow.
"""


class TrackerManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-location-crosshairs",
                "blockPriority": 0,
                "subtitle_field": "tracker_type",
                "trackers": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/trackers@v1"]
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
    tracker_type: Literal["bytetrack", "sort", "ocsort"] = Field(
        default="bytetrack",
        description="Tracking algorithm to use. 'bytetrack' for two-stage association "
        "(robust in crowds), 'sort' for lightweight IoU-based tracking, 'ocsort' for "
        "observation-centric SORT with improved occlusion handling.",
        json_schema_extra={
            "always_visible": True,
        },
    )
    minimum_iou_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.1,
        description="Minimum IoU required to associate a detection with an existing track. "
        "Default: 0.1.",
        examples=[0.1, "$inputs.minimum_iou_threshold"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    minimum_consecutive_frames: Union[
        Optional[int], Selector(kind=[INTEGER_KIND])
    ] = Field(
        default=2,
        description="Number of consecutive frames a track must be matched before it is "
        "emitted as a confirmed track (tracker_id != -1). Default: 2.",
        examples=[2, "$inputs.minimum_consecutive_frames"],
        json_schema_extra={
            "always_visible": True,
        },
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
    track_activation_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.7,
        description="Minimum detection confidence required to spawn a new track. "
        "Detections below this threshold are not used to create new tracks. Default: 0.7.",
        examples=[0.7, "$inputs.track_activation_threshold"],
        json_schema_extra={
            "relevant_for": {
                "tracker_type": {
                    "values": ["bytetrack", "sort"],
                    "required": False,
                },
            },
        },
    )
    high_conf_det_threshold: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.6,
        description="Confidence threshold for high-confidence detections used in "
        "association. Default: 0.6.",
        examples=[0.6, "$inputs.high_conf_det_threshold"],
        json_schema_extra={
            "relevant_for": {
                "tracker_type": {
                    "values": ["bytetrack", "ocsort"],
                    "required": False,
                },
            },
        },
    )
    direction_consistency_weight: Union[
        Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.2,
        description="Weight for the direction consistency term in the OC-SORT association "
        "cost. Higher values prioritise alignment between historical motion direction and "
        "the direction to the candidate detection. Default: 0.2.",
        examples=[0.2, "$inputs.direction_consistency_weight"],
        json_schema_extra={
            "relevant_for": {
                "tracker_type": {
                    "values": ["ocsort"],
                    "required": False,
                },
            },
        },
    )
    delta_t: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=3,
        description="Number of past frames used by OC-SORT to estimate per-track velocity "
        "for direction consistency momentum. Default: 3.",
        examples=[3, "$inputs.delta_t"],
        json_schema_extra={
            "relevant_for": {
                "tracker_type": {
                    "values": ["ocsort"],
                    "required": False,
                },
            },
        },
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


class TrackerBlockV1(WorkflowBlock):
    def __init__(self) -> None:
        # Keyed by f"{video_id}::{tracker_type}" for independent state per algorithm
        self._trackers: Dict[str, Any] = {}
        self._per_video_cache: Dict[str, InstanceCache] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TrackerManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        tracker_type: str = "bytetrack",
        lost_track_buffer: int = 30,
        minimum_iou_threshold: float = 0.1,
        minimum_consecutive_frames: int = 2,
        instances_cache_size: int = 16384,
        track_activation_threshold: float = 0.7,
        high_conf_det_threshold: float = 0.6,
        direction_consistency_weight: float = 0.2,
        delta_t: int = 3,
    ) -> BlockResult:
        metadata = image.video_metadata
        fps = metadata.fps
        if not fps:
            fps = 30
            logger.warning(
                f"fps not available in VideoMetadata for {self.__class__.__name__}, "
                "defaulting to 30 fps for tracker initialisation"
            )
        video_id = metadata.video_identifier
        key = f"{video_id}::{tracker_type}"

        if key not in self._trackers:
            if tracker_type == "bytetrack":
                self._trackers[key] = ByteTrackTracker(
                    lost_track_buffer=lost_track_buffer,
                    frame_rate=fps,
                    track_activation_threshold=track_activation_threshold,
                    minimum_consecutive_frames=minimum_consecutive_frames,
                    minimum_iou_threshold=minimum_iou_threshold,
                    high_conf_det_threshold=high_conf_det_threshold,
                )
            elif tracker_type == "sort":
                self._trackers[key] = SORTTracker(
                    lost_track_buffer=lost_track_buffer,
                    frame_rate=fps,
                    track_activation_threshold=track_activation_threshold,
                    minimum_consecutive_frames=minimum_consecutive_frames,
                    minimum_iou_threshold=minimum_iou_threshold,
                )
            elif tracker_type == "ocsort":
                self._trackers[key] = OCSORTTracker(
                    lost_track_buffer=lost_track_buffer,
                    frame_rate=fps,
                    minimum_consecutive_frames=minimum_consecutive_frames,
                    minimum_iou_threshold=minimum_iou_threshold,
                    high_conf_det_threshold=high_conf_det_threshold,
                    direction_consistency_weight=direction_consistency_weight,
                    delta_t=delta_t,
                )
            else:
                raise ValueError(f"Unknown tracker_type: {tracker_type!r}")

        tracker = self._trackers[key]
        merged = sv.Detections.merge(detections[i] for i in range(len(detections)))
        tracked_detections = tracker.update(merged)

        # Filter out immature / unmatched tracks (tracker_id == -1)
        if tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
            valid_mask = tracked_detections.tracker_id != -1
            tracked_detections = tracked_detections[valid_mask]

        if key not in self._per_video_cache:
            self._per_video_cache[key] = InstanceCache(size=instances_cache_size)
        cache = self._per_video_cache[key]

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
