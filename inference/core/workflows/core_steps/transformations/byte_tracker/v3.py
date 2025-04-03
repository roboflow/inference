from collections import deque
from typing import Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core import logger
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
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
    "Track and update object positions across video frames using ByteTrack."
)
LONG_DESCRIPTION = """
The `ByteTrackerBlock` integrates ByteTrack, an advanced object tracking algorithm, 
to manage object tracking across sequential video frames within workflows.

This block accepts detections and their corresponding video frames as input, 
initializing trackers for each detection based on configurable parameters like track 
activation threshold, lost track buffer, minimum matching threshold, and frame rate. 
These parameters allow fine-tuning of the tracking process to suit specific accuracy 
and performance needs.

!!! Note "New outputs introduced in `v3`"

    The block has not changed compared to `v2` apart from the fact that there are two 
    new outputs added:
    
    * **`new_instances`:** delivers sv.Detections objects with bounding boxes that have 
    tracker IDs which were first seen - specific tracked instance **will only be
    listed in that output once - when new tracker ID is generated** 
    
    * **`already_seen_instances`:** delivers sv.Detections objects with bounding boxes that have 
    tracker IDs which were already seen - specific tracked instance **will only be
    listed in that output each time the tracker associates the bounding box with already seen
    tracker ID** 
"""


class ByteTrackerBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Byte Tracker",
            "version": "v3",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-location-crosshairs",
                "blockPriority": 0,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/byte_tracker@v3"]
    image: Selector(kind=[IMAGE_KIND])
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Objects to be tracked.",
        examples=["$steps.object_detection_model.predictions"],
    )
    track_activation_threshold: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.25,
        description="Detection confidence threshold for track activation."
        " Increasing track_activation_threshold improves accuracy and stability but might miss true detections."
        " Decreasing it increases completeness but risks introducing noise and instability.",
        examples=[0.25, "$inputs.confidence"],
    )
    lost_track_buffer: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=30,
        description="Number of frames to buffer when a track is lost."
        " Increasing lost_track_buffer enhances occlusion handling, significantly reducing"
        " the likelihood of track fragmentation or disappearance caused by brief detection gaps.",
        examples=[30, "$inputs.lost_track_buffer"],
    )
    minimum_matching_threshold: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.8,
        description="Threshold for matching tracks with detections."
        " Increasing minimum_matching_threshold improves accuracy but risks fragmentation."
        " Decreasing it improves completeness but risks false positives and drift.",
        examples=[0.8, "$inputs.min_matching_threshold"],
    )
    minimum_consecutive_frames: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=1,
        description="Number of consecutive frames that an object must be tracked before it is considered a 'valid' track."
        " Increasing minimum_consecutive_frames prevents the creation of accidental tracks from false detection"
        " or double detection, but risks missing shorter tracks.",
        examples=[1, "$inputs.min_consecutive_frames"],
    )
    instances_cache_size: int = Field(
        default=16384,
        description="Size of the instances cache to decide if specific tracked instance is new or already seen",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
            OutputDefinition(
                name="new_instances", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name="already_seen_instances", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ByteTrackerBlockV3(WorkflowBlock):
    def __init__(
        self,
    ):
        self._trackers: Dict[str, sv.ByteTrack] = {}
        self._per_video_cache: Dict[str, InstanceCache] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ByteTrackerBlockManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        instances_cache_size: int = 16384,
    ) -> BlockResult:
        metadata = image.video_metadata
        fps = metadata.fps
        if not fps:
            fps = 0
            logger.warning(
                f"Malformed fps in VideoMetadata, {self.__class__.__name__} requires fps in order to initialize ByteTrack"
            )
        if metadata.video_identifier not in self._trackers:
            self._trackers[metadata.video_identifier] = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                minimum_consecutive_frames=minimum_consecutive_frames,
                frame_rate=fps,
            )
        tracker = self._trackers[metadata.video_identifier]
        tracked_detections = tracker.update_with_detections(
            sv.Detections.merge(detections[i] for i in range(len(detections)))
        )
        if metadata.video_identifier not in self._per_video_cache:
            self._per_video_cache[metadata.video_identifier] = InstanceCache(
                size=instances_cache_size
            )
        cache = self._per_video_cache[metadata.video_identifier]
        not_seen_instances_mask, seen_instances_mask = [], []
        for tracker_id in tracked_detections.tracker_id.tolist():
            already_seen = cache.record_instance(tracker_id=tracker_id)
            not_seen_instances_mask.append(not already_seen)
            seen_instances_mask.append(already_seen)
        not_seen_instances_detections = tracked_detections[not_seen_instances_mask]
        already_seen_instances_detections = tracked_detections[seen_instances_mask]
        return {
            OUTPUT_KEY: tracked_detections,
            "new_instances": not_seen_instances_detections,
            "already_seen_instances": already_seen_instances_detections,
        }


class InstanceCache:

    def __init__(self, size: int):
        size = max(1, size)
        self._cache_inserts_track = deque(maxlen=size)
        self._cache = set()

    def record_instance(self, tracker_id: int) -> bool:
        in_cache = tracker_id in self._cache
        if not in_cache:
            self._cache_new_tracker_id(tracker_id=tracker_id)
        return in_cache

    def _cache_new_tracker_id(self, tracker_id: int) -> None:
        while len(self._cache) >= self._cache_inserts_track.maxlen:
            to_drop = self._cache_inserts_track.popleft()
            self._cache.remove(to_drop)
        self._cache_inserts_track.append(tracker_id)
        self._cache.add(tracker_id)
