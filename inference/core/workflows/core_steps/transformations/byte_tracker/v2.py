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
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
    WorkflowImageSelector,
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
"""


class ByteTrackerBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Byte Tracker",
            "version": "v2",
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
    type: Literal["roboflow_core/byte_tracker@v2"]
    image: WorkflowImageSelector
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

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ByteTrackerBlockV2(WorkflowBlock):
    def __init__(
        self,
    ):
        self._trackers: Dict[str, sv.ByteTrack] = {}

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
        return {OUTPUT_KEY: tracked_detections}
