from typing import Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
    WorkflowVideoMetadataSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "tracked_detections"
TYPE: str = "roboflow_core/byte_tracker@v1"
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
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        },
        protected_namespaces=(),
    )
    type: Literal[f"{TYPE}", "ByteTracker"]
    metadata: WorkflowVideoMetadataSelector
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Objects to be tracked",
        examples=["$steps.object_detection_model.predictions"],
    )
    track_activation_threshold: Union[Optional[float], WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.25,
        description="Detection confidence threshold for track activation",
        examples=[0.25, "$inputs.confidence"],
    )
    lost_track_buffer: Union[Optional[int], WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=30,
        description="Number of frames to buffer when a track is lost",
        examples=[30, "$inputs.lost_track_buffer"],
    )
    minimum_matching_threshold: Union[Optional[float], WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.8,
        description="Threshold for matching tracks with detections",
        examples=[0.8, "$inputs.min_matching_threshold"],
    )
    frame_rate: Union[Optional[float], WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=None,
        description="Frame rate of the video",
        examples=[10, "$inputs.frame_rate"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.1.0,<2.0.0"


class ByteTrackerBlockV1(WorkflowBlock):
    def __init__(
        self,
    ):
        self._trackers: Dict[str, sv.ByteTrack] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ByteTrackerBlockManifest

    def run(
        self,
        metadata: VideoMetadata,
        predictions: sv.Detections,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: Optional[float] = None,
    ) -> BlockResult:
        if metadata.video_identifier not in self._trackers:
            if frame_rate is None:
                frame_rate = metadata.fps
            if not frame_rate:
                frame_rate = 10
            self._trackers[metadata.video_identifier] = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate,
            )
        tracker = self._trackers[metadata.video_identifier]
        tracked_detections = tracker.update_with_detections(predictions)
        return {OUTPUT_KEY: tracked_detections}
