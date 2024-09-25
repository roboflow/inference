from copy import deepcopy
from typing import Dict, List, Literal, Optional, Type

import supervision as sv
from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
    WorkflowVideoMetadataSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class VideoMetadataBlockManifest(WorkflowBlockManifest):
    type: Literal["ExampleVideoMetadataProcessing"]
    name: str = Field(description="name field")
    metadata: WorkflowVideoMetadataSelector

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="frame_number", kind=[INTEGER_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.1.0,<2.0.0"


class VideoMetadataBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return VideoMetadataBlockManifest

    def run(
        self,
        metadata: VideoMetadata,
    ) -> BlockResult:
        return {"frame_number": metadata.frame_number}


class TrackerManifest(WorkflowBlockManifest):
    type: Literal["Tracker"]
    name: str = Field(description="name field")
    metadata: WorkflowVideoMetadataSelector
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="tracker_id", kind=[LIST_OF_VALUES_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.1.0,<2.0.0"


class TrackerBlock(WorkflowBlock):

    def __init__(self):
        self._trackers: Dict[str, sv.ByteTrack] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TrackerManifest

    def run(
        self,
        metadata: VideoMetadata,
        predictions: sv.Detections,
    ) -> BlockResult:
        if metadata.video_identifier not in self._trackers:
            self._trackers[metadata.video_identifier] = sv.ByteTrack()
        tracked_detections = self._trackers[
            metadata.video_identifier
        ].update_with_detections(detections=deepcopy(predictions))
        result = (
            tracked_detections.tracker_id.tolist()
            if tracked_detections.tracker_id is not None
            else None
        )
        return {"tracker_id": result}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [VideoMetadataBlock, TrackerBlock]
