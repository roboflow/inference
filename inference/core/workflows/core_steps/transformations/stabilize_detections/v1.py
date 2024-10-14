from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "tracked_detections"
LONG_DESCRIPTION = """
This block stores last known position for each bounding box
If box disappears then this block will bring it back so short gaps are filled with last known box position
The block requires detections to be tracked (i.e. each object must have unique tracker_id assigned,
which persists between frames)
WARNING: this block will produce many short-lived bounding boxes for unstable trackers!
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Stabilizer",
            "version": "v1",
            "short_description": "Apply smoothing algorithm to reduce noise and flickering across video frames",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/guard_tracked_detections@v1"]
    image: WorkflowImageSelector
    detections: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked detections",
        examples=["$steps.object_detection_model.predictions"],
    )
    consider_detection_gone_timeout: Union[Optional[float], WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=2,
        description="Drop detections that had not been seen for longer than this timeout (in seconds)",
        examples=[2, "$inputs.disappeared_detections_timeout"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class StabilizeTrackedDetectionsBlockV1(WorkflowBlock):
    def __init__(self):
        self._batch_of_last_known_detections: Dict[
            str, Dict[Union[int, str], Tuple[float, sv.Detections]]
        ] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        consider_detection_gone_timeout: float,
    ) -> BlockResult:
        metadata = image.video_metadata
        if metadata.comes_from_video_file and metadata.fps != 0:
            ts = metadata.frame_number / metadata.fps
        else:
            ts = metadata.frame_timestamp.timestamp()
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        cached_detections = self._batch_of_last_known_detections.setdefault(
            metadata.video_identifier, {}
        )
        this_frame_tracked_ids = set()
        for i, tracked_id in zip(range(len(detections)), detections.tracker_id):
            this_frame_tracked_ids.add(tracked_id)
            cached_detections[tracked_id] = (ts, detections[i])
        for tracked_id in list(cached_detections.keys()):
            last_seen_ts = cached_detections[tracked_id][0]
            if ts - last_seen_ts > consider_detection_gone_timeout:
                del cached_detections[tracked_id]
        return {
            OUTPUT_KEY: sv.Detections.merge(
                cached_detection[1] for cached_detection in cached_detections.values()
            )
        }
