from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
# TODO: Cleanup below imports
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
    WorkflowVideoMetadataSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "velocity_detections"
SHORT_DESCRIPTION = "Calculate the velocity of tracked objects in video frames."
LONG_DESCRIPTION = """
The `VelocityBlock` computes the velocity of objects tracked across video frames.
It requires detections with unique `tracker_id` assigned to each object, which persists between frames.
The velocities are calculated based on the displacement of object centers over time.
"""


class VelocityManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Velocity",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )
    type: Literal["roboflow_core/velocity@v1"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    metadata: WorkflowVideoMetadataSelector
    detections: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.object_detection_model.predictions"],
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


class VelocityBlockV1(WorkflowBlock):
    def __init__(self):
        self._previous_positions: Dict[
            str, Dict[Union[int, str], Tuple[np.ndarray, float]]
        ] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return VelocityManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        metadata: VideoMetadata,
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                "tracker_id not initialized, VelocityBlock requires detections to be tracked"
            )
        if metadata.comes_from_video_file and metadata.fps != 0:
            ts_current = metadata.frame_number / metadata.fps
        else:
            ts_current = metadata.frame_timestamp.timestamp()

        video_id = metadata.video_identifier
        previous_positions = self._previous_positions.setdefault(video_id, {})

        num_detections = len(detections)

        # Compute current positions (center of bounding boxes)
        bbox_xyxy = detections.xyxy  # Shape (num_detections, 4)
        x_centers = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
        y_centers = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
        current_positions = np.stack([x_centers, y_centers], axis=1)  # Shape (num_detections, 2)

        velocities = np.zeros_like(current_positions)  # Shape (num_detections, 2)
        speeds = np.zeros(num_detections)  # Shape (num_detections,)

        for i, tracker_id in enumerate(detections.tracker_id):
            current_position = current_positions[i]

            # Ensure tracker_id is of type int or str
            tracker_id = int(tracker_id)

            if tracker_id in previous_positions:
                prev_position, prev_timestamp = previous_positions[tracker_id]
                delta_time = ts_current - prev_timestamp

                if delta_time > 0:
                    displacement = current_position - prev_position
                    velocity = displacement / delta_time  # Pixels per second
                    speed = np.linalg.norm(velocity)  # Speed is the magnitude of velocity vector
                else:
                    velocity = np.array([0, 0])
                    speed = 0.0
            else:
                velocity = np.array([0, 0])  # No previous position
                speed = 0.0

            # Store current position and timestamp for the next frame
            previous_positions[tracker_id] = (current_position, ts_current)

            velocities[i] = velocity
            speeds[i] = speed

        # Add velocities and speeds to detections
        detections.data['velocity'] = velocities
        detections.data['speed'] = speeds

        return {OUTPUT_KEY: detections}