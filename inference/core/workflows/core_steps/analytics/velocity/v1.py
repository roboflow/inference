from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
    StepOutputSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "velocity_detections"
SHORT_DESCRIPTION = "Calculate the velocity and speed of tracked objects with smoothing and unit conversion."
LONG_DESCRIPTION = """
The `VelocityBlock` computes the velocity and speed of objects tracked across video frames.
It includes options to smooth the velocity and speed measurements over time and to convert units from pixels per second to meters per second.
It requires detections from Byte Track with unique `tracker_id` assigned to each object, which persists between frames.
The velocities are calculated based on the displacement of object centers over time.

Note: due to perspective and camera distortions calculated velocity will be different depending on object position in relation to the camera.

"""
VELOCITY_KEY_IN_SV_DETECTIONS = "velocity"
SPEED_KEY_IN_SV_DETECTIONS = "speed"
SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS = "smoothed_velocity"
SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS = "smoothed_speed"


class VelocityManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Velocity",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-gauge",
                "blockPriority": 2.5,
            },
        }
    )
    type: Literal["roboflow_core/velocity@v1"]
    image: WorkflowImageSelector
    detections: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Model predictions to calculate the velocity for.",
        examples=["$steps.object_detection_model.predictions"],
    )
    smoothing_alpha: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=0.5,
        description="Smoothing factor (alpha) for exponential moving average (0 < alpha <= 1). Lower alpha means more smoothing.",
        examples=[0.5],
    )
    pixels_per_meter: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=1.0,
        description="Conversion from pixels to meters. Velocity will be converted to meters per second using this value.",
        examples=[0.01],  # Example: 1 pixel = 0.01 meters
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
        # Store previous positions and timestamps for each tracker_id
        self._previous_positions: Dict[
            str, Dict[Union[int, str], Tuple[np.ndarray, float]]
        ] = {}
        # Store smoothed velocities for each tracker_id
        self._smoothed_velocities: Dict[str, Dict[Union[int, str], np.ndarray]] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return VelocityManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        smoothing_alpha: float,
        pixels_per_meter: float,
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                "tracker_id not initialized, VelocityBlock requires detections to be tracked"
            )
        if not (0 < smoothing_alpha <= 1):
            raise ValueError(
                "smoothing_alpha must be between 0 (exclusive) and 1 (inclusive)"
            )
        if not (pixels_per_meter > 0):
            raise ValueError("pixels_per_meter must be greater than 0")

        if image.video_metadata.comes_from_video_file and image.video_metadata.fps != 0:
            ts_current = image.video_metadata.frame_number / image.video_metadata.fps
        else:
            ts_current = image.video_metadata.frame_timestamp.timestamp()

        video_id = image.video_metadata.video_identifier
        previous_positions = self._previous_positions.setdefault(video_id, {})
        smoothed_velocities = self._smoothed_velocities.setdefault(video_id, {})

        num_detections = len(detections)

        # Compute current positions (center of bounding boxes)
        bbox_xyxy = detections.xyxy  # Shape (num_detections, 4)
        x_centers = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2
        y_centers = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2
        current_positions = np.stack(
            [x_centers, y_centers], axis=1
        )  # Shape (num_detections, 2)

        velocities = np.zeros_like(current_positions)  # Shape (num_detections, 2)
        speeds = np.zeros(num_detections)  # Shape (num_detections,)
        smoothed_velocities_arr = np.zeros_like(current_positions)
        smoothed_speeds = np.zeros(num_detections)

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
                    speed = np.linalg.norm(
                        velocity
                    )  # Speed is the magnitude of velocity vector
                else:
                    velocity = np.array([0, 0])
                    speed = 0.0
            else:
                velocity = np.array([0, 0])  # No previous position
                speed = 0.0

            # Apply exponential moving average for smoothing
            if tracker_id in smoothed_velocities:
                prev_smoothed_velocity = smoothed_velocities[tracker_id]
                smoothed_velocity = (
                    smoothing_alpha * velocity
                    + (1 - smoothing_alpha) * prev_smoothed_velocity
                )
            else:
                smoothed_velocity = velocity  # Initialize with current velocity

            smoothed_speed = np.linalg.norm(smoothed_velocity)

            # Store current position and timestamp for the next frame
            previous_positions[tracker_id] = (current_position, ts_current)
            smoothed_velocities[tracker_id] = smoothed_velocity

            # Convert velocities and speeds to meters per second if required
            velocity_m_s = velocity / pixels_per_meter
            smoothed_velocity_m_s = smoothed_velocity / pixels_per_meter
            speed_m_s = speed / pixels_per_meter
            smoothed_speed_m_s = smoothed_speed / pixels_per_meter

            velocities[i] = velocity_m_s
            speeds[i] = speed_m_s
            smoothed_velocities_arr[i] = smoothed_velocity_m_s
            smoothed_speeds[i] = smoothed_speed_m_s

            # Add velocity and speed to detections.data
            # Ensure that 'data' is a dictionary for each detection
            if detections.data is None:
                detections.data = {}

            # Initialize dictionaries if not present
            if VELOCITY_KEY_IN_SV_DETECTIONS not in detections.data:
                detections.data[VELOCITY_KEY_IN_SV_DETECTIONS] = {}
            if SPEED_KEY_IN_SV_DETECTIONS not in detections.data:
                detections.data[SPEED_KEY_IN_SV_DETECTIONS] = {}
            if SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS not in detections.data:
                detections.data[SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS] = {}
            if SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS not in detections.data:
                detections.data[SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS] = {}

            # Assign velocity data to the corresponding tracker_id
            detections.data[VELOCITY_KEY_IN_SV_DETECTIONS][
                tracker_id
            ] = velocity_m_s.tolist()  # [vx, vy]
            detections.data[SPEED_KEY_IN_SV_DETECTIONS][
                tracker_id
            ] = speed_m_s  # Scalar
            detections.data[SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS][
                tracker_id
            ] = smoothed_velocity_m_s.tolist()  # [vx, vy]
            detections.data[SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS][
                tracker_id
            ] = smoothed_speed_m_s  # Scalar

        return {OUTPUT_KEY: detections}
