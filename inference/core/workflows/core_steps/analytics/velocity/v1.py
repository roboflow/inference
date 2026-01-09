from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.constants import (
    SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS,
    SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS,
    SPEED_KEY_IN_SV_DETECTIONS,
    VELOCITY_KEY_IN_SV_DETECTIONS,
)
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
Calculate the velocity and speed of tracked objects across video frames by measuring displacement of object centers over time, applying exponential moving average smoothing to reduce noise, and converting measurements from pixels per second to meters per second for traffic speed monitoring, movement analysis, behavior tracking, and performance measurement workflows.

## How This Block Works

This block measures how fast objects are moving by tracking their positions across video frames. The block:

1. Receives tracked detection predictions with unique tracker IDs and an image with embedded video metadata
2. Extracts video metadata from the image:
   - Accesses video_metadata to get frame timestamps or frame numbers and frame rate (fps)
   - Extracts video_identifier to maintain separate tracking state for different videos
   - Determines current timestamp using frame_number/fps for video files or frame_timestamp for streams
3. Validates that detections have tracker IDs (required for tracking object movement across frames)
4. Calculates object center positions:
   - Computes the center point (x, y) of each bounding box in the current frame
   - Uses bounding box coordinates to find geometric centers
5. Retrieves or initializes tracking state:
   - Maintains previous positions and timestamps for each tracker_id per video
   - Stores smoothed velocity history for each tracker_id per video
   - Creates new tracking entries for objects appearing for the first time
6. Calculates velocity and speed for each tracked object:
   - **For objects with previous positions**: Computes displacement (change in position) and time delta (change in time) between current and previous frames
   - **Velocity**: Calculates velocity vector as displacement divided by time delta (pixels per second)
   - **Speed**: Computes speed as the magnitude (length) of the velocity vector (total pixels per second regardless of direction)
   - **For new objects**: Sets velocity and speed to zero (no movement data available yet)
7. Applies exponential moving average smoothing:
   - Smooths velocity measurements using exponential moving average with configurable smoothing factor (alpha)
   - Reduces noise and jitter in velocity calculations from detection variations
   - Lower alpha values provide more smoothing (slower response to changes), higher alpha values provide less smoothing (faster response to changes)
   - Calculates smoothed velocity and smoothed speed for each object
8. Converts units to meters per second:
   - Divides pixel-based velocities and speeds by pixels_per_meter conversion factor
   - Converts all measurements (velocity, speed, smoothed_velocity, smoothed_speed) to real-world units
   - Enables comparison with real-world speed measurements (e.g., km/h, mph)
9. Stores velocity data in detection metadata:
   - Adds four velocity metrics to each detection: velocity (m/s), speed (m/s), smoothed_velocity (m/s), smoothed_speed (m/s)
   - Velocity is a 2D vector [vx, vy] representing direction and magnitude of movement
   - Speed is a scalar value representing total speed regardless of direction
   - All measurements are stored in detections.data for downstream use
10. Updates tracking state for next frame:
    - Saves current positions and timestamps for all tracked objects
    - Stores smoothed velocities for next frame's smoothing calculations
11. Returns detections enhanced with velocity information:
    - Outputs the same detection objects with added velocity metadata
    - Each detection now includes velocity and speed data in its metadata

Velocity is calculated based on the displacement of object centers (bounding box centers) over time. The block maintains separate tracking state for each video, allowing velocity calculation across multiple video streams. Due to perspective distortion and camera positioning, calculated velocity may vary depending on where objects appear in the frame - objects closer to the camera or at different depths will have different pixel-per-second values for the same real-world speed. The smoothing helps reduce noise from detection inaccuracies and frame-to-frame variations.

## Common Use Cases

- **Traffic Speed Monitoring**: Measure vehicle speeds on roads and highways (e.g., monitor traffic speeds, detect speeding violations, analyze traffic flow rates), enabling traffic enforcement and analysis workflows
- **Sports Performance Analysis**: Track athlete movement and speed during sports activities (e.g., measure player speeds, analyze sprint performance, track movement patterns), enabling sports analytics workflows
- **Security and Surveillance**: Monitor movement speed of people or objects in security scenarios (e.g., detect running or suspicious rapid movement, monitor crowd flow speeds, track object movement rates), enabling security monitoring workflows
- **Retail Analytics**: Analyze customer movement patterns and walking speeds in retail spaces (e.g., measure customer flow rates, analyze shopping behavior patterns, track movement efficiency), enabling retail behavior analysis workflows
- **Wildlife Behavior Studies**: Track animal movement speeds and patterns in natural habitats (e.g., measure animal speeds, analyze migration patterns, study movement behavior), enabling wildlife research workflows
- **Industrial Monitoring**: Monitor speeds of vehicles, equipment, or products in industrial settings (e.g., track conveyor speeds, measure vehicle speeds in facilities, monitor production line movement rates), enabling industrial automation workflows

## Connecting to Other Blocks

This block receives tracked detections and an image with embedded video metadata, and produces detections enhanced with velocity metadata:

- **After Byte Tracker blocks** to calculate velocity for tracked objects (e.g., measure speeds of tracked vehicles, analyze tracked person movement, monitor tracked object velocities), enabling tracking-to-velocity workflows
- **After object detection or instance segmentation blocks** with tracking enabled to measure movement speeds (e.g., calculate vehicle speeds, track person movement rates, monitor object velocities), enabling detection-to-velocity workflows
- **Before visualization blocks** to display velocity information (e.g., visualize speed overlays, display velocity vectors, show movement speed annotations), enabling velocity visualization workflows
- **Before logic blocks** like Continue If to make decisions based on speed thresholds (e.g., continue if speed exceeds limit, filter based on velocity ranges, trigger actions on speed violations), enabling speed-based decision workflows
- **Before notification blocks** to alert on speed violations or threshold events (e.g., alert on speeding violations, notify on rapid movement, trigger speed-based alerts), enabling velocity-based notification workflows
- **Before data storage blocks** to record velocity measurements (e.g., log speed data, store velocity statistics, record movement metrics), enabling velocity data logging workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The image's video_metadata should include frame rate (fps) for video files or frame timestamps for streamed video to calculate accurate time deltas. The block maintains persistent tracking state across frames for each video using video_identifier, so it should be used in video workflows where frames are processed sequentially. For accurate velocity measurement, detections should be provided consistently across frames with valid tracker IDs. The pixels_per_meter conversion factor should be calibrated based on camera setup and scene geometry for accurate real-world speed measurements. Note that velocity accuracy may vary due to perspective distortion depending on object position in the frame.
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
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-gauge",
                "blockPriority": 2.5,
            },
        }
    )
    type: Literal["roboflow_core/velocity@v1"]
    image: WorkflowImageSelector = Field(
        description="Image with embedded video metadata. The video_metadata contains fps, frame_number, frame_timestamp, and video_identifier. Required for calculating time deltas and maintaining separate velocity tracking state for different videos.",
    )
    detections: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. Velocity is calculated based on displacement of bounding box centers over time. Output detections include velocity (m/s vector), speed (m/s scalar), smoothed_velocity (m/s vector), and smoothed_speed (m/s scalar) in detection metadata.",
        examples=["$steps.object_detection_model.predictions"],
    )
    smoothing_alpha: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=0.5,
        description="Smoothing factor (alpha) for exponential moving average, range 0 < alpha <= 1. Controls how much smoothing is applied to velocity measurements. Lower values (closer to 0) provide more smoothing - slower response to changes, less noise. Higher values (closer to 1) provide less smoothing - faster response to changes, more noise. Default 0.5 balances smoothness and responsiveness.",
        examples=[0.5, 0.3, 0.7],
    )
    pixels_per_meter: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=1.0,
        description="Conversion factor from pixels to meters for real-world speed calculation. Velocity measurements in pixels per second are divided by this value to convert to meters per second. Must be greater than 0. For accurate real-world speeds, calibrate based on camera height, angle, and scene geometry. Example: if 1 pixel = 0.01 meters (1cm), use 0.01. Default 1.0 means no conversion (results in pixels per second).",
        examples=[0.01, 0.1, 1.0],
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

        detections.data[VELOCITY_KEY_IN_SV_DETECTIONS] = np.array(velocities)
        detections.data[SPEED_KEY_IN_SV_DETECTIONS] = np.array(speeds)
        detections.data[SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS] = np.array(
            smoothed_velocities_arr
        )
        detections.data[SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS] = np.array(smoothed_speeds)

        return {OUTPUT_KEY: detections}
