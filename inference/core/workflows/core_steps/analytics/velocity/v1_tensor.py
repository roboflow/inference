from typing import Dict, List, Optional, Tuple, Union

import torch
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.core_steps.common.tensor_native import (
    TensorNativeDetections,
    split_key_point_prediction,
)
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
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    Selector,
    StepOutputSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    BlockResult,
    RuntimeRestriction,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

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
            TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
                    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]


class _VideoVelocityState:
    """Device-resident velocity state for one video.

    Positions and smoothed velocities stay torch tensors on the prediction's
    device across frames - the state never round-trips through host memory.
    Only the row bookkeeping (tracker_id -> row) and the absolute timestamps
    live on host: timestamps are per-frame python floats by origin, and unix
    epochs do not fit float32 (~100 s resolution at 1.7e9) while MPS offers no
    float64 tensor to hold them.
    """

    def __init__(self) -> None:
        self.rows_by_tracker: Dict[int, int] = {}
        self.timestamps_by_tracker: Dict[int, float] = {}
        self.positions: Optional[torch.Tensor] = None  # (K, 2)
        self.smoothed_velocities: Optional[torch.Tensor] = None  # (K, 2)

    def to_device(self, device: torch.device) -> None:
        if self.positions is not None and self.positions.device != device:
            self.positions = self.positions.to(device)
            self.smoothed_velocities = self.smoothed_velocities.to(device)


class VelocityBlockV1(WorkflowBlock):
    def __init__(self):
        # Per-video device-tensor state (see _VideoVelocityState).
        self._states: Dict[str, _VideoVelocityState] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return VelocityManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: Union[
            Detections, InstanceDetections, Tuple[KeyPoints, Optional[Detections]]
        ],
        smoothing_alpha: float,
        pixels_per_meter: float,
    ) -> BlockResult:
        # Keypoint predictions arrive as a (KeyPoints, Detections) tuple; velocity
        # only needs the bbox component. Keep the keypoints to re-wrap the output
        # (velocity preserves detection order, so the components stay aligned).
        key_points, detections = split_key_point_prediction(detections)
        num_detections = int(detections.xyxy.shape[0])
        bboxes_metadata = detections.bboxes_metadata
        if bboxes_metadata is None:
            bboxes_metadata = [{} for _ in range(num_detections)]
        else:
            bboxes_metadata = [
                dict(box_metadata) if box_metadata is not None else {}
                for box_metadata in bboxes_metadata
            ]
        tracker_ids = [
            box_metadata.get("tracker_id") for box_metadata in bboxes_metadata
        ]
        if num_detections > 0 and any(tracker_id is None for tracker_id in tracker_ids):
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
        state = self._states.setdefault(video_id, _VideoVelocityState())

        if num_detections > 0:
            self._compute_and_attach_velocities(
                detections=detections,
                bboxes_metadata=bboxes_metadata,
                tracker_ids=[int(tracker_id) for tracker_id in tracker_ids],
                state=state,
                ts_current=float(ts_current),
                smoothing_alpha=float(smoothing_alpha),
                pixels_per_meter=float(pixels_per_meter),
            )
        detections.bboxes_metadata = bboxes_metadata

        if key_points is not None:
            return {OUTPUT_KEY: (key_points, detections)}
        return {OUTPUT_KEY: detections}

    @staticmethod
    def _compute_and_attach_velocities(
        detections: TensorNativeDetections,
        bboxes_metadata: List[dict],
        tracker_ids: List[int],
        state: _VideoVelocityState,
        ts_current: float,
        smoothing_alpha: float,
        pixels_per_meter: float,
    ) -> None:
        """Vectorised on-device velocity update.

        All geometry (centers, displacement, EMA smoothing, unit conversion)
        runs as torch ops on the prediction's device; the per-row previous
        state is gathered from the device-resident tensors. The ONLY
        device->host transfer is the single batched `.cpu()` of the (N, 6)
        result block at the end - required because the block's output contract
        stores velocities as python floats in ``bboxes_metadata``.
        """
        xyxy = detections.xyxy.detach()
        device = xyxy.device
        state.to_device(device)
        centers = (
            xyxy[:, :2].to(torch.float32) + xyxy[:, 2:].to(torch.float32)
        ) * 0.5  # (N, 2)

        # Host-side bookkeeping: tracker ids and timestamps are host data by
        # origin, so none of this touches the device.
        known_flags = [
            tracker_id in state.rows_by_tracker for tracker_id in tracker_ids
        ]
        previous_rows = [
            state.rows_by_tracker.get(tracker_id, 0) for tracker_id in tracker_ids
        ]
        time_deltas = [
            (ts_current - state.timestamps_by_tracker[tracker_id] if known else 0.0)
            for tracker_id, known in zip(tracker_ids, known_flags)
        ]

        if state.positions is not None and any(known_flags):
            row_index = torch.as_tensor(previous_rows, dtype=torch.long, device=device)
            previous_positions = state.positions[row_index]  # (N, 2)
            previous_smoothed = state.smoothed_velocities[row_index]  # (N, 2)
        else:
            previous_positions = centers
            previous_smoothed = torch.zeros_like(centers)

        known = torch.as_tensor(known_flags, dtype=torch.bool, device=device)
        deltas = torch.as_tensor(time_deltas, dtype=torch.float32, device=device)
        has_movement_window = known & (deltas > 0)
        safe_deltas = torch.where(deltas > 0, deltas, torch.ones_like(deltas))
        velocity = torch.where(
            has_movement_window.unsqueeze(1),
            (centers - previous_positions) / safe_deltas.unsqueeze(1),
            torch.zeros_like(centers),
        )  # pixels per second, (N, 2)
        speed = torch.linalg.vector_norm(velocity, dim=1)  # (N,)
        # EMA smoothing: known trackers blend with their previous smoothed
        # velocity; new trackers initialise with the current velocity. The
        # state keeps UNCONVERTED (pixels/s) values, exactly like the numpy
        # sibling - unit conversion applies to outputs only.
        smoothed_velocity = torch.where(
            known.unsqueeze(1),
            smoothing_alpha * velocity + (1 - smoothing_alpha) * previous_smoothed,
            velocity,
        )
        smoothed_speed = torch.linalg.vector_norm(smoothed_velocity, dim=1)

        # The single batched device->host hop: the output contract stores per-box
        # python floats in bboxes_metadata.
        results = (
            torch.cat(
                [
                    velocity,
                    speed.unsqueeze(1),
                    smoothed_velocity,
                    smoothed_speed.unsqueeze(1),
                ],
                dim=1,
            )
            / pixels_per_meter
        )
        results_host = results.cpu().tolist()  # (N, 6)
        for box_metadata, row in zip(bboxes_metadata, results_host):
            box_metadata[VELOCITY_KEY_IN_SV_DETECTIONS] = [row[0], row[1]]
            box_metadata[SPEED_KEY_IN_SV_DETECTIONS] = row[2]
            box_metadata[SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS] = [row[3], row[4]]
            box_metadata[SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS] = row[5]

        # State update - stays on device. Trackers absent from this frame keep
        # their previous position/timestamp/smoothing (they may reappear), same
        # as the numpy sibling's dict semantics.
        current_set = set(tracker_ids)
        survivors = [
            (tracker_id, row)
            for tracker_id, row in state.rows_by_tracker.items()
            if tracker_id not in current_set
        ]
        if survivors and state.positions is not None:
            survivor_index = torch.as_tensor(
                [row for _, row in survivors], dtype=torch.long, device=device
            )
            new_positions = torch.cat([state.positions[survivor_index], centers], dim=0)
            new_smoothed = torch.cat(
                [state.smoothed_velocities[survivor_index], smoothed_velocity], dim=0
            )
        else:
            survivors = []
            new_positions = centers
            new_smoothed = smoothed_velocity
        state.positions = new_positions
        state.smoothed_velocities = new_smoothed
        rows_by_tracker = {
            tracker_id: row for row, (tracker_id, _) in enumerate(survivors)
        }
        timestamps_by_tracker = {
            tracker_id: state.timestamps_by_tracker[tracker_id]
            for tracker_id, _ in survivors
        }
        base = len(survivors)
        for offset, tracker_id in enumerate(tracker_ids):
            rows_by_tracker[tracker_id] = base + offset
            timestamps_by_tracker[tracker_id] = ts_current
        state.rows_by_tracker = rows_by_tracker
        state.timestamps_by_tracker = timestamps_by_tracker
