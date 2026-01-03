from collections import deque
from typing import Deque, Dict, List, Literal, Optional, Set, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

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
LONG_DESCRIPTION = """
Apply smoothing algorithms to reduce noise and flickering in tracked detections across video frames by using Kalman filtering to predict object velocities, exponential moving average to smooth bounding box positions, and gap filling to restore temporarily missing detections for improved tracking stability and smoother visualization workflows.

## How This Block Works

This block stabilizes tracked detections by reducing jitter, smoothing positions, and filling gaps when objects temporarily disappear from detection. The block:

1. Receives tracked detection predictions with unique tracker IDs and an image with embedded video metadata
2. Extracts video metadata from the image:
   - Accesses video_metadata to get video_identifier
   - Uses video_identifier to maintain separate stabilization state for different videos
3. Validates that detections have tracker IDs (required for tracking object movement across frames)
4. Initializes or retrieves stabilization state for the video:
   - Maintains a cache of last known detections for each tracker_id per video
   - Creates or retrieves a Kalman filter for velocity prediction per video
   - Stores separate state for each video using video_identifier
5. Measures object velocities for existing tracks:
   - Calculates velocity by comparing current frame bounding box centers to previous frame centers
   - Computes displacement (change in position) for objects present in both current and previous frames
   - Velocity measurements are used to update the Kalman filter
6. Updates Kalman filter with velocity measurements:
   - Uses Kalman filtering to predict smoothed velocities based on historical measurements
   - Maintains a sliding window of velocity measurements (controlled by smoothing_window_size)
   - Applies exponential moving average within the Kalman filter to smooth velocity estimates
   - Filters out noise from detection inaccuracies and frame-to-frame variations
7. Smooths bounding boxes for objects present in current frame:
   - Applies exponential moving average smoothing to bounding box coordinates
   - Combines previous frame position with current frame position using bbox_smoothing_coefficient
   - Formula: smoothed_bbox = alpha * current_bbox + (1 - alpha) * previous_bbox
   - Reduces jitter and flickering from detection variations
8. Predicts positions for missing detections:
   - Uses Kalman filter predicted velocities to estimate positions of objects that disappeared
   - Applies predicted velocity to last known bounding box position
   - Fills gaps by restoring detections that were temporarily missing from current frame
   - Smooths predicted positions using exponential moving average
9. Manages tracking state:
   - Updates cache with current frame detections for next frame calculations
   - Removes tracking entries for objects that have been missing longer than smoothing_window_size frames
   - Maintains separate state per video_identifier
10. Merges and returns stabilized detections:
    - Combines smoothed detections (from current frame) and predicted detections (for missing objects)
    - Outputs stabilized detection objects with reduced noise and filled gaps
    - All detections maintain their tracker IDs for consistent tracking

The block uses two complementary smoothing techniques: **Kalman filtering** for velocity prediction (estimating how fast objects are moving) and **exponential moving average** for position smoothing (reducing bounding box jitter). The Kalman filter maintains a history of velocity measurements and uses statistical estimation to predict future velocities while filtering out noise. The exponential moving average smooths bounding box coordinates by blending current and previous positions. Gap filling uses predicted velocities to restore detections that temporarily disappear, helping maintain track continuity. Note: This block may produce short-lived bounding boxes for unstable trackers, as it attempts to fill gaps even when objects are inconsistently detected.

## Common Use Cases

- **Video Visualization**: Reduce flickering and jitter in video annotations for smoother visualizations (e.g., smooth bounding box movements, reduce annotation noise, improve video visualization quality), enabling stable video visualization workflows
- **Tracking Stability**: Improve tracking stability when detections are noisy or inconsistent (e.g., stabilize noisy detections, reduce tracking jitter, improve tracking continuity), enabling stable tracking workflows
- **Temporary Occlusion Handling**: Fill gaps when objects are temporarily occluded or missing from detections (e.g., maintain tracks during brief occlusions, fill detection gaps, preserve tracking continuity), enabling occlusion handling workflows
- **Real-Time Monitoring**: Improve visual quality in real-time monitoring applications (e.g., smooth live video annotations, reduce flickering in monitoring displays, improve real-time visualization), enabling stable real-time monitoring workflows
- **Analytics Accuracy**: Reduce noise in analytics calculations that depend on stable detection positions (e.g., improve position-based analytics, reduce noise in measurements, stabilize movement calculations), enabling accurate analytics workflows
- **Quality Control**: Improve detection quality for downstream processing (e.g., smooth detections before analysis, reduce noise for better processing, stabilize inputs for other blocks), enabling quality improvement workflows

## Connecting to Other Blocks

This block receives tracked detections and an image, and produces stabilized tracked_detections:

- **After Byte Tracker blocks** to stabilize tracked detections (e.g., smooth tracked object positions, reduce tracking jitter, fill tracking gaps), enabling tracking-stabilization workflows
- **After object detection or instance segmentation blocks** with tracking enabled to stabilize detections (e.g., smooth detection positions, reduce detection noise, improve tracking stability), enabling detection-stabilization workflows
- **Before visualization blocks** to display stabilized detections (e.g., visualize smooth bounding boxes, display stable annotations, show gap-filled detections), enabling stable visualization workflows
- **Before analytics blocks** to provide stable inputs for analysis (e.g., analyze stabilized positions, process smooth movement data, work with gap-filled detections), enabling stable analytics workflows
- **Before velocity or path analysis blocks** to improve measurement accuracy (e.g., calculate velocities from stable positions, analyze paths from smooth trajectories, measure from gap-filled detections), enabling accurate measurement workflows
- **In video processing pipelines** where detection stability is required for downstream processing (e.g., stabilize detections in processing chains, improve quality for analysis, reduce noise in pipelines), enabling stable video processing workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The image's video_metadata should include video_identifier to maintain separate stabilization state for different videos. The block maintains persistent stabilization state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For optimal stabilization, detections should be provided consistently across frames with valid tracker IDs. The smoothing_window_size controls how many historical velocity measurements are used for Kalman filtering and how long missing detections are retained. The bbox_smoothing_coefficient (0-1) controls the balance between current and previous positions - lower values provide more smoothing but slower response to changes, higher values provide less smoothing but faster response. Note: This block may produce short-lived bounding boxes for unstable trackers as it attempts to fill gaps even when objects are inconsistently detected.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Stabilizer",
            "version": "v1",
            "short_description": "Apply a smoothing algorithm to reduce noise and flickering across video frames.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "video",
                "icon": "fas fa-waveform-lines",
                "blockPriority": 4,
            },
        }
    )
    type: Literal["roboflow_core/stabilize_detections@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Image with embedded video metadata. The video_metadata contains video_identifier to maintain separate stabilization state for different videos. Required for persistent state management across frames.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. The block applies Kalman filtering for velocity prediction, exponential moving average for position smoothing, and gap filling for missing detections. Output detections are stabilized with reduced noise and jitter.",
        examples=["$steps.object_detection_model.predictions"],
    )
    smoothing_window_size: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=3,
        description="Size of the sliding window for velocity smoothing in Kalman filter, controlling how many historical velocity measurements are used. Also determines how long missing detections are retained before removal. Larger values provide more smoothing but slower adaptation to changes. Smaller values provide less smoothing but faster adaptation. Detections missing for longer than this number of frames are removed from tracking state. Typical range: 3-10 frames.",
        examples=[3, 5, 10, "$inputs.smoothing_window_size"],
    )
    bbox_smoothing_coefficient: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.2,
        description="Exponential moving average coefficient (alpha) for bounding box position smoothing, range 0.0-1.0. Controls the blend between current and previous bounding box positions: smoothed_bbox = alpha * current + (1-alpha) * previous. Lower values (closer to 0) provide more smoothing - slower response to changes, less jitter. Higher values (closer to 1) provide less smoothing - faster response to changes, more jitter. Default 0.2 balances smoothness and responsiveness. Typical range: 0.1-0.5.",
        examples=[0.2, 0.1, 0.5, "$inputs.bbox_smoothing_coefficient"],
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
        return ">=1.3.0,<2.0.0"


class StabilizeTrackedDetectionsBlockV1(WorkflowBlock):
    def __init__(self):
        self._batch_of_last_known_detections: Dict[
            str, Dict[Union[int, str], sv.Detections]
        ] = {}
        self._batch_of_kalman_filters: Dict[Union[int, str], VelocityKalmanFilter] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        smoothing_window_size: int,
        bbox_smoothing_coefficient: float,
    ) -> BlockResult:
        metadata = image.video_metadata
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        cached_detections = self._batch_of_last_known_detections.setdefault(
            metadata.video_identifier, {}
        )
        kalman_filter = self._batch_of_kalman_filters.setdefault(
            metadata.video_identifier,
            VelocityKalmanFilter(smoothing_window_size=smoothing_window_size),
        )
        measured_velocities = {}
        for i, (tracker_id, xyxy) in enumerate(
            zip(detections.tracker_id, detections.xyxy)
        ):
            if tracker_id not in cached_detections:
                continue
            x1, y1, x2, y2 = xyxy
            this_frame_center_xy = [x1 + abs(x2 - x1), y1 + abs(y2 - y1)]
            x1, y1, x2, y2 = cached_detections[tracker_id].xyxy[0]
            prev_frame_center_xy = [x1 + abs(x2 - x1), y1 + abs(y2 - y1)]
            measured_velocities[tracker_id] = (
                this_frame_center_xy[0] - prev_frame_center_xy[0],
                this_frame_center_xy[1] - prev_frame_center_xy[1],
            )
        predicted_velocities = kalman_filter.update(measurements=measured_velocities)

        predicted_detections = {}
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id in cached_detections:
                prev_frame_detection = cached_detections[tracker_id]
                prev_frame_xyxy = prev_frame_detection.xyxy[0]
                curr_frame_detection = detections[i]
                curr_frame_xyxy = curr_frame_detection.xyxy[0]
                curr_frame_detection.xyxy[0] = smooth_xyxy(
                    prev_xyxy=prev_frame_xyxy,
                    curr_xyxy=curr_frame_xyxy,
                    alpha=bbox_smoothing_coefficient,
                )
                predicted_detections[tracker_id] = curr_frame_detection
            else:
                predicted_detections[tracker_id] = detections[i]
            cached_detections[tracker_id] = detections[i]
        for tracker_id, predicted_velocity in predicted_velocities.items():
            if tracker_id in predicted_detections:
                continue
            prev_frame_detection = cached_detections[tracker_id]
            prev_frame_xyxy = prev_frame_detection.xyxy[0]
            curr_frame_xyxy = np.array(
                [
                    prev_frame_detection.xyxy[0]
                    + np.array([predicted_velocity, predicted_velocity]).flatten()
                ]
            )
            prev_frame_detection.xyxy = smooth_xyxy(
                prev_xyxy=prev_frame_xyxy,
                curr_xyxy=curr_frame_xyxy,
                alpha=bbox_smoothing_coefficient,
            )
            predicted_detections[tracker_id] = prev_frame_detection
        for tracker_id in list(cached_detections.keys()):
            if (
                tracker_id not in kalman_filter.tracked_vectors
                and tracker_id not in predicted_detections
            ):
                del cached_detections[tracker_id]
        merged_detections = sv.Detections.merge(predicted_detections.values())
        if len(merged_detections) == 0:
            merged_detections.tracker_id = np.array([])
        return {OUTPUT_KEY: merged_detections}


def smooth_xyxy(prev_xyxy: np.ndarray, curr_xyxy: np.ndarray, alpha=0.2) -> np.ndarray:
    smoothed_xyxy = alpha * curr_xyxy + (1 - alpha) * prev_xyxy

    return smoothed_xyxy


class VelocityKalmanFilter:
    def __init__(self, smoothing_window_size: int):
        self.time_step = 1
        self.smoothing_window_size = smoothing_window_size
        self.state_transition_matrix = np.array([[1, 0], [0, 1]])
        self.process_noise_covariance = np.eye(2) * 0.001
        self.measurement_noise_covariance = np.eye(2) * 0.01
        self.tracked_vectors: Dict[
            Union[int, str],
            Dict[
                Literal["velocity", "error_covariance", "history"],
                Union[np.ndarray, Deque[float, float]],
            ],
        ] = {}

    def predict(self) -> Dict[Union[int, str], np.ndarray]:
        predictions: Dict[Union[int, str], np.ndarray] = {}
        for tracker_id, data in self.tracked_vectors.items():
            data["velocity"] = np.dot(self.state_transition_matrix, data["velocity"])
            data["error_covariance"] = (
                np.dot(
                    np.dot(self.state_transition_matrix, data["error_covariance"]),
                    self.state_transition_matrix.T,
                )
                + self.process_noise_covariance
            )
            predictions[tracker_id] = data["velocity"]
        return predictions

    def update(
        self, measurements: Dict[Union[int, str], Tuple[float, float]]
    ) -> Dict[Union[int, str], np.ndarray]:
        updated_vector_ids: Set[Union[int, str]] = set()
        for tracker_id, velocity in measurements.items():
            updated_vector_ids.add(tracker_id)
            if tracker_id in self.tracked_vectors:
                measurement = np.array(velocity).reshape(2, 1)
                tracked_vector = self.tracked_vectors[tracker_id]
                tracked_vector["history"].appendleft(measurement)
                smoothed_measurement = np.mean(tracked_vector["history"], axis=0)
                measurement_residual = smoothed_measurement - tracked_vector["velocity"]
                residual_covariance = (
                    tracked_vector["error_covariance"]
                    + self.measurement_noise_covariance
                )
                kalman_gain = np.dot(
                    tracked_vector["error_covariance"],
                    np.linalg.inv(residual_covariance),
                )
                tracked_vector["velocity"] = tracked_vector["velocity"] + np.dot(
                    kalman_gain, measurement_residual
                )
                tracked_vector["error_covariance"] = tracked_vector[
                    "error_covariance"
                ] - np.dot(kalman_gain, tracked_vector["error_covariance"])
            else:
                self.tracked_vectors[tracker_id] = {
                    "velocity": np.array([[velocity[0]], [velocity[1]]]),
                    "error_covariance": np.eye(2),
                    "history": deque(
                        [np.array([[velocity[0]], [velocity[1]]])],
                        maxlen=self.smoothing_window_size,
                    ),
                }

        predicted_velocities = self.predict()

        for tracker_id in set(self.tracked_vectors.keys()) - updated_vector_ids:
            if self.tracked_vectors[tracker_id]["history"]:
                self.tracked_vectors[tracker_id]["history"].popleft()
            if not self.tracked_vectors[tracker_id]["history"]:
                del self.tracked_vectors[tracker_id]

        return predicted_velocities
