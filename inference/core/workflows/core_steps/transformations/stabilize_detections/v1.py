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
    image: Selector(kind=[IMAGE_KIND])
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked detections",
        examples=["$steps.object_detection_model.predictions"],
    )
    smoothing_window_size: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=3,
        description="Predicted movement of detection will be smoothed based on historical measurements of velocity,"
        " this parameter controls number of historical measurements taken under account when calculating smoothed velocity."
        " Detections will be removed from generating smoothed predictions if they had been missing for longer than this number of frames.",
        examples=[5, "$inputs.smoothing_window_size"],
    )
    bbox_smoothing_coefficient: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.2,
        description="Bounding box smoothing coefficient applied when given tracker_id is present on current frame."
        " This parameter must be initialized with value between 0 and 1",
        examples=[0.2, "$inputs.bbox_smoothing_coefficient"],
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
