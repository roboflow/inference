from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.constants import (
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    VIDEO_METADATA_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "path_deviation_detections"
SHORT_DESCRIPTION = "Calculate Fréchet distance of object from the reference path."
LONG_DESCRIPTION = """
Measure how closely tracked objects follow a reference path by calculating the Fréchet distance between the object's actual trajectory and the expected reference path, enabling path compliance monitoring, route deviation detection, quality control in automated systems, and behavioral analysis workflows.

## How This Block Works

This block compares the actual movement path of tracked objects against a predefined reference path to measure deviation. The block:

1. Receives tracked detection predictions with unique tracker IDs, video metadata, and a reference path definition
2. Validates that detections have tracker IDs (required for tracking object movement across frames)
3. Initializes or retrieves path tracking state for the video:
   - Maintains a history of positions for each tracked object per video
   - Stores object paths using video_identifier to separate state for different videos
   - Creates new path tracking entries for objects appearing for the first time
4. Extracts anchor point coordinates for each detection:
   - Uses the triggering_anchor to determine which point on the bounding box to track (default: CENTER)
   - Gets the (x, y) coordinates of the anchor point for each detection in the current frame
   - The anchor point represents the position of the object used for path comparison
5. Accumulates object paths over time:
   - Appends each object's anchor point to its path history as frames are processed
   - Maintains separate path histories for each unique tracker_id
   - Builds complete trajectory paths by accumulating positions across all processed frames
6. Calculates Fréchet distance for each tracked object:
   - **Fréchet Distance**: Measures the similarity between two curves (paths) considering both location and ordering of points
   - Compares the object's accumulated path (actual trajectory) against the reference path (expected trajectory)
   - Uses dynamic programming to compute the minimum "leash length" required to traverse both paths simultaneously
   - Accounts for the order of points along each path, not just point-to-point distances
   - Lower values indicate the object follows the reference path closely, higher values indicate greater deviation
7. Stores path deviation in detection metadata:
   - Adds the Fréchet distance value to each detection's metadata
   - Each detection includes path_deviation representing how much it deviates from the reference path
   - Distance is measured in pixels (same units as image coordinates)
8. Maintains persistent path tracking:
   - Path histories accumulate across frames for the entire video
   - Each object's deviation is calculated based on its complete path from the start of tracking
   - Separate tracking state maintained for each video_identifier
9. Returns detections enhanced with path deviation information:
   - Outputs detection objects with added path_deviation metadata
   - Each detection now includes the Fréchet distance measuring its deviation from the reference path

The Fréchet distance is a metric that measures the similarity between two curves by finding the minimum length of a "leash" that connects a point moving along one curve to a point moving along the other curve, where both points move forward along their respective curves. Unlike simple Euclidean distance, Fréchet distance considers the ordering and continuity of points along paths, making it ideal for comparing trajectories where the sequence of movement matters. An object that follows the reference path exactly will have a Fréchet distance of 0, while objects that deviate significantly will have larger distances.

## Common Use Cases

- **Path Compliance Monitoring**: Monitor whether vehicles, robots, or objects follow predefined routes (e.g., verify vehicles stay in lanes, check robots follow programmed paths, ensure objects follow expected routes), enabling compliance monitoring workflows
- **Quality Control**: Detect deviations in manufacturing or assembly processes where objects should follow specific paths (e.g., detect conveyor belt deviations, monitor assembly line paths, check product movement patterns), enabling quality control workflows
- **Traffic Analysis**: Analyze vehicle movement patterns and detect lane departures or route deviations (e.g., detect vehicles leaving lanes, monitor route adherence, analyze traffic pattern compliance), enabling traffic analysis workflows
- **Security Monitoring**: Detect suspicious movement patterns or deviations from expected paths in security scenarios (e.g., detect unauthorized route deviations, monitor perimeter breach attempts, track movement compliance), enabling security monitoring workflows
- **Automated Systems**: Monitor and validate that automated systems (robots, AGVs, drones) follow expected paths correctly (e.g., verify robot navigation accuracy, check automated vehicle paths, validate drone flight paths), enabling automated system validation workflows
- **Behavioral Analysis**: Study movement patterns and path adherence in behavioral research (e.g., analyze animal movement patterns, study path following behavior, measure route preference deviations), enabling behavioral research workflows

## Connecting to Other Blocks

This block receives tracked detections, video metadata, and a reference path, and produces detections enhanced with path_deviation metadata:

- **After Byte Tracker blocks** to measure path deviation for tracked objects (e.g., measure tracked vehicle path compliance, analyze tracked person route adherence, monitor tracked object path deviations), enabling tracking-to-path-analysis workflows
- **After object detection or instance segmentation blocks** with tracking enabled to analyze movement paths (e.g., analyze vehicle paths, track object route compliance, measure path deviations), enabling detection-to-path-analysis workflows
- **Before visualization blocks** to display path deviation information (e.g., visualize paths and deviations, display reference and actual paths, show deviation metrics), enabling path deviation visualization workflows
- **Before logic blocks** like Continue If to make decisions based on path deviation thresholds (e.g., continue if deviation exceeds limit, filter based on path compliance, trigger actions on route violations), enabling path-based decision workflows
- **Before notification blocks** to alert on path deviations or compliance violations (e.g., alert on route deviations, notify on path compliance issues, trigger deviation-based alerts), enabling path-based notification workflows
- **Before data storage blocks** to record path deviation measurements (e.g., log path compliance data, store deviation statistics, record route adherence metrics), enabling path deviation data logging workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The reference path must be defined as a list of at least 2 points, where each point is a tuple or list of exactly 2 coordinates (x, y). The block requires video metadata with video_identifier to maintain separate path tracking state for different videos. The block maintains persistent path tracking across frames for each video, accumulating complete trajectories, so it should be used in video workflows where frames are processed sequentially. For accurate path deviation measurement, detections should be provided consistently across frames with valid tracker IDs. The Fréchet distance is calculated in pixels (same units as image coordinates).
"""


class PathDeviationManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Path Deviation",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-tower-observation",
            },
        }
    )
    type: Literal["roboflow_core/path_deviation_analytics@v1"]
    metadata: Selector(kind=[VIDEO_METADATA_KIND]) = Field(
        description="Video metadata containing video_identifier to maintain separate path tracking state for different videos. Required for persistent path accumulation across frames.",
    )
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. The block tracks anchor point positions across frames to build object trajectories and compares them against the reference path. Output detections include path_deviation metadata containing the Fréchet distance from the reference path.",
        examples=["$steps.object_detection_model.predictions"],
    )
    triggering_anchor: Union[str, Selector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]] = Field(  # type: ignore
        description="Point on the bounding box used to track object position for path calculation. Options: CENTER (default), BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. This anchor point's coordinates are accumulated over frames to build the object's trajectory path, which is compared against the reference path using Fréchet distance.",
        default="CENTER",
        examples=["CENTER"],
    )
    reference_path: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Expected reference path as a list of at least 2 points, where each point is a tuple or list of [x, y] coordinates. Example: [(100, 200), (200, 300), (300, 400)] defines a path with 3 points. The Fréchet distance measures how closely tracked objects follow this reference path. Points should be ordered along the expected trajectory.",
        examples=[[(100, 200), (200, 300), (300, 400)], "$inputs.expected_path"],
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


class PathDeviationAnalyticsBlockV1(WorkflowBlock):
    def __init__(self):
        self._object_paths: Dict[
            str, Dict[Union[int, str], List[Tuple[float, float]]]
        ] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PathDeviationManifest

    def run(
        self,
        detections: sv.Detections,
        metadata: VideoMetadata,
        triggering_anchor: str,
        reference_path: List[Tuple[int, int]],
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )

        video_id = metadata.video_identifier
        if video_id not in self._object_paths:
            self._object_paths[video_id] = {}

        anchor_points = detections.get_anchors_coordinates(anchor=triggering_anchor)
        result_detections = []
        for i, tracker_id in enumerate(detections.tracker_id):
            detection = detections[i]
            anchor_point = anchor_points[i]
            if tracker_id not in self._object_paths[video_id]:
                self._object_paths[video_id][tracker_id] = []
            self._object_paths[video_id][tracker_id].append(anchor_point)

            object_path = np.array(self._object_paths[video_id][tracker_id])
            ref_path = np.array(reference_path)

            frechet_distance = self._calculate_frechet_distance(object_path, ref_path)
            detection[PATH_DEVIATION_KEY_IN_SV_DETECTIONS] = np.array(
                [frechet_distance], dtype=np.float64
            )
            result_detections.append(detection)

        return {OUTPUT_KEY: sv.Detections.merge(result_detections)}

    def _calculate_frechet_distance(
        self, path1: np.ndarray, path2: np.ndarray
    ) -> float:
        dist_matrix = np.ones((len(path1), len(path2))) * -1
        return self._compute_distance(
            dist_matrix, len(path1) - 1, len(path2) - 1, path1, path2
        )

    def _compute_distance(
        self,
        dist_matrix: np.ndarray,
        i: int,
        j: int,
        path1: np.ndarray,
        path2: np.ndarray,
    ) -> float:
        if dist_matrix[i, j] > -1:
            return dist_matrix[i, j]
        elif i == 0 and j == 0:
            dist_matrix[i, j] = self._euclidean_distance(path1[0], path2[0])
        elif i > 0 and j == 0:
            dist_matrix[i, j] = max(
                self._compute_distance(dist_matrix, i - 1, 0, path1, path2),
                self._euclidean_distance(path1[i], path2[0]),
            )
        elif i == 0 and j > 0:
            dist_matrix[i, j] = max(
                self._compute_distance(dist_matrix, 0, j - 1, path1, path2),
                self._euclidean_distance(path1[0], path2[j]),
            )
        elif i > 0 and j > 0:
            dist_matrix[i, j] = max(
                min(
                    self._compute_distance(dist_matrix, i - 1, j, path1, path2),
                    self._compute_distance(dist_matrix, i - 1, j - 1, path1, path2),
                    self._compute_distance(dist_matrix, i, j - 1, path1, path2),
                ),
                self._euclidean_distance(path1[i], path2[j]),
            )
        else:
            dist_matrix[i, j] = float("inf")
        return dist_matrix[i, j]

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sqrt(np.sum((point1 - point2) ** 2))
