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
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "path_deviation_detections"
SHORT_DESCRIPTION = "Calculate Fréchet distance of object from the reference path."
LONG_DESCRIPTION = """
The `PathDeviationAnalyticsBlock` is an analytics block designed to measure the Frechet distance
of tracked objects from a user-defined reference path. The block requires detections to be tracked
(i.e. each object must have a unique tracker_id assigned, which persists between frames).
"""


class PathDeviationManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Path Deviation",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-road",
                "blockPriority": 3,
            },
        }
    )
    type: Literal["roboflow_core/path_deviation_analytics@v2"]
    image: WorkflowImageSelector
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    triggering_anchor: Union[str, Selector(kind=[STRING_KIND]), Literal[tuple(sv.Position.list())]] = Field(  # type: ignore
        description=f"Triggering anchor. Allowed values: {', '.join(sv.Position.list())}",
        default="CENTER",
        examples=["CENTER"],
    )
    reference_path: Union[list, Selector(kind=[LIST_OF_VALUES_KIND]), Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Reference path in a format [(x1, y1), (x2, y2), (x3, y3), ...]",
        examples=["$inputs.expected_path"],
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
        return ">=1.2.0,<2.0.0"


class PathDeviationAnalyticsBlockV2(WorkflowBlock):
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
        image: WorkflowImageData,
        triggering_anchor: str,
        reference_path: List[Tuple[int, int]],
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                f"tracker_id not initialized, {self.__class__.__name__} requires detections to be tracked"
            )
        metadata = image.video_metadata
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
