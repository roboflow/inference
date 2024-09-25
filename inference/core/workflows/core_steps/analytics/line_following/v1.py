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
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
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

OUTPUT_KEY: str = "frechet_distance"
SHORT_DESCRIPTION = "Calculate Fréchet distance of object from reference path"
LONG_DESCRIPTION = """
The `LineFollowingAnalyticsBlock` is an analytics block designed to measure the Frechet distance
of tracked objects from a user-defined reference path. The block requires detections to be tracked
(i.e. each object must have a unique tracker_id assigned, which persists between frames).
"""

class LineFollowingManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Line following",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )
    type: Literal["roboflow_core/line_following_analytics@v1"]
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
    triggering_anchor: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description=f"Triggering anchor. Allowed values: {', '.join(sv.Position.list())}",
        default="CENTER",
        examples=["CENTER"],
    )
    reference_path: Union[list, StepOutputSelector(kind=[LIST_OF_VALUES_KIND]), WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Reference path in a format [(x1, y1), (x2, y2), (x3, y3), ...]",
        examples=["$inputs.expected_path"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[FLOAT_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

class LineFollowingAnalyticsBlockV1(WorkflowBlock):
    def __init__(self):
        self._object_paths: Dict[str, Dict[Union[int, str], List[Tuple[float, float]]]] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LineFollowingManifest

    def _calculate_frechet_distance(self, path1: np.ndarray, path2: np.ndarray) -> float:
        def euclidean_distance(point1, point2):
            return np.sqrt(np.sum((point1 - point2) ** 2))

        def compute_distance(dist_matrix, i, j, path1, path2):
            if dist_matrix[i, j] > -1:
                return dist_matrix[i, j]
            elif i == 0 and j == 0:
                dist_matrix[i, j] = euclidean_distance(path1[0], path2[0])
            elif i > 0 and j == 0:
                dist_matrix[i, j] = max(compute_distance(dist_matrix, i-1, 0, path1, path2), euclidean_distance(path1[i], path2[0]))
            elif i == 0 and j > 0:
                dist_matrix[i, j] = max(compute_distance(dist_matrix, 0, j-1, path1, path2), euclidean_distance(path1[0], path2[j]))
            elif i > 0 and j > 0:
                dist_matrix[i, j] = max(min(compute_distance(dist_matrix, i-1, j, path1, path2), 
                                            compute_distance(dist_matrix, i-1, j-1, path1, path2), 
                                            compute_distance(dist_matrix, i, j-1, path1, path2)),
                                        euclidean_distance(path1[i], path2[j]))
            else:
                dist_matrix[i, j] = float("inf")
            return dist_matrix[i, j]

        dist_matrix = np.ones((len(path1), len(path2))) * -1
        return compute_distance(dist_matrix, len(path1)-1, len(path2)-1, path1, path2)

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

        max_frechet_distance = 0.0

        for i, tracker_id in enumerate(detections.tracker_id):
            anchor_point = getattr(detections, triggering_anchor.lower())[i]

            if tracker_id not in self._object_paths[video_id]:
                self._object_paths[video_id][tracker_id] = []
            self._object_paths[video_id][tracker_id].append(tuple(anchor_point))

            object_path = np.array(self._object_paths[video_id][tracker_id])
            ref_path = np.array(reference_path)

            frechet_distance = self._calculate_frechet_distance(object_path, ref_path)
            max_frechet_distance = max(max_frechet_distance, frechet_distance)

        return {OUTPUT_KEY: max_frechet_distance}
