from collections import deque
from typing import Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
    WorkflowVideoMetadataSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "On Object Location Updated",
            "version": "v1",
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "video_trigger",
        }
    )
    type: Literal["roboflow_core/on_object_location_updated@v1"]
    video_metadata: WorkflowVideoMetadataSelector
    predictions: StepOutputSelector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
    )
    per_video_cache_size: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            default=16384,
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="tracker_ids",
                kind=[LIST_OF_VALUES_KIND],
            ),
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class OnObjectLocationUpdatedBlockV1(WorkflowBlock):

    def __init__(self):
        self._per_video_cache: Dict[str, InstanceCache] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        video_metadata: VideoMetadata,
        predictions: sv.Detections,
        per_video_cache_size: int,
    ) -> BlockResult:
        if predictions.tracker_id is None:
            return {
                "tracker_ids": None,
                "predictions": None,
            }
        if video_metadata.video_identifier not in self._per_video_cache:
            self._per_video_cache[video_metadata.video_identifier] = InstanceCache(
                size=per_video_cache_size
            )
        cache = self._per_video_cache[video_metadata.video_identifier]
        output_predictions_mask = []
        for tracker_id in predictions.tracker_id.tolist():
            output_predictions_mask.append(
                not cache.record_instance(tracker_id=tracker_id)
            )
        filtered_prediction = predictions[output_predictions_mask]
        if len(filtered_prediction) == 0:
            return {
                "tracker_ids": None,
                "predictions": None,
            }
        return {
            "tracker_ids": filtered_prediction.tracker_id.tolist(),
            "predictions": filtered_prediction,
        }


class InstanceCache:

    def __init__(self, size: int):
        size = max(1, size)
        self._cache_inserts_track = deque(maxlen=size)
        self._cache = set()

    def record_instance(self, tracker_id: int) -> bool:
        in_cache = tracker_id in self._cache
        if not in_cache:
            self._cache_new_tracker_id(tracker_id=tracker_id)
        return in_cache

    def _cache_new_tracker_id(self, tracker_id: int) -> None:
        while len(self._cache) >= self._cache_inserts_track.maxlen:
            to_drop = self._cache_inserts_track.popleft()
            self._cache.remove(to_drop)
        self._cache_inserts_track.append(tracker_id)
        self._cache.add(tracker_id)
