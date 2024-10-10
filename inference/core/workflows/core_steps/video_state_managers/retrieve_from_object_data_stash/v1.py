from typing import Dict, List, Literal, Optional, Type

from pydantic import ConfigDict

from inference.core.workflows.core_steps.common.cache import TrackedInstancesCache
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    StepOutputSelector,
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
            "name": "Retrieve From Object Data Stash",
            "version": "v1",
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "video_state_manager",
        }
    )
    type: Literal["roboflow_core/retrieve_from_object_data_stash@v1"]
    video_metadata: WorkflowVideoMetadataSelector
    tracker_ids: StepOutputSelector(kind=[LIST_OF_VALUES_KIND])
    data_to_retrieve: List[str]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="tracker_id", kind=[INTEGER_KIND]),
            OutputDefinition(name="*"),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        return [OutputDefinition(name="tracker_id", kind=[INTEGER_KIND])] + [
            OutputDefinition(name=field) for field in self.data_to_retrieve
        ]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class RetrieveFromObjectDataStashBlockV1(WorkflowBlock):

    def __init__(self, tracked_instances_cache: TrackedInstancesCache):
        self._tracked_instances_cache = tracked_instances_cache

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["tracked_instances_cache"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        video_metadata: VideoMetadata,
        tracker_ids: List[int],
        data_to_retrieve: List[str],
    ) -> BlockResult:
        results = []
        for tracker_id in tracker_ids:
            result = {
                "tracker_id": tracker_id,
            }
            for field in data_to_retrieve:
                result[field] = self._tracked_instances_cache.get(
                    video_id=video_metadata.video_identifier,
                    tracker_id=tracker_id,
                    field=field,
                )
            results.append(result)
        print("STASH RESULTS", results)
        return results
