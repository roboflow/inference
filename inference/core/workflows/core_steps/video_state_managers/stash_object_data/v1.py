from copy import copy, deepcopy
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.cache import TrackedInstancesCache
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
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


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stash Object Data",
            "version": "v1",
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "video_state_manager",
        }
    )
    type: Literal["roboflow_core/stash_object_data@v1"]
    video_metadata: WorkflowVideoMetadataSelector
    tracker_ids: StepOutputSelector(kind=[LIST_OF_VALUES_KIND])
    data_to_stash: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References data to be used to construct each and every cache field",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
    )
    data_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL definitions of operations to be performed on defined data w.r.t. field",
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
        default_factory=lambda: {},
    )

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {"data_to_stash": 1}

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "tracker_ids"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class StashObjectDataBlockV1(WorkflowBlock):

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
        tracker_ids: Optional[List[int]],
        data_to_stash: Dict[str, Batch[Any]],
        data_operations: Dict[str, AllOperationsType],
    ) -> BlockResult:
        if tracker_ids is None:
            return {}
        print("data_to_stash", data_to_stash["instance_appear_time"]._content)
        for key, batch in data_to_stash.items():
            for batch_idx, value in batch.iter_with_indices():
                if value is None:
                    continue
                tracker_id = tracker_ids[batch_idx[-1]]
                if key in data_operations:
                    operations_chain = build_operations_chain(
                        operations=data_operations[key]
                    )
                    value = operations_chain(value, global_parameters={})
                self._tracked_instances_cache.save(
                    video_id=video_metadata.video_identifier,
                    tracker_id=tracker_id,
                    field=key,
                    value=value,
                )
        return {}
