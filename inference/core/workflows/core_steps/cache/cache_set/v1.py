from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.cache.memory_cache import WorkflowMemoryCache
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    Selector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Stores a value in a cache entry for later retrieval.

Use the `Cache Get` block to fetch values from the cache.
"""

SHORT_DESCRIPTION = "Stores a value in a cache entry for later retrieval."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Cache Set",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "advanced",
                "icon": "far fa-memory",
            },
        }
    )
    type: Literal["roboflow_core/cache_set@v1"]
    image: WorkflowImageSelector = Field(
        description="The image data to use as a reference for the cache namespace.",
        examples=["$inputs.image"],
    )
    key: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The key of the cache entry to set.",
        examples=["my_cache_key"],
    )
    value: Union[Selector(kind=[WILDCARD_KIND, LIST_OF_VALUES_KIND])] = Field(
        description="The value to store in the cache.",
        examples=["any_value"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output",
                kind=[WILDCARD_KIND],
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CacheSetBlockV1(WorkflowBlock):
    namespace = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def __init__(
        self,
        step_execution_mode: StepExecutionMode,
    ):
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["step_execution_mode"]

    def __del__(self):
        if self.namespace:
            WorkflowMemoryCache.clear_namespace(self.namespace)

    def run(self, image: WorkflowImageData, key: str, value: Any) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(
                "Cache blocks require running locally or on a dedicated deployment."
            )

        metadata = image.video_metadata
        namespace = metadata.video_identifier or "default"
        self.namespace = namespace

        cache = WorkflowMemoryCache.get_dict(namespace)
        cache[key] = value
        return {"output": value}
