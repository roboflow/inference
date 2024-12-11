from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData
)
from inference.core.workflows.execution_engine.entities.types import (
    WILDCARD_KIND,
    IMAGE_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

from inference.core.workflows.core_steps.cache.memory_cache import WorkflowMemoryCache

LONG_DESCRIPTION = """
Fetches a previously stored value from a cache entry.

Use the `Cache Set` block to store values in the cache.
"""

SHORT_DESCRIPTION = (
    "Fetches a previously stored value from a cache entry."
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Cache Get",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
        }
    )
    type: Literal["roboflow_core/cache_get@v1"]
    image: Selector(
        kind=[IMAGE_KIND],
        pattern=r"(^\$inputs.[A-Za-z_0-9\-]+$)"
    ) = Field(
        description="The image data to use as a reference for the cache namespace.",
        examples=["$inputs.image"],
    )
    key: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The key of the cache entry to fetch.",
        examples=["my_cache_key"],
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


class CacheGetBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.namespace = None

    def __del__(self):
        if self.namespace:
            WorkflowMemoryCache.clear_namespace(self.namespace)

    def run(self, image: WorkflowImageData, key: str) -> BlockResult:
        metadata = image.video_metadata
        namespace = metadata.video_identifier or "default"
        self.namespace = namespace
        
        cache = WorkflowMemoryCache.get_dict(namespace)
        return {"output": cache.get(key, None)}