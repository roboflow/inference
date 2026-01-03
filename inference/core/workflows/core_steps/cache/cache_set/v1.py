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
Store a value in an in-memory cache by key, using the image's video identifier as a namespace to enable data sharing between workflow steps, caching intermediate results, and avoiding redundant computations within the same workflow execution context.

## How This Block Works

This block stores values in an in-memory cache that can be later retrieved using the Cache Get block. The block:

1. Receives image, cache key, and value to store:
   - Takes an input image to determine the cache namespace
   - Receives a cache key (string) identifying the cache entry
   - Receives a value (any data type) to store in the cache
2. Determines cache namespace:
   - Extracts video identifier from the image's video metadata
   - Uses the video identifier as the cache namespace (isolates cache entries per video/stream)
   - Falls back to "default" namespace if no video identifier is present
3. Stores value in cache:
   - Accesses the in-memory cache dictionary for the determined namespace
   - Stores the value using the specified key in the cache
   - Overwrites any existing value with the same key (cache keys are unique within a namespace)
4. Returns stored value:
   - Outputs the stored value as a pass-through (same value that was stored)
   - The output can be used by subsequent workflow steps

The cache is namespaced by video identifier, meaning different videos or streams have separate cache storage. This allows workflows processing multiple videos to maintain separate caches for each video. The cache is stored in memory and is cleared when the workflow execution completes or when the block is destroyed. Cache Set must be used in conjunction with Cache Get - values are stored with Cache Set and retrieved with Cache Get using the same key and namespace (determined by the same video identifier).

## Common Use Cases

- **Shared State Between Steps**: Store intermediate results in one workflow step for retrieval in another step (e.g., store detection results for later analysis, cache classification predictions for filtering, save metadata for subsequent blocks), enabling state sharing workflows
- **Avoid Redundant Computations**: Cache expensive computation results for reuse across multiple workflow steps (e.g., cache model predictions, store processed images, save transformation results), enabling computation caching workflows
- **Video Frame Context**: Maintain context across video frames by storing frame-specific data (e.g., cache previous frame detections, store frame sequence metadata, save tracking state), enabling frame context workflows
- **Conditional Workflow Logic**: Store decision results or flags that control workflow execution in subsequent steps (e.g., cache filtering decisions, store validation results, save workflow state), enabling conditional execution workflows
- **Data Aggregation**: Accumulate data across workflow steps by storing values in cache (e.g., store detection counts, cache statistics, save result collections), enabling data aggregation workflows
- **Temporary Storage**: Use cache as temporary storage for values that need to be accessed by multiple workflow steps without passing through the workflow graph (e.g., store cross-step data, maintain temporary state, share non-linear workflow data), enabling temporary storage workflows

## Connecting to Other Blocks

This block stores values in cache and passes through the stored value:

- **Before Cache Get block** to store values that will be retrieved later (e.g., store detections for retrieval, cache predictions for later use, save metadata for access), enabling cache storage workflows
- **After model or processing blocks** to cache their outputs for later use (e.g., cache model predictions, store processed results, save computation outputs), enabling result caching workflows
- **In workflow branches** to store shared values accessible from parallel or conditional execution paths (e.g., store shared state, cache common results, save branch data), enabling branch coordination workflows
- **Before blocks that need cached data** to store values that will be used by subsequent blocks (e.g., store inputs for later processing, cache data for filtering, save values for analysis), enabling cached data workflows
- **In conditional logic workflows** to store flags or decisions for later use (e.g., store validation results, cache decision flags, save conditional state), enabling conditional logic workflows
- **With video processing workflows** to maintain frame-specific or video-specific cache namespaces (e.g., store frame context, cache video-specific data, save stream-specific values), enabling video context workflows

## Requirements

This block requires an input image (used to determine the cache namespace via video identifier), a cache key (string) to identify the cache entry, and a value (any data type) to store. The block only works in LOCAL execution mode - it will raise a NotImplementedError if used in other execution modes. Values stored in the cache can be retrieved later using the Cache Get block with the same key and namespace (same video identifier). The cache is stored in memory and is automatically cleared when the workflow execution completes. The cache is namespaced by video identifier, so different videos have separate cache storage. If a key already exists in the cache, storing a new value with the same key will overwrite the previous value. The stored value can be any data type (strings, numbers, lists, detections, images, etc.).
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
        description="Input image used to determine the cache namespace. The block extracts the video identifier from the image's video metadata and uses it as the cache namespace. If no video identifier is present, the block uses 'default' as the namespace. The namespace isolates cache entries so different videos or streams have separate cache storage. Use the same image (with the same video identifier) for both Cache Set and Cache Get blocks to access the same cache namespace.",
        examples=["$inputs.image", "$steps.input.output"],
    )
    key: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Cache key (string) identifying the cache entry where the value will be stored. The key must be used with the same value when retrieving the value with the Cache Get block. Keys are case-sensitive and must be exact matches. If a key already exists in the cache, storing a new value will overwrite the previous value. Use descriptive keys to identify different cached values (e.g., 'detections', 'classification_result', 'frame_metadata').",
        examples=["my_cache_key", "detections", "classification_result", "$inputs.cache_key"],
    )
    value: Union[Selector(kind=[WILDCARD_KIND, LIST_OF_VALUES_KIND])] = Field(
        description="Value to store in the cache. Can be any data type including strings, numbers, lists, detections, images, classifications, or any other workflow data type. The value is stored in memory and can be retrieved later using the Cache Get block with the same key and namespace. The value is also passed through as the block's output, allowing it to be used by subsequent workflow steps.",
        examples=["any_value", "$steps.detection.predictions", "$steps.classification.predictions", "$inputs.metadata"],
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
