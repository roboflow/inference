from threading import Lock
from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.cache.memory_cache import WorkflowMemoryCache
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    Selector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    BlockResult,
    RuntimeRestriction,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Retrieve a previously stored value from an in-memory cache by key, using the image's video identifier as a namespace to enable data sharing between workflow steps, caching intermediate results, and avoiding redundant computations within the same workflow execution context.

## How This Block Works

This block retrieves values from an in-memory cache that was previously stored using the Cache Set block. The block:

1. Receives image and cache key:
   - Takes an input image to determine the cache namespace
   - Receives a cache key (string) identifying which value to retrieve
2. Determines cache namespace:
   - Extracts video identifier from the image's video metadata
   - Uses the video identifier as the cache namespace (isolates cache entries per video/stream)
   - Falls back to "default" namespace if no video identifier is present
3. Looks up cached value:
   - Accesses the in-memory cache dictionary for the determined namespace
   - Searches for the specified key in the cache
   - Returns the cached value if found, or False if the key does not exist
4. Returns retrieved value:
   - Outputs the cached value (can be any data type: strings, numbers, lists, detections, etc.)
   - Returns False if the key was not found in the cache
   - The output type matches whatever was originally stored with Cache Set

The cache is namespaced by video identifier, meaning different videos or streams have separate cache storage. This allows workflows processing multiple videos to maintain separate caches for each video. The cache lives in process-wide memory. Each Cache Set / Cache Get instance retains every video namespace it touches and releases those retains when the instance is closed or garbage-collected; a namespace is deleted only when no instance still retains it. Cleanup is not tied to workflow execution completing. Cache Get must be used in conjunction with Cache Set - values are stored with Cache Set and retrieved with Cache Get using the same key and namespace (determined by the same video identifier).

## Common Use Cases

- **Shared State Between Steps**: Store intermediate results in one workflow step and retrieve them in another step (e.g., store detection results for later analysis, cache classification predictions for filtering, share metadata between blocks), enabling state sharing workflows
- **Avoid Redundant Computations**: Cache expensive computation results and reuse them across multiple workflow steps (e.g., cache model predictions, store processed images, reuse transformation results), enabling computation caching workflows
- **Video Frame Context**: Maintain context across video frames by storing frame-specific data (e.g., cache previous frame detections, store frame sequence metadata, maintain tracking state), enabling frame context workflows
- **Conditional Workflow Logic**: Store decision results or flags that control workflow execution in subsequent steps (e.g., cache filtering decisions, store validation results, maintain workflow state), enabling conditional execution workflows
- **Data Aggregation**: Accumulate data across workflow steps by storing values in cache and retrieving/updating them (e.g., aggregate detection counts, accumulate statistics, build result collections), enabling data aggregation workflows
- **Temporary Storage**: Use cache as temporary storage for values that need to be accessed by multiple workflow steps without passing through the workflow graph (e.g., store cross-step data, maintain temporary state, share non-linear workflow data), enabling temporary storage workflows

## Connecting to Other Blocks

This block retrieves cached values and can be used throughout workflows:

- **After Cache Set block** to retrieve values that were previously stored (e.g., retrieve stored detections, get cached predictions, access stored metadata), enabling cache retrieval workflows
- **In workflow branches** to access shared cache values from parallel or conditional execution paths (e.g., retrieve shared state, access cached results, get common data), enabling branch coordination workflows
- **Before blocks that need cached data** to provide cached values as input (e.g., provide cached detections to analysis, use cached predictions for filtering, pass cached metadata to processing), enabling cached input workflows
- **In conditional logic workflows** to retrieve flags or decisions stored by Cache Set (e.g., get cached validation results, retrieve decision flags, access conditional state), enabling conditional logic workflows
- **With video processing workflows** to maintain frame-specific or video-specific cache namespaces (e.g., retrieve frame context, access video-specific cache, get stream-specific data), enabling video context workflows
- **Before output or sink blocks** to include cached data in final results (e.g., include cached aggregations, output cached statistics, return cached results), enabling output workflows

## Requirements

This block requires an input image (used to determine the cache namespace via video identifier) and a cache key (string) to look up the stored value. The cache is held in process memory, so it is only reliable when one long-lived process (an InferencePipeline or a stateful video worker) handles every frame of a video; on stateless multi-replica HTTP deployments entries may not survive between requests. Values must be previously stored using the Cache Set block with the same key and namespace (same video identifier). The cache lives in process-wide memory and is not cleared merely because a workflow run finishes; each block instance releases the namespaces it retained when it is closed or garbage-collected. The cache is namespaced by video identifier, so different videos have separate cache storage. If a key is not found in the cache, the block returns False. The cached value can be any data type (strings, numbers, lists, detections, images, etc.) depending on what was originally stored.
"""

SHORT_DESCRIPTION = "Fetches a previously stored value from a cache entry."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Cache Get",
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
    type: Literal["roboflow_core/cache_get@v1"]
    image: WorkflowImageSelector = Field(
        description="Input image used to determine the cache namespace. The block extracts the video identifier from the image's video metadata and uses it as the cache namespace. If no video identifier is present, the block uses 'default' as the namespace. The namespace isolates cache entries so different videos or streams have separate cache storage. Use the same image (with the same video identifier) for both Cache Set and Cache Get blocks to access the same cache namespace.",
        examples=["$inputs.image", "$steps.input.output"],
    )
    key: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Cache key (string) identifying which value to retrieve from the cache. The key must match the key used when storing the value with the Cache Set block. If the key does not exist in the cache, the block returns False. Keys are case-sensitive and must be exact matches. Use descriptive keys to identify different cached values (e.g., 'detections', 'classification_result', 'frame_metadata').",
        examples=[
            "my_cache_key",
            "detections",
            "classification_result",
            "$inputs.cache_key",
        ],
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

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        # The cache lives in this process, keyed by video_identifier, so it is
        # only meaningful when one engine instance sees every frame of a video.
        # That is exactly the caveat the shared stateful-video preset states.
        # Where model steps execute is irrelevant: this block has no remote
        # path and always runs in-process.
        return [STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION]


class CacheGetBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def __init__(self):
        # Namespaces this instance has retained. close() releases each once;
        # __del__ calls close() as a GC fallback, not as "workflow complete".
        self._namespaces: set = set()
        # Per-instance lock: workflows run steps in a ThreadPoolExecutor, so
        # the same block instance can be called concurrently. The lock makes
        # the check→retain→get_dict sequence atomic and protects cleanup.
        self._lock = Lock()

    def close(self) -> None:
        with self._lock:
            namespaces = getattr(self, "_namespaces", None)
            if not namespaces:
                return
            for namespace in list(namespaces):
                WorkflowMemoryCache.release_namespace(namespace)
            namespaces.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def run(self, image: WorkflowImageData, key: str) -> BlockResult:
        metadata = image.video_metadata
        namespace = metadata.video_identifier or "default"
        # Per-instance lock prevents a race where Thread A adds the namespace
        # to _namespaces but hasn't called get_dict yet, and Thread B sees it
        # in the set and reads WorkflowMemoryCache.cache directly — KeyError.
        with self._lock:
            if namespace not in self._namespaces:
                self._namespaces.add(namespace)
                cache = WorkflowMemoryCache.get_dict(namespace)
            else:
                cache = WorkflowMemoryCache.cache[namespace]
            return {"output": cache.get(key, False)}
