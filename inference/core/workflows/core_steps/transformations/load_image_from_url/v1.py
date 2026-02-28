import hashlib
from typing import List, Literal, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field

from inference.core.cache.lru_cache import LRUCache
from inference.core.utils.image_utils import load_image_from_url
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Load an image from a URL.

This block downloads an image from the provided URL and makes it available
for use in the workflow pipeline. Optionally, the block can cache downloaded
images to avoid re-fetching the same URL multiple times.
"""

# Module-level cache instance following common pattern
image_cache = LRUCache(capacity=64)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Load Image From URL",
            "version": "v1",
            "short_description": "Load an image from a URL.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fas fa-image",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/load_image_from_url@v1"]
    url: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="URL of the image to load",
        examples=["https://example.com/image.jpg", "$inputs.image_url"],
    )
    cache: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Whether to cache the downloaded image to avoid re-fetching",
        examples=[True, False, "$inputs.cache_image"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> str:
        return ">=1.0.0,<2.0.0"


class LoadImageFromUrlBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, url: str, cache: bool = True, **kwargs) -> BlockResult:
        try:
            # Generate cache key using URL hash (following common pattern)
            cache_key = hashlib.md5(url.encode("utf-8")).hexdigest()

            # Check cache if enabled
            if cache:
                cached_image = image_cache.get(cache_key)
                if cached_image is not None:
                    return {"image": cached_image}

            # Load image using secure utility
            numpy_image = load_image_from_url(value=url)

            # Create proper parent metadata
            parent_metadata = ImageParentMetadata(parent_id=str(uuid4()))

            workflow_image = WorkflowImageData(
                parent_metadata=parent_metadata,
                numpy_image=numpy_image,
            )

            # Store in cache if enabled
            if cache:
                image_cache.set(cache_key, workflow_image)

            return {"image": workflow_image}
        except Exception as e:
            raise RuntimeError(f"Failed to load image from URL {url}: {str(e)}")
