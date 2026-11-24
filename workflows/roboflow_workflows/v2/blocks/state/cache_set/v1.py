"""Store any value under a key and pass it through.

Canonical type: ``roboflow_core/cache_set@v1``.
"""

from typing import Any, Dict, Literal
from pydantic import Field, StrictStr
from roboflow_workflows.v2.blocks.state._shared.object_cache import (
    DEFAULT_SOURCE,
    GLOBAL_NAMESPACE,
    CacheNamespace,
    ObjectCache,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class CacheSetBlock(Block):
    """Store any value under a key and pass it through.

    An ``ImageData`` value is stored as an owned copy; any other value by
    reference (images nested inside it included). Entries stay until
    ``ObjectCache.clear``. For portable JSON values with a missing-key
    default, use ``roboflow_core/state_set@v1`` instead.
    """

    type = "roboflow_core/cache_set@v1"
    metadata = {
        "name": "Cache Set",
        "port_of": "roboflow_core/cache_set@v1",
        "section": "advanced",
    }
    outputs = {
        "output": Output(
            source="value", description="The stored value, passed through as given."
        ),
    }
    retains_inputs = ("value",)

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Image of this call; its video identifier names the 'source' "
                "namespace. Also sets the cadence: the block runs once per image."
            ),
            examples=["$inputs.image"],
        )
        key: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "Cache key; exact, case-sensitive match. An existing value is "
                "replaced."
            ),
            examples=["detections", "$inputs.cache_key"],
        )
        value: Ref(allow_none=True) = Field(
            description=(
                "Any value to store, None included. Images are copied; other "
                "values are kept by reference, so later in-place changes reach "
                "the cache. A filtered or missing upstream value skips the step."
            ),
            examples=["$steps.model.predictions"],
        )
        scope: Literal["global", "source"] = Field(
            default="source",
            description=(
                "'source' (default, as V1): one namespace per video identifier of "
                "the image, else per source of the image's own sample context, "
                "else 'default'. Images without a video identifier or a source "
                "share the 'default' namespace. 'global': one namespace shared by "
                "every source."
            ),
        )
        after: None | Ref(allow_none=True) = Field(
            default=None,
            description=(
                "Optional upstream value to wait for, such as a cache_get "
                "output; its value is ignored, None included. The step is "
                "skipped when that value is filtered or missing."
            ),
        )

    def __init__(self, *, object_cache: ObjectCache) -> None:
        self._object_cache = object_cache

    def run(
        self,
        *,
        image: ImageData,
        key: str,
        value: Any,
        scope: Literal["global", "source"],
        after: Any,
    ) -> Dict[str, Any]:
        """Store ``value`` under ``key``.

        Args:
            image: Image of this call; names the ``source`` namespace.
            key: Cache key.
            value: Value to store.
            scope: ``source`` or ``global``.
            after: Ignored; only orders this step.

        Returns:
            ``output``: ``value`` as given (not the stored image copy).
        """
        stored = value.copy() if isinstance(value, ImageData) else value
        namespace = self._namespace(scope, image)
        self._object_cache.set(namespace, key, stored)

        return {"output": value}

    def _namespace(
        self, scope: Literal["global", "source"], image: ImageData
    ) -> CacheNamespace:
        if scope == "global":
            return GLOBAL_NAMESPACE

        metadata = image.video_metadata
        if metadata is not None and metadata.video_identifier:
            return CacheNamespace(scope="source", source=metadata.video_identifier)

        # The image's own context: another input without one (a scalar
        # ``value`` or ``after``) must not drop it from the call's context.
        context = self.execution_context
        sample = context.sample_at(context.indices[0], field="image")
        source = sample.source_id if sample is not None else DEFAULT_SOURCE
        namespace = CacheNamespace(scope="source", source=source)

        return namespace
