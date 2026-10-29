"""Read a value stored by ``roboflow_core/cache_set@v1``; ``False`` when missing.

Canonical type: ``roboflow_core/cache_get@v1``.
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


class CacheGetBlock(Block):
    """Read a value stored by ``roboflow_core/cache_set@v1``; ``False`` when missing.

    The output is the stored object itself. Do not modify it in place unless
    every later reader should see the change. A stored ``False`` reads the
    same as a missing key. For portable JSON values with a chosen missing-key
    default, use ``roboflow_core/state_get@v1`` instead.
    """

    type = "roboflow_core/cache_get@v1"
    metadata = {
        "name": "Cache Get",
        "port_of": "roboflow_core/cache_get@v1",
        "section": "advanced",
    }
    outputs = {
        "output": Output(
            source="image",
            description=(
                "Stored value; False when the key is missing. A stored False "
                "reads the same."
            ),
        ),
    }
    retains_outputs = ("output",)

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
                "Cache key; exact, case-sensitive match. A missing key outputs False."
            ),
            examples=["detections", "$inputs.cache_key"],
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
                "Optional upstream value to wait for, such as "
                "$steps.<cache_set step>.output, so this read sees that write "
                "in the same run. Its value is ignored, None included. The "
                "step is skipped when that value is filtered or missing."
            ),
        )

    def __init__(self, *, object_cache: ObjectCache) -> None:
        self._object_cache = object_cache

    def run(
        self,
        *,
        image: ImageData,
        key: str,
        scope: Literal["global", "source"],
        after: Any,
    ) -> Dict[str, Any]:
        """Read ``key``.

        Args:
            image: Image of this call; names the ``source`` namespace.
            key: Cache key.
            scope: ``source`` or ``global``.
            after: Ignored; only orders this step.

        Returns:
            ``output``: the stored object, ``None`` when ``None`` was stored,
            or ``False`` when the key is missing.
        """
        namespace = self._namespace(scope, image)
        value = self._object_cache.get(namespace, key, False)

        return {"output": value}

    def _namespace(
        self, scope: Literal["global", "source"], image: ImageData
    ) -> CacheNamespace:
        if scope == "global":
            return GLOBAL_NAMESPACE

        metadata = image.video_metadata
        if metadata is not None and metadata.video_identifier:
            return CacheNamespace(scope="source", source=metadata.video_identifier)

        # The image's own context: another input without one (``after``) must
        # not drop it from the call's context.
        context = self.execution_context
        sample = context.sample_at(context.indices[0], field="image")
        source = sample.source_id if sample is not None else DEFAULT_SOURCE
        namespace = CacheNamespace(scope="source", source=source)

        return namespace
