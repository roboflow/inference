"""In-process object cache: ports of V1 ``roboflow_core/cache_get@v1`` and
``roboflow_core/cache_set@v1``.

``ObjectCache`` is a constructor resource, separate from the portable JSON
``managed_state``. It holds any Python object, by reference, in one process::

    v2/cache_set@v1   value ──▶ cache[namespace][key] ──▶ output = value
    v2/cache_get@v1   cache[namespace][key] ──▶ output   (False when missing)

Cache or managed state? Both keep values between runs::

    cache_get / cache_set    any Python object, by reference, in this process;
                             scope "source" by default; missing key -> False
    state_get / state_set    portable JSON values, copied; scope "global" by
                             default; missing key -> the step's ``default``

Use the cache for objects such as images or predictions. Use managed state
for JSON values that need a real missing-key default.

A missing key and a stored ``False`` both read ``False``, as in V1. To tell
them apart, store a value that is never ``False``, for example ``{"value": x}``.

Namespace, chosen per call::

    scope="global"   one namespace shared by every source
    scope="source"   (default, as V1) image.video_metadata.video_identifier,
                     else the image input's SampleContext.source_id,
                     else "default"

Images with neither a video identifier nor a source, such as a plain batch
of uploaded images, all share the ``"default"`` namespace. One key then
holds one value for all of them: every image reads the last image's write.
Frame delay also uses this fallback when video metadata is present.
Without video metadata, it rejects the image because it needs frame numbers.

Only the ``image`` input's own context names the source, not the context
the call's inputs share: a ``value`` or ``after`` without a context, such as
a workflow parameter, keeps the image's source. This picks where an entry
lives; output contexts follow ``Output.source`` as for any block.

Who shares one cache::

    create_session()                         own cache per session
    create_session({"object_cache": cache})  this cache; pass the same
                                             instance to share it across sessions

The per-session default comes from the catalogue provider
``{"object_cache": Factory(ObjectCache)}``. A constructor default would give
``cache_get`` and ``cache_set`` two different caches. Lifetime::

    runs, graph updates            keep the cache
    update(..., reset=True)        per-session cache: created again, empty
                                   caller cache: kept, never cleared
    session.close()                caller cache: kept

Only ``ObjectCache.clear`` removes entries from a cache that is kept. The
cache has no size limit or expiry, as in V1: every new key and source adds
an entry that stays. A long-running deployment with changing keys or
sources should call ``clear`` itself.

Ownership of stored values:

- An ``ImageData`` value is stored as ``value.copy()``: pixels the cache owns,
  so a camera or decoder buffer can be reused after the pulse.
- Every other value is stored by reference, as in V1. This includes images
  inside lists, dicts or predictions: the cache does not copy them, so the
  caller must not hand it a borrowed buffer it plans to reuse.
- ``cache_get`` returns the stored object itself, also the stored image copy.
  A later in-place change by any reader changes the cached value. The blocks
  declare this (``retains_inputs``, ``retains_outputs``), so the compiler's
  mutation check reports such a step.

Ordering: two cache steps in one run are ordered only by the graph. Bind the
getter's ``after`` to ``$steps.<setter>.output`` to read this run's write.
``value`` and ``after`` accept a present ``None`` (a stored ``None`` is a
value); a filtered or missing upstream value still skips the step.
The lock keeps the dictionary consistent between threads; it does not order
steps, and two cache steps are never one transaction. A pipeline with more
than one pulse in flight orders each step across pulses, not a get/set pair:
pulse 2's getter can run before pulse 1's setter. Use ``max_in_flight=1``
when every pulse must read the previous pulse's write.
"""

from threading import Lock
from typing import Any, Dict, Literal, NamedTuple, Optional

from pydantic import Field, StrictStr
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)

__all__ = [
    "CacheGetBlock",
    "CacheNamespace",
    "CacheScope",
    "CacheSetBlock",
    "DEFAULT_SOURCE",
    "GLOBAL_NAMESPACE",
    "ObjectCache",
]

CacheScope = Literal["global", "source"]

DEFAULT_SOURCE = "default"
"""Source namespace of a call with neither a video identifier nor a source."""


class CacheNamespace(NamedTuple):
    """Where a cache entry lives.

    Args:
        scope: ``global`` or ``source``.
        source: The source name for ``source``; ``None`` for ``global``.
    """

    scope: CacheScope
    source: Optional[str] = None


GLOBAL_NAMESPACE = CacheNamespace(scope="global")


class ObjectCache:
    """Thread-safe in-process store of Python objects, by namespace and key.

    Values are kept by reference until replaced or cleared. There is no
    expiry, size limit or eviction. One lock guards the dictionary, so
    concurrent ``get``, ``set`` and ``clear`` calls each see a consistent
    state; a sequence of calls is not atomic.
    """

    def __init__(self) -> None:
        self._namespaces: Dict[CacheNamespace, Dict[str, Any]] = {}
        self._lock = Lock()

    def get(self, namespace: CacheNamespace, key: str, default: Any = None) -> Any:
        """Return the stored value, or ``default`` when the key is missing.

        Args:
            namespace: Namespace of the entry.
            key: Entry key.
            default: Value for a missing key. A stored ``None`` is returned
                as ``None``.

        Returns:
            The stored object itself (not a copy), or ``default``.
        """
        with self._lock:
            entries = self._namespaces.get(namespace, {})
            value = entries.get(key, default)

        return value

    def set(self, namespace: CacheNamespace, key: str, value: Any) -> None:
        """Store ``value`` by reference, replacing any earlier value.

        Args:
            namespace: Namespace of the entry.
            key: Entry key.
            value: Any object.
        """
        with self._lock:
            self._namespaces.setdefault(namespace, {})[key] = value

    def clear(
        self, *, scope: Optional[CacheScope] = None, source: Optional[str] = None
    ) -> int:
        """Remove entries; nothing else ever removes them.

        ``clear()`` empties the cache, ``clear(scope="global")`` the global
        namespace, ``clear(scope="source")`` every source namespace and
        ``clear(scope="source", source="cam-1")`` one source namespace.

        Args:
            scope: Scope to clear; ``None`` clears both.
            source: One source to clear; needs ``scope="source"``.

        Returns:
            Number of removed entries.

        Raises:
            ValueError: When ``source`` is given without ``scope="source"``,
                or ``scope`` is unknown.
        """
        if scope not in (None, "global", "source"):
            raise ValueError(f"scope must be 'global', 'source' or None, got {scope!r}")
        if source is not None and scope != "source":
            raise ValueError("clearing one source needs scope='source'")

        with self._lock:
            cleared = [
                namespace
                for namespace in self._namespaces
                if (scope is None or namespace.scope == scope)
                and (source is None or namespace.source == source)
            ]
            removed = sum(len(self._namespaces.pop(name)) for name in cleared)

        return removed


_SCOPE_DESCRIPTION = (
    "'source' (default, as V1): one namespace per video identifier of the "
    "image, else per source of the image's own sample context, else "
    "'default'. Images without a video identifier or a source share the "
    "'default' namespace. 'global': one namespace shared by every source."
)
_IMAGE_DESCRIPTION = (
    "Image of this call; its video identifier names the 'source' namespace. "
    "Also sets the cadence: the block runs once per image."
)
_KEY_DESCRIPTION = "Cache key; exact, case-sensitive match."


class _CacheBlock(Block):
    """Shared constructor and namespace rule of the cache blocks."""

    def __init__(self, *, object_cache: ObjectCache) -> None:
        self._object_cache = object_cache

    def _namespace(self, scope: CacheScope, image: ImageData) -> CacheNamespace:
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


class CacheSetBlock(_CacheBlock):
    """Store any value under a key and pass it through.

    An ``ImageData`` value is stored as an owned copy; any other value by
    reference (images nested inside it included). Entries stay until
    ``ObjectCache.clear``. For portable JSON values with a missing-key
    default, use ``v2/state_set`` instead.
    """

    type = "v2/cache_set@v1"
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
            description=_IMAGE_DESCRIPTION, examples=["$inputs.image"]
        )
        key: StrictStr | Ref(STRING_KIND) = Field(
            description=_KEY_DESCRIPTION + " An existing value is replaced.",
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
        scope: CacheScope = Field(default="source", description=_SCOPE_DESCRIPTION)
        after: None | Ref(allow_none=True) = Field(
            default=None,
            description=(
                "Optional upstream value to wait for, such as a cache_get "
                "output; its value is ignored, None included. The step is "
                "skipped when that value is filtered or missing."
            ),
        )

    def run(
        self,
        *,
        image: ImageData,
        key: str,
        value: Any,
        scope: CacheScope,
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
        self._object_cache.set(self._namespace(scope, image), key, stored)

        return {"output": value}


class CacheGetBlock(_CacheBlock):
    """Read a value stored by ``v2/cache_set@v1``; ``False`` when missing.

    The output is the stored object itself. Do not modify it in place unless
    every later reader should see the change. A stored ``False`` reads the
    same as a missing key. For portable JSON values with a chosen missing-key
    default, use ``v2/state_get`` instead.
    """

    type = "v2/cache_get@v1"
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
            description=_IMAGE_DESCRIPTION, examples=["$inputs.image"]
        )
        key: StrictStr | Ref(STRING_KIND) = Field(
            description=_KEY_DESCRIPTION + " A missing key outputs False.",
            examples=["detections", "$inputs.cache_key"],
        )
        scope: CacheScope = Field(default="source", description=_SCOPE_DESCRIPTION)
        after: None | Ref(allow_none=True) = Field(
            default=None,
            description=(
                "Optional upstream value to wait for, such as "
                "$steps.<cache_set step>.output, so this read sees that write "
                "in the same run. Its value is ignored, None included. The "
                "step is skipped when that value is filtered or missing."
            ),
        )

    def run(
        self, *, image: ImageData, key: str, scope: CacheScope, after: Any
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
        value = self._object_cache.get(self._namespace(scope, image), key, False)

        return {"output": value}
