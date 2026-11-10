"""In-process object cache used by the ``roboflow_core/cache_get@v1`` and
``roboflow_core/cache_set@v1`` block definitions (ports of the V1 blocks).

``ObjectCache`` is a constructor resource, separate from the portable JSON
``managed_state``. It holds any Python object, by reference, in one process::

    roboflow_core/cache_set@v1   value ──▶ cache[namespace][key] ──▶ output = value
    roboflow_core/cache_get@v1   cache[namespace][key] ──▶ output   (False when missing)

Cache or managed state? Both keep values between runs::

    cache_get / cache_set    any Python object, by reference, in this process;
                             scope "source" by default; missing key -> False
    state_get / state_set    portable JSON values, copied; scope "global" by
                             default; missing key -> the step's ``default``

Use the cache for objects such as images or predictions. Use managed state
for JSON values that need a real missing-key default.

A missing key and a stored ``False`` both read ``False``, as in V1. To tell
them apart, store a value that is never ``False``, for example ``{"value": x}``.

Namespace, chosen per call by each block's own ``_namespace`` method::

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

__all__ = [
    "CacheNamespace",
    "CacheScope",
    "DEFAULT_SOURCE",
    "GLOBAL_NAMESPACE",
    "ObjectCache",
]

CacheScope = Literal["global", "source"]

DEFAULT_SOURCE = "default"


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
