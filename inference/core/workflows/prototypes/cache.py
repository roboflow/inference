from typing import Any, Optional, Protocol


class WorkflowsCache(Protocol):
    """Shared key/value cache injected into Workflow blocks.

    Only ``get`` and ``set`` are declared: that is the entire surface the 9
    calling blocks use. ``inference.core.cache.base.BaseCache`` also offers
    sorted-set and lock operations - if a block ever needs one, add it here
    first.

    The server's implementation is Redis-backed when ``REDIS_HOST`` is set and
    honours ``expire``. There is deliberately NO default implementation here:
    see the Phase 4 preamble - the initializer stays bound to the server's
    cache until Phase 9 injects it at the composition roots.
    """

    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: Any, expire: Optional[float] = None) -> None: ...
