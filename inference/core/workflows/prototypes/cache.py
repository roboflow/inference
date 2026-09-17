from typing import Any, Optional, Protocol


class WorkflowsCache(Protocol):
    """Shared key/value cache injected into Workflow blocks.

    Only ``get`` and ``set`` are declared: that is the entire surface the 9
    calling blocks use. ``inference.core.cache.base.BaseCache`` also offers
    sorted-set and lock operations - if a block ever needs one, add it here
    first.

    The server's implementation is Redis-backed when ``REDIS_HOST`` is set and
    honours ``expire``. The standalone default is
    ``inference.core.workflows.utils.in_memory_cache.InMemoryWorkflowsCache``;
    every server composition root overrides it with ``workflows_core.cache``
    (see ``inference/core/interfaces/roboflow_platform_client.py``).
    """

    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: Any, expire: Optional[float] = None) -> None: ...
