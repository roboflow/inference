"""Managed state of V2 workflows: global and per-source atomic key-value state.

Blocks request it as the constructor resource ``managed_state`` (see
``ManagedState``). The default store is in memory and private to the
session; ``state.redis.RedisStateBackend`` shares state across processes
through the same API. Importing this package never imports ``redis`` and
starts no thread or connection.

``MANAGED_STATE_RESOURCE`` is the reserved V2 resource name.
"""

from roboflow_workflows.execution_engine.v2.state.api import (
    MANAGED_STATE_RESOURCE,
    MISSING,
    ManagedState,
    StateBackend,
    StateScope,
)
from roboflow_workflows.execution_engine.v2.state.errors import (
    StateBackendError,
    StateError,
    StateOutcomeUnknownError,
    StateOverflowError,
    StateScopeError,
    StateTypeError,
    StateValueError,
)
from roboflow_workflows.execution_engine.v2.state.memory import InMemoryStateBackend

__all__ = [
    "InMemoryStateBackend",
    "MANAGED_STATE_RESOURCE",
    "MISSING",
    "ManagedState",
    "StateBackend",
    "StateBackendError",
    "StateError",
    "StateOutcomeUnknownError",
    "StateOverflowError",
    "StateScope",
    "StateScopeError",
    "StateTypeError",
    "StateValueError",
]
