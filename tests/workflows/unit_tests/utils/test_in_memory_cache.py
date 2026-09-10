"""The standalone cache default.

Deliberately not a copy of `inference.core.cache.memory.MemoryCache`: that one
starts a daemon expiry thread and reads MEMORY_CACHE_EXPIRE_INTERVAL from the
server env. Expiry here is lazy - checked on read.

The server treats a FALSY `expire` as "no expiry" (`memory.py:84`, `if expire:`),
so `expire=0` must not expire the value.
"""

import inspect
import time

from inference.core.cache.memory import MemoryCache
from inference.core.workflows.prototypes.cache import WorkflowsCache
from inference.core.workflows.utils.in_memory_cache import InMemoryWorkflowsCache


def test_get_returns_none_for_a_missing_key() -> None:
    assert InMemoryWorkflowsCache().get("nope") is None


def test_set_then_get_round_trips() -> None:
    cache = InMemoryWorkflowsCache()
    cache.set(key="k", value={"a": 1})
    assert cache.get("k") == {"a": 1}


def test_an_expired_value_is_gone() -> None:
    cache = InMemoryWorkflowsCache()
    cache.set(key="k", value="v", expire=0.01)
    time.sleep(0.05)
    assert cache.get("k") is None


def test_overwriting_clears_a_previous_deadline() -> None:
    cache = InMemoryWorkflowsCache()
    cache.set(key="k", value="old", expire=0.01)
    cache.set(key="k", value="new")
    time.sleep(0.05)
    assert cache.get("k") == "new"


def test_it_satisfies_the_workflows_cache_port() -> None:
    for method in ("get", "set"):
        port = list(inspect.signature(getattr(WorkflowsCache, method)).parameters)
        real = list(
            inspect.signature(getattr(InMemoryWorkflowsCache, method)).parameters
        )
        assert port == real, f"{method}: {port} != {real}"


def test_it_matches_the_server_cache_on_every_expire_shape() -> None:
    reference = MemoryCache()
    candidate = InMemoryWorkflowsCache()
    for cache in (reference, candidate):
        cache.set(key="none", value=1, expire=None)
        cache.set(key="zero", value=2, expire=0)
        cache.set(key="omitted", value=3)
        cache.set(key="long", value=4, expire=30)
        cache.set(key="short", value=5, expire=0.01)
    time.sleep(0.05)
    for key, expected in (
        ("none", 1),
        ("zero", 2),
        ("omitted", 3),
        ("long", 4),
        ("short", None),
        ("missing", None),
    ):
        assert reference.get(key) == expected, f"reference disagrees on {key}"
        assert candidate.get(key) == expected, f"candidate disagrees on {key}"
