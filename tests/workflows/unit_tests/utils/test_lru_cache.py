from inference.core.cache.lru_cache import LRUCache as ServerLRUCache
from inference.core.workflows.utils.lru_cache import LRUCache

OPS = [
    ("set", "a", 1),
    ("set", "b", 2),
    ("get", "a", None),
    ("set", "c", 3),
    ("get", "b", None),
    ("get", "c", None),
    ("set_max_size", 1, None),
    ("get", "c", None),
    ("get", "a", None),
    ("get", "missing", None),
]


def _run(cache):
    seen = []
    for op, key, value in OPS:
        if op == "set":
            cache.set(key, value)
        elif op == "get":
            seen.append((key, cache.get(key)))
        else:
            cache.set_max_size(key)
    return seen


def test_copy_matches_the_server_lru_cache_exactly() -> None:
    assert _run(LRUCache(capacity=2)) == _run(ServerLRUCache(capacity=2))


def test_copy_carries_set_max_size() -> None:
    # visualizations/grid/v1.py:189 calls set_max_size(len(images) + 1).
    cache = LRUCache(capacity=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set_max_size(1)
    assert cache.capacity == 1
    assert len(cache.cache) == 1
