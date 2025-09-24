import hashlib
from collections import deque
from threading import Lock
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError

V = TypeVar("V")


class BasicWorkflowsCache(Generic[V]):
    """
    Base cache which is capable of hashing compound payloads based on
    list of injected hash functions. Hash functions are to produce stable hashing strings.
    Each function is invoked on `get_hash_key(...)` kwarg (use named args only!),
    output string is concatenated and md5 value is calculated.

    Cache is size bounded, each entry lives until `cache_size` new entries appear.

    Raises `WorkflowEnvironmentConfigurationError` when `get_hash_key(...)` is not
    provided with params corresponding to all hash functions.

    Thread safe thanks to thread lock on `get(...)` and `cache(...)`.
    """

    def __init__(
        self,
        cache_size: int,
        hash_functions: List[Tuple[str, Callable[[Any], str]]],
    ):
        self._keys_buffer = deque(maxlen=max(cache_size, 1))
        self._cache: Dict[str, V] = {}
        self._hash_functions = hash_functions
        self._cache_lock = Lock()

    def get_hash_key(self, **kwargs) -> str:
        hash_chunks = []
        for key_name, hashing_function in self._hash_functions:
            if key_name not in kwargs:
                raise WorkflowEnvironmentConfigurationError(
                    public_message=f"Cache is miss configured.",
                    context="workflows_cache | hash_key_generation",
                )
            hash_value = hashing_function(kwargs[key_name])
            hash_chunks.append(hash_value)
        return hashlib.md5("<|>".join(hash_chunks).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[V]:
        with self._cache_lock:
            return self._cache.get(key)

    def cache(self, key: str, value: V) -> None:
        with self._cache_lock:
            if len(self._keys_buffer) == self._keys_buffer.maxlen:
                to_pop = self._keys_buffer.popleft()
                del self._cache[to_pop]
            self._keys_buffer.append(key)
            self._cache[key] = value
