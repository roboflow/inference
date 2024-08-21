import json
import time
from threading import Lock
from uuid import uuid4

from typing_extensions import Any, Dict, List, Optional

from inference.core.cache import cache
from inference.core.cache.redis import RedisCache
from inference.core.logger import logger


class RedisQueue:
    """
    Store and forget, keys with specified hash tag are handled by external service
    """

    def __init__(
        self,
        hash_tag: str = "UsageCollector",
        redis_cache: Optional[RedisCache] = None,
    ):
        # prefix must contain hash-tag to avoid CROSSLOT errors when using mget
        # hash-tag is common part of the key wrapped within '{}'
        # removing hash-tag will cause clients utilizing mget to fail
        self._prefix: str = f"{{{hash_tag}}}:{uuid4().hex[:5]}:{time.time()}"
        self._redis_cache: RedisCache = redis_cache or cache
        self._increment: int = 0
        self._lock: Lock = Lock()

    def put(self, payload: Any):
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload)
            except Exception as exc:
                logger.error("Failed to parse payload '%s' to JSON - %s", payload, exc)
                return
        with self._lock:
            try:
                self._increment += 1
                redis_key = f"{self._prefix}:{self._increment}"
                self._redis_cache.client.set(
                    name=redis_key,
                    value=payload,
                )
                self._redis_cache.client.zadd(
                    name="UsageCollector",
                    mapping={redis_key: time.time()},
                )
            except Exception as exc:
                logger.error("Failed to store usage records '%s', %s", payload, exc)

    @staticmethod
    def full() -> bool:
        return False

    def empty(self) -> bool:
        return True

    def get_nowait(self) -> List[Dict[str, Any]]:
        return []
