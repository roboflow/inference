import time
from threading import Lock

from typing_extensions import Any, Dict, List, Optional

from inference.core.cache import cache
from inference.core.cache.redis import RedisCache
from inference.core.logger import logger


class RedisQueue:
    """
    Store and forget, keys with specified prefix are handled by external service
    """

    def __init__(
        self,
        prefix: str = f"UsageCollector:{time.time()}",
        redis_cache: Optional[RedisCache] = None,
    ):
        self._prefix: str = prefix
        self._redis_cache: RedisCache = redis_cache or cache
        self._increment: int = 0
        self._lock: Lock = Lock()

    def put(self, payload: Any):
        with self._lock:
            try:
                self._increment += 1
                self._redis_cache.zadd(
                    key=f"{self._prefix}:{self._increment}",
                    value=payload,
                    score=time.time(),
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
