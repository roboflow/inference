"""Workflows cache selection: Redis-backed when ``REDIS_HOST`` is set.

Keys, value encoding and lock semantics are those of the legacy server's
``RedisCache``, so a Redis shared with a legacy server holds the same entries.
"""

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Type

from roboflow_workflows.utils.in_memory_cache import InMemoryWorkflowsCache

from inference_server import configuration

logger = logging.getLogger(__name__)


class RedisWorkflowsCache:
    """Redis-backed ``WorkflowsCache`` with the legacy lock semantics.

    Args:
        client: A connected ``redis.Redis`` client.
        lock_not_owned_error: The client library's error for releasing a lock
            whose TTL already expired; it is logged, not raised.
    """

    def __init__(
        self, client: Any, *, lock_not_owned_error: Type[BaseException]
    ) -> None:
        self.client = client
        self._lock_not_owned_error = lock_not_owned_error

    def get(self, key: str) -> Any:
        """Return the JSON-decoded value at ``key``, raw bytes when not JSON.

        Args:
            key: Cache key.

        Returns:
            The decoded value, or None when the key is absent or expired.
        """
        item = self.client.get(key)
        if item is None:
            return None
        try:
            return json.loads(item)
        except (TypeError, ValueError):
            return item

    def set(self, key: str, value: Any, expire: Optional[float] = None) -> None:
        """Store ``value`` JSON-encoded (bytes are stored as given).

        Args:
            key: Cache key.
            value: Value to store.
            expire: Seconds until the key expires; None keeps it.
        """
        if not isinstance(value, bytes):
            value = json.dumps(value)
        self.client.set(key, value, ex=expire)

    def acquire_lock(self, key: str, expire: Optional[float] = None) -> Any:
        """Acquire a Redis lock at ``key``, waiting up to ``expire`` seconds.

        Args:
            key: Lock key.
            expire: Lock TTL and acquisition wait, in seconds.

        Returns:
            The held lock.

        Raises:
            TimeoutError: If the lock could not be acquired in time.
        """
        lock = self.client.lock(key, blocking=True, timeout=expire)
        acquired = lock.acquire(blocking_timeout=expire)
        if not acquired:
            raise TimeoutError("Couldn't get lock")
        if expire is not None:
            lock.extend(expire)

        return lock

    @contextmanager
    def lock(self, key: str, expire: Optional[float] = None) -> Iterator[Any]:
        """Hold the lock at ``key`` for the duration of the block.

        Args:
            key: Lock key.
            expire: Lock TTL and acquisition wait, in seconds.

        Yields:
            The held lock.
        """
        lock = self.acquire_lock(key, expire=expire)
        try:
            yield lock
        finally:
            try:
                lock.release()
            except self._lock_not_owned_error:
                logger.warning("Lock at cache key %s expired before release", key)


def build_workflows_cache() -> Any:
    """Build the process-wide Workflows cache from the ``REDIS_*`` settings.

    Returns:
        A ``RedisWorkflowsCache`` when ``REDIS_HOST`` is set and reachable,
        otherwise an ``InMemoryWorkflowsCache``.
    """
    if configuration.REDIS_HOST is None:
        return InMemoryWorkflowsCache()
    try:
        import redis
    except ImportError:
        logger.error(
            "REDIS_HOST is set but the redis package is not installed. "
            "In-memory Workflows cache to be used."
        )
        return InMemoryWorkflowsCache()

    client = redis.Redis(
        host=configuration.REDIS_HOST,
        port=configuration.REDIS_PORT,
        db=0,
        decode_responses=False,
        ssl=configuration.REDIS_SSL,
        socket_timeout=configuration.REDIS_TIMEOUT,
        socket_connect_timeout=configuration.REDIS_TIMEOUT,
    )
    try:
        client.ping()
    except (redis.exceptions.TimeoutError, redis.exceptions.ConnectionError):
        logger.error(
            "Could not connect to Redis under %s:%s. In-memory Workflows cache "
            "to be used.",
            configuration.REDIS_HOST,
            configuration.REDIS_PORT,
        )
        return InMemoryWorkflowsCache()
    logger.info("Redis Workflows cache initialised")
    cache = RedisWorkflowsCache(
        client, lock_not_owned_error=redis.exceptions.LockNotOwnedError
    )

    return cache
