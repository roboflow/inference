import asyncio
import inspect
import json
import pickle
import threading
import time
from contextlib import asynccontextmanager
from copy import copy
from typing import Any, Optional

import redis

from inference.core import logger
from inference.core.cache.base import BaseCache
from inference.core.entities.responses.inference import InferenceResponseImage
from inference.core.env import MEMORY_CACHE_EXPIRE_INTERVAL


class RedisCache(BaseCache):
    """
    MemoryCache is an in-memory cache that implements the BaseCache interface.

    Attributes:
        cache (dict): A dictionary to store the cache values.
        expires (dict): A dictionary to store the expiration times of the cache values.
        zexpires (dict): A dictionary to store the expiration times of the sorted set values.
        _expire_thread (threading.Thread): A thread that runs the _expire method.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ssl: bool = False,
        timeout: float = 2.0,
    ) -> None:
        """
        Initializes a new instance of the MemoryCache class.
        """
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False,
            ssl=ssl,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
        )
        logger.debug("Attempting to diagnose Redis connection...")
        self.client.ping()
        logger.debug("Redis connection established.")
        self.zexpires = dict()

        self._expire_thread = threading.Thread(target=self._expire, daemon=True)
        self._expire_thread.start()

    def _expire(self):
        """
        Removes the expired keys from the cache and zexpires dictionaries.

        This method runs in an infinite loop and sleeps for MEMORY_CACHE_EXPIRE_INTERVAL seconds between each iteration.
        """
        while True:
            logger.debug("Redis cleaner thread starts cleaning...")
            now = time.time()
            for k, v in copy(list(self.zexpires.items())):
                if v < now:
                    tolerance_factor = 1e-14  # floating point accuracy
                    self.zremrangebyscore(
                        k[0], k[1] - tolerance_factor, k[1] + tolerance_factor
                    )
                    del self.zexpires[k]
            logger.debug("Redis cleaner finished task.")
            sleep_time = MEMORY_CACHE_EXPIRE_INTERVAL - (time.time() - now)
            time.sleep(max(sleep_time, 0))

    def get(self, key: str):
        """
        Gets the value associated with the given key.

        Args:
            key (str): The key to retrieve the value.

        Returns:
            str: The value associated with the key, or None if the key does not exist or is expired.
        """
        item = self.client.get(key)
        if item is not None:
            try:
                return json.loads(item)
            except (TypeError, ValueError):
                return item

    def set(self, key: str, value: str, expire: float = None):
        """
        Sets a value for a given key with an optional expire time.

        Args:
            key (str): The key to store the value.
            value (str): The value to store.
            expire (float, optional): The time, in seconds, after which the key will expire. Defaults to None.
        """
        if not isinstance(value, bytes):
            value = json.dumps(value)
        self.client.set(key, value, ex=expire)

    def zadd(self, key: str, value: Any, score: float, expire: float = None):
        """
        Adds a member with the specified score to the sorted set stored at key.

        Args:
            key (str): The key of the sorted set.
            value (str): The value to add to the sorted set.
            score (float): The score associated with the value.
            expire (float, optional): The time, in seconds, after which the key will expire. Defaults to None.
        """
        # serializable_value = self.ensure_serializable(value)
        value = json.dumps(value)
        self.client.zadd(key, {value: score})
        if expire:
            self.zexpires[(key, score)] = expire + time.time()

    def zrangebyscore(
        self,
        key: str,
        min: Optional[float] = -1,
        max: Optional[float] = float("inf"),
        withscores: bool = False,
    ):
        """
        Retrieves a range of members from a sorted set.

        Args:
            key (str): The key of the sorted set.
            start (int, optional): The starting score of the range. Defaults to -1.
            stop (int, optional): The ending score of the range. Defaults to float("inf").
            withscores (bool, optional): Whether to return the scores along with the values. Defaults to False.

        Returns:
            list: A list of values (or value-score pairs if withscores is True) in the specified score range.
        """
        res = self.client.zrangebyscore(key, min, max, withscores=withscores)
        if withscores:
            return [(json.loads(x), y) for x, y in res]
        else:
            return [json.loads(x) for x in res]

    def zremrangebyscore(
        self,
        key: str,
        min: Optional[float] = -1,
        max: Optional[float] = float("inf"),
    ):
        """
        Removes all members in a sorted set within the given scores.

        Args:
            key (str): The key of the sorted set.
            start (int, optional): The minimum score of the range. Defaults to -1.
            stop (int, optional): The maximum score of the range. Defaults to float("inf").

        Returns:
            int: The number of members removed from the sorted set.
        """
        return self.client.zremrangebyscore(key, min, max)

    def ensure_serializable(self, value: Any):
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, Exception):
                    value[k] = str(v)
                elif inspect.isclass(v) and isinstance(v, InferenceResponseImage):
                    value[k] = v.dict()
        return value

    def acquire_lock(self, key: str, expire=None) -> Any:
        l = self.client.lock(key, blocking=True, timeout=expire)
        acquired = l.acquire(blocking_timeout=expire)
        if not acquired:
            raise TimeoutError("Couldn't get lock")
        # refresh the lock
        if expire is not None:
            l.extend(expire)
        return l

    def set_numpy(self, key: str, value: Any, expire: float = None):
        serialized_value = pickle.dumps(value)
        self.set(key, serialized_value, expire=expire)

    def get_numpy(self, key: str) -> Any:
        serialized_value = self.get(key)
        if serialized_value is not None:
            return pickle.loads(serialized_value)
        else:
            return None
