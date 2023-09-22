import asyncio
import inspect
import json
import threading
import time
from typing import Any, Optional

import redis

from inference.core.cache.base import BaseCache
from inference.core.data_models import InferenceResponseImage
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

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        """
        Initializes a new instance of the MemoryCache class.
        """
        self.client = redis.Redis(host=host, port=port, db=db)

        self.zexpires = dict()

        self._expire_thread = threading.Thread(target=self._expire)

    def _expire(self):
        """
        Removes the expired keys from the cache and zexpires dictionaries.

        This method runs in an infinite loop and sleeps for MEMORY_CACHE_EXPIRE_INTERVAL seconds between each iteration.
        """
        while True:
            now = time.time()
            for k, v in self.zexpires.items():
                if v < now:
                    self.zremrangebyscore(k[0], k[1], k[1])
                    del self.zexpires[k]
            while time.time() - now < MEMORY_CACHE_EXPIRE_INTERVAL:
                asyncio.sleep(0.01)

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
            return json.loads(item)

    def set(self, key: str, value: str, expire: float = None):
        """
        Sets a value for a given key with an optional expire time.

        Args:
            key (str): The key to store the value.
            value (str): The value to store.
            expire (float, optional): The time, in seconds, after which the key will expire. Defaults to None.
        """
        self.client.set(key, json.dumps(value), ex=expire)

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
        self.client.zadd(key, {json.dumps(value): score})
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
