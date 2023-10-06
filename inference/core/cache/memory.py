import threading
import time
from typing import Any, Optional

from inference.core.cache.base import BaseCache
from inference.core.env import MEMORY_CACHE_EXPIRE_INTERVAL


class MemoryCache(BaseCache):
    """
    MemoryCache is an in-memory cache that implements the BaseCache interface.

    Attributes:
        cache (dict): A dictionary to store the cache values.
        expires (dict): A dictionary to store the expiration times of the cache values.
        zexpires (dict): A dictionary to store the expiration times of the sorted set values.
        _expire_thread (threading.Thread): A thread that runs the _expire method.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the MemoryCache class.
        """
        self.cache = dict()
        self.expires = dict()
        self.zexpires = dict()

        self._expire_thread = threading.Thread(target=self._expire)
        self._expire_thread.daemon = True
        self._expire_thread.start()

    def _expire(self):
        """
        Removes the expired keys from the cache and zexpires dictionaries.

        This method runs in an infinite loop and sleeps for MEMORY_CACHE_EXPIRE_INTERVAL seconds between each iteration.
        """
        while True:
            now = time.time()
            keys_to_delete = []
            for k, v in self.expires.copy().items():
                if v < now:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del self.cache[k]
                del self.expires[k]
            keys_to_delete = []
            for k, v in self.zexpires.copy().items():
                if v < now:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del self.cache[k[0]][k[1]]
                del self.zexpires[k]
            while time.time() - now < MEMORY_CACHE_EXPIRE_INTERVAL:
                time.sleep(0.1)

    def get(self, key: str):
        """
        Gets the value associated with the given key.

        Args:
            key (str): The key to retrieve the value.

        Returns:
            str: The value associated with the key, or None if the key does not exist or is expired.
        """
        if key in self.expires:
            if self.expires[key] < time.time():
                del self.cache[key]
                del self.expires[key]
                return None
        return self.cache.get(key)

    def set(self, key: str, value: str, expire: float = None):
        """
        Sets a value for a given key with an optional expire time.

        Args:
            key (str): The key to store the value.
            value (str): The value to store.
            expire (float, optional): The time, in seconds, after which the key will expire. Defaults to None.
        """
        self.cache[key] = value
        if expire:
            self.expires[key] = expire + time.time()

    def zadd(self, key: str, value: Any, score: float, expire: float = None):
        """
        Adds a member with the specified score to the sorted set stored at key.

        Args:
            key (str): The key of the sorted set.
            value (str): The value to add to the sorted set.
            score (float): The score associated with the value.
            expire (float, optional): The time, in seconds, after which the key will expire. Defaults to None.
        """
        if not key in self.cache:
            self.cache[key] = dict()
        self.cache[key][score] = value
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
        if not key in self.cache:
            return []
        keys = sorted([k for k in self.cache[key].keys() if min <= k <= max])
        if withscores:
            return [(self.cache[key][k], k) for k in keys]
        else:
            return [self.cache[key][k] for k in keys]

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
        res = self.zrangebyscore(key, min=min, max=max, withscores=True)
        keys_to_delete = [k[1] for k in res]
        for k in keys_to_delete:
            del self.cache[key][k]
        return len(keys_to_delete)
