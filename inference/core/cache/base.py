from contextlib import contextmanager
from typing import Any, Optional

from inference.core import logger


class BaseCache:
    """
    BaseCache is an abstract base class that defines the interface for a cache.
    """

    def get(self, key: str):
        """
        Gets the value associated with the given key.

        Args:
            key (str): The key to retrieve the value.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def set(self, key: str, value: str, expire: float = None):
        """
        Sets a value for a given key with an optional expire time.

        Args:
            key (str): The key to store the value.
            value (str): The value to store.
            expire (float, optional): The time, in seconds, after which the key will expire. Defaults to None.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def zadd(self, key: str, value: str, score: float, expire: float = None):
        """
        Adds a member with the specified score to the sorted set stored at key.

        Args:
            key (str): The key of the sorted set.
            value (str): The value to add to the sorted set.
            score (float): The score associated with the value.
            expire (float, optional): The time, in seconds, after which the key will expire. Defaults to None.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

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
            start (int, optional): The starting index of the range. Defaults to -1.
            stop (int, optional): The ending index of the range. Defaults to float("inf").
            withscores (bool, optional): Whether to return the scores along with the values. Defaults to False.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def zremrangebyscore(
        self,
        key: str,
        start: Optional[int] = -1,
        stop: Optional[int] = float("inf"),
    ):
        """
        Removes all members in a sorted set within the given scores.

        Args:
            key (str): The key of the sorted set.
            start (int, optional): The minimum score of the range. Defaults to -1.
            stop (int, optional): The maximum score of the range. Defaults to float("inf").

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def acquire_lock(self, key: str, expire: float = None) -> Any:
        raise NotImplementedError()

    @contextmanager
    def lock(self, key: str, expire: float = None) -> Any:
        logger.debug(f"Acquiring lock at cache key: {key}")
        l = self.acquire_lock(key, expire=expire)
        try:
            yield l
        finally:
            logger.debug(f"Releasing lock at cache key: {key}")
            l.release()

    def set_numpy(self, key: str, value: Any, expire: float = None):
        """
        Caches a numpy array.

        Args:
            key (str): The key to store the value.
            value (Any): The value to store.
            expire (float, optional): The time, in seconds, after which the key will expire. Defaults to None.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def get_numpy(self, key: str) -> Any:
        """
        Retrieves a numpy array from the cache.

        Args:
            key (str): The key of the value to retrieve.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()
