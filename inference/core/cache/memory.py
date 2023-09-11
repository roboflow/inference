import time

from inference.core.cache.base import BaseCache


class MemoryCache(BaseCache):
    def __init__(self) -> None:
        self.cache = dict()
        self.expires = dict()

    def set(self, key: str, value: str, expire: float = None):
        self.cache[key] = value
        if expire:
            self.expires[key] = expire + time.time()

    def zadd(self, key: str, value: str, score: float):
        if not key in self.cache:
            self.cache[key] = dict()
        self.cache[key][score] = value

    def zrange(self, key: str, start: int, stop: int):
        if not key in self.cache:
            return []
        keys = sorted([k for k in self.cache[key].keys() if start <= k <= stop])
        return [(k, self.cache[key][k]) for k in keys]

    def get(self, key: str):
        if key in self.expires:
            if self.expires[key] < time.time():
                del self.cache[key]
                del self.expires[key]
                return None
        return self.cache.get(key)
