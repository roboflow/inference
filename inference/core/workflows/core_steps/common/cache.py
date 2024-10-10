from collections import deque
from threading import Lock
from typing import Any, Dict


class TrackedInstancesCache:

    def __init__(self, cache_size: int) -> None:
        self._cache_access_lock = Lock()
        self._keys_ingestion_order = deque(maxlen=cache_size)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def save(self, video_id: str, tracker_id: int, field: str, value: Any) -> None:
        key = hash_key(video_id=video_id, tracker_id=tracker_id)
        with self._cache_access_lock:
            if key in self._cache:
                self._cache[key][field] = value
                return None
            while len(self._cache) >= self._keys_ingestion_order.maxlen:
                to_remove = self._keys_ingestion_order.popleft()
                del self._cache[to_remove]
            self._cache[key] = {field: value}
            self._keys_ingestion_order.append(key)

    def get(self, video_id: str, tracker_id: int, field: str) -> Any:
        key = hash_key(video_id=video_id, tracker_id=tracker_id)
        return self._cache.get(key, {}).get(field)


def hash_key(video_id: str, tracker_id: int) -> str:
    return f"video_id:{video_id}:tracker_id:{tracker_id}"
