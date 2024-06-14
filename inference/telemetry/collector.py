import asyncio
import atexit
from collections import defaultdict
from queue import Queue
from threading import Lock
import time
from functools import wraps
from typing import Any, Callable, DefaultDict, List, Union

from .config import get_telemetry_settings, TelemetrySettings


class UsageCollector:
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if not hasattr(cls, "_instance"):
                cls._instance = super().__new__(cls)
                cls._instance._queue = None
        return cls._instance

    def __init__(self):
        with UsageCollector._lock:
            if self._queue:
                return

        self._settings: TelemetrySettings = get_telemetry_settings()
        self._stats = self._create_empty_stats()
        # TODO: use persistent queue, i.e. https://pypi.org/project/persist-queue/
        self._queue = Queue(maxsize=self._settings.queue_size)

        # TODO: collect list of blocks where telemetry is compulsory
        self._blocks_with_compulsory_telemetry = set()

        atexit.register(self._cleanup)

    @staticmethod
    def _create_empty_stats() -> DefaultDict[str, Union[float, List[float]]]:
        return defaultdict(
            lambda: {"min_time": float("inf"), "max_time": 0, "executions": []}
        )

    def _collect_stats(
        self,
        func_name: str,
        duration: float,
    ):
        with UsageCollector._lock:
            func_stats = self._stats[func_name]
            func_stats["min_time"] = min(func_stats["min_time"], duration)
            func_stats["max_time"] = max(func_stats["max_time"], duration)
            func_stats["executions"].append(duration)

    def _enqueue(self):
        with UsageCollector._lock:
            self._queue.put(self._stats)
            self._stats = self._create_empty_stats()

    def _cleanup(self):
        with UsageCollector._lock:
            if self._stats:
                self._enqueue()

    def __call__(self, func: Callable[[Any], Any]):
        if (
            self._settings.opt_out
            and func.__name__ not in self._blocks_with_compulsory_telemetry
        ):
            return func

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            t1 = time.perf_counter()
            result = func(*args, **kwargs)
            t2 = time.perf_counter()

            self._collect_stats(
                func_name=func.__name__,
                duration=t2 - t1,
            )

            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            t1 = time.perf_counter()
            result = await func(*args, **kwargs)
            t2 = time.perf_counter()

            self._collect_stats(
                func_name=func.__name__,
                duration=t2 - t1,
            )

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


collector = UsageCollector()
