import functools
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from typing import Optional, List, Dict, Union, Generator, Deque, Any


class InferenceProfiler(ABC):

    @classmethod
    @abstractmethod
    def init(cls, **kwargs) -> "InferenceProfiler":
        pass

    @abstractmethod
    @contextmanager
    def profile_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> Generator[None, None, None]:
        pass

    @abstractmethod
    def start_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        pass

    @abstractmethod
    def end_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        pass

    @abstractmethod
    def notify_event(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        pass

    @abstractmethod
    def export_trace(self) -> List[dict]:
        pass


class NullInferenceProfiler(InferenceProfiler):

    @classmethod
    def init(cls, **kwargs) -> "NullInferenceProfiler":
        return cls()

    @contextmanager
    def profile_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> Generator[None, None, None]:
        yield None

    def start_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        pass

    def end_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        pass

    def notify_event(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        pass

    def export_trace(self) -> List[dict]:
        return []


class BaseInferenceProfiler(InferenceProfiler):

    @classmethod
    def init(
        cls,
        buffer_size: int = 16384,
        **kwargs,
    ) -> "BaseInferenceProfiler":
        buffer = deque(maxlen=buffer_size)
        return cls(buffer=buffer)

    def __init__(self, buffer: Deque[dict]):
        self._buffer = buffer

    @contextmanager
    def profile_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> Generator[None, None, None]:
        start_ts = round(time.monotonic() * 10**6)
        try:
            yield None
        except Exception as e:
            self.notify_event(
                name=f"{name}_error",
            )
            raise e
        finally:
            duration = round(time.monotonic() * 10**6) - start_ts
            self._add_event(
                name=name,
                event_type="X",
                categories=categories,
                metadata=metadata,
                extra_event_fields={"dur": duration},
                timestamp=start_ts,
            )

    def start_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        self._add_event(
            name=name, event_type="B", categories=categories, metadata=metadata
        )

    def end_execution_phase(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        self._add_event(
            name=name, event_type="E", categories=categories, metadata=metadata
        )

    def notify_event(
        self,
        name: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    ) -> None:
        self._add_event(
            name=name, event_type="I", categories=categories, metadata=metadata
        )

    def export_trace(self) -> List[dict]:
        return list(self._buffer)

    def _add_event(
        self,
        name: str,
        event_type: str,
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
        extra_event_fields: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        event = {
            "name": name,
            "ph": event_type,
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
        }
        if timestamp is not None:
            event["ts"] = timestamp
        else:
            event["ts"] = round(time.monotonic() * 10**6)
        if categories:
            event["cat"] = ",".join(categories)
        if metadata:
            event["args"] = metadata
        if extra_event_fields:
            for k, v in extra_event_fields.items():
                event[k] = v
        self._buffer.append(event)


def execution_phase(
    name: str,
    categories: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Union[str, int, float, bool, list, dict]]] = None,
    profiler_parameter: str = "profiler",
    runtime_metadata: Optional[List[str]] = None,
):
    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(kwargs.get(profiler_parameter), InferenceProfiler):
                return func(*args, **kwargs)
            profiler: InferenceProfiler = kwargs[profiler_parameter]
            actual_metadata = metadata or {}
            if runtime_metadata is not None:
                runtime_metadata_dict = {k: kwargs.get(k) for k in runtime_metadata}
                actual_metadata.update(runtime_metadata_dict)
            with profiler.profile_execution_phase(
                name=name,
                categories=categories,
                metadata=actual_metadata,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator
