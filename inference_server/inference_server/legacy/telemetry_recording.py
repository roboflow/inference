import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from inference_model_manager.pipelines import InvalidPipelineIdError

from inference_models.errors import BaseInferenceModelsError
from inference_server import telemetry

_STAT_ERROR_WRAPPERS = (PermissionError, LookupError, RuntimeError)


def _recorded_error_type(error: BaseException) -> str:
    cause = error.__cause__
    if type(error) in _STAT_ERROR_WRAPPERS and isinstance(
        cause, (BaseInferenceModelsError, InvalidPipelineIdError)
    ):
        return type(cause).__name__

    return type(error).__name__


RECORDED_LOAD_EVENTS: contextvars.ContextVar[Optional[set[int]]] = (
    contextvars.ContextVar("recorded_load_events", default=None)
)


@contextmanager
def request_telemetry_scope() -> Iterator[None]:
    if RECORDED_LOAD_EVENTS.get() is not None:
        yield
        return

    token = RECORDED_LOAD_EVENTS.set(set())
    try:
        yield
    finally:
        RECORDED_LOAD_EVENTS.reset(token)


def record_telemetry(record: Callable[..., None], *args: Any) -> None:
    try:
        record(*args)
    except Exception:
        pass


def record_route_error(error: Exception) -> None:
    record_telemetry(telemetry.record_error, error)
    record_telemetry(telemetry.record_error_metric, _recorded_error_type(error))
