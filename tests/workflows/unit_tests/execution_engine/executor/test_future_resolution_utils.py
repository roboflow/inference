from concurrent.futures import Future

import pytest

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.execution_engine.v1.executor.utils import (
    contains_future,
    maybe_resolve_futures,
    resolve_future_result,
    resolve_futures,
)


def _completed_future(value):
    future = Future()
    future.set_result(value)
    return future


def _failed_future(error):
    future = Future()
    future.set_exception(error)
    return future


def test_contains_future_detects_future_inside_batch() -> None:
    batch = Batch.init(content=[_completed_future("resolved")], indices=[(0,)])

    assert contains_future(batch) is True


def test_contains_future_returns_false_for_resolved_batch() -> None:
    batch = Batch.init(content=["resolved"], indices=[(0,)])

    assert contains_future(batch) is False


def test_resolve_futures_resolves_future_inside_batch() -> None:
    batch = Batch.init(
        content=[_completed_future("first"), _completed_future("second")],
        indices=[(0,), (1,)],
    )

    resolved = resolve_futures(batch)

    assert isinstance(resolved, Batch)
    assert list(resolved) == ["first", "second"]
    assert resolved.indices == [(0,), (1,)]


def test_resolve_futures_resolves_batch_nested_in_dict() -> None:
    value = {
        "predictions": Batch.init(
            content=[_completed_future({"frame": 0})],
            indices=[(0,)],
        )
    }

    resolved = resolve_futures(value)

    resolved_batch = resolved["predictions"]
    assert isinstance(resolved_batch, Batch)
    assert list(resolved_batch) == [{"frame": 0}]
    assert resolved_batch.indices == [(0,)]


def test_maybe_resolve_futures_is_noop_without_futures() -> None:
    value = Batch.init(content=[{"frame": 0}], indices=[(0,)])

    resolved = maybe_resolve_futures(value)

    assert resolved is value


def test_resolve_futures_propagates_future_exception() -> None:
    resolution_error = RuntimeError("future resolution failed")
    batch = Batch.init(content=[_failed_future(resolution_error)], indices=[(0,)])

    with pytest.raises(RuntimeError) as error_info:
        resolve_futures(batch)

    assert error_info.value is resolution_error


def test_resolve_future_result_raises_on_timeout() -> None:
    future = Future()

    with pytest.raises(ExecutionEngineRuntimeError, match="Timed out"):
        resolve_future_result(
            future,
            context="test | future_resolution",
            timeout=0.001,
        )


def test_resolve_futures_raises_on_timeout() -> None:
    future = Future()
    batch = Batch.init(content=[future], indices=[(0,)])

    with pytest.raises(ExecutionEngineRuntimeError, match="Timed out"):
        resolve_futures(batch, timeout=0.001)
