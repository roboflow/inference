import contextvars
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from typing import Any, Callable, Generator, Iterable, List, Optional, TypeVar

from roboflow_workflows.environment import WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT
from roboflow_workflows.errors import ExecutionEngineRuntimeError
from roboflow_workflows.execution_engine.entities.base import Batch

T = TypeVar("T")


def run_steps_in_parallel(
    steps: List[Callable[[], T]],
    max_workers: int = 1,
    executor: Optional[ThreadPoolExecutor] = None,
    flat_dispatch: bool = False,
) -> List[T]:
    steps = [wrap_with_context_snapshot(step) for step in steps]
    if executor is None:
        with ThreadPoolExecutor(max_workers=max_workers) as inner_executor:
            return list(inner_executor.map(_run, steps))
    if flat_dispatch:
        # Dedicated (run-owned) pool: submit the whole wave in one flat map,
        # exactly like the historical executor=None branch - no chunk
        # barriers. Shared host pools must keep batched dispatch so one
        # wide wave cannot monopolise them.
        #
        # Failure path: the historical per-wave `with` block ran
        # `shutdown(wait=True)` before a step error could leave this
        # function, so callers between here and `_run_workflow`'s `finally`
        # (profiler phases, execution-phase closes) never observed a
        # still-running sibling. Reproduce that by cancelling not-yet-
        # started futures and draining in-flight ones before propagating.
        # `CancelledError` is a `BaseException` (3.8+), so it must be
        # caught explicitly - `except Exception` would mask the step error.
        # The run-owned pool is NOT shut down here; its creator owns it.
        futures = [executor.submit(_run, step) for step in steps]
        try:
            return [future.result() for future in futures]
        finally:
            for future in futures:
                future.cancel()
            for future in futures:
                try:
                    future.exception()
                except CancelledError:
                    pass
    results = []
    for batch in create_batches(sequence=steps, batch_size=max_workers):
        batch_results = list(executor.map(_run, batch))
        results.extend(batch_results)
    return results


def wrap_with_context_snapshot(fun: Callable[[], T]) -> Callable[[], T]:
    """Bind ``fun`` to run inside its own copy of the caller's context.

    ``ThreadPoolExecutor`` workers do not inherit the submitting thread's
    ``contextvars`` state, and pool threads are reused across requests. Taking
    a fresh snapshot per task and entering it with ``Context.run`` gives every
    task the caller's context - any ``ContextVar``, not a hand-picked list -
    while guaranteeing a reused worker thread cannot retain state a previous
    task set: each snapshot is a throwaway ``Context`` scoped to this one
    call. A separate snapshot is required per task because the same
    ``Context`` object cannot be entered concurrently.
    """
    ctx = contextvars.copy_context()
    return lambda: ctx.run(fun)


def create_batches(
    sequence: Iterable[T], batch_size: int
) -> Generator[List[T], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if len(current_batch) > 0:
        yield current_batch


def _run(fun: Callable[[], T]) -> T:
    return fun()


def resolve_future_result(
    future: Future,
    *,
    context: str,
    timeout: float = WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT,
) -> Any:
    try:
        return future.result(timeout=timeout)
    except TimeoutError as error:
        raise ExecutionEngineRuntimeError(
            public_message=(
                "Timed out while resolving an asynchronous workflow future."
            ),
            context=context,
            inner_error=error,
        ) from error


def resolve_futures(
    value: Any,
    timeout: float = WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT,
    context: str = "workflow_execution | future_resolution",
) -> Any:
    if isinstance(value, Future):
        return resolve_futures(
            resolve_future_result(value, context=context, timeout=timeout),
            timeout=timeout,
            context=context,
        )
    if isinstance(value, Batch):
        return Batch.init(
            content=[
                resolve_futures(element, timeout=timeout, context=context)
                for element in value
            ],
            indices=value.indices,
        )
    if isinstance(value, list):
        return [
            resolve_futures(element, timeout=timeout, context=context)
            for element in value
        ]
    if isinstance(value, tuple):
        return tuple(
            resolve_futures(element, timeout=timeout, context=context)
            for element in value
        )
    if isinstance(value, dict):
        return {
            key: resolve_futures(element, timeout=timeout, context=context)
            for key, element in value.items()
        }
    return value


def contains_future(value: Any) -> bool:
    if isinstance(value, Future):
        return True
    if isinstance(value, Batch):
        return any(contains_future(element) for element in value)
    if isinstance(value, (list, tuple)):
        return any(contains_future(element) for element in value)
    if isinstance(value, dict):
        return any(contains_future(element) for element in value.values())
    return False


def maybe_resolve_futures(
    value: Any,
    timeout: float = WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT,
    context: str = "workflow_execution | future_resolution",
) -> Any:
    if not contains_future(value):
        return value
    return resolve_futures(value=value, timeout=timeout, context=context)
