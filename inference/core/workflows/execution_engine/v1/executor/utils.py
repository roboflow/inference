from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Generator, Iterable, List, Optional, TypeVar

from inference.core.workflows.execution_engine.entities.base import Batch

T = TypeVar("T")


def run_steps_in_parallel(
    steps: List[Callable[[], T]],
    max_workers: int = 1,
    executor: Optional[ThreadPoolExecutor] = None,
) -> List[T]:
    if executor is None:
        with ThreadPoolExecutor(max_workers=max_workers) as inner_executor:
            return list(inner_executor.map(_run, steps))
    results = []
    for batch in create_batches(sequence=steps, batch_size=max_workers):
        batch_results = list(executor.map(_run, batch))
        results.extend(batch_results)
    return results


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


def resolve_futures(
    value: Any,
    context: str = "workflow_execution | future_resolution",
) -> Any:
    if isinstance(value, Future):
        return resolve_futures(value.result(), context=context)
    if isinstance(value, Batch):
        return Batch.init(
            content=[resolve_futures(element, context=context) for element in value],
            indices=value.indices,
        )
    if isinstance(value, list):
        return [resolve_futures(element, context=context) for element in value]
    if isinstance(value, tuple):
        return tuple(resolve_futures(element, context=context) for element in value)
    if isinstance(value, dict):
        return {
            key: resolve_futures(element, context=context)
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
    context: str = "workflow_execution | future_resolution",
) -> Any:
    if not contains_future(value):
        return value
    return resolve_futures(value=value, context=context)
