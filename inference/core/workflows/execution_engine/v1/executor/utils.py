from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generator, Iterable, List, Optional, TypeVar

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
