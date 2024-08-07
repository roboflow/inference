from multiprocessing.pool import ThreadPool
from typing import Callable, List, TypeVar

T = TypeVar("T")


def run_steps_in_parallel(
    steps: List[Callable[[], T]], max_workers: int = 1
) -> List[T]:
    with ThreadPool(processes=max_workers) as pool:
        return pool.map(func=_run_step, iterable=steps)


def _run_step(step: Callable[[], T]) -> T:
    return step()
