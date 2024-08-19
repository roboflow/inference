import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, TypeVar

T = TypeVar("T")


def run_steps_in_parallel(
    steps: List[Callable[[], T]], max_workers: int = 1
) -> List[T]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run, steps))


def _run(fun: Callable[[], T]) -> T:
    return fun()
