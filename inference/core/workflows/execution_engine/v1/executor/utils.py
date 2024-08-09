import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, TypeVar

T = TypeVar("T")


def run_steps_in_parallel(
    steps: List[Callable[[], T]], max_workers: int = 1
) -> List[T]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(step) for step in steps]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results
