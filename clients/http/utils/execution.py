from typing import Iterable, TypeVar, Callable, List
from multiprocessing.pool import ThreadPool

T = TypeVar("T")
V = TypeVar("V")


def run_in_multithreading(
    iterable: Iterable[T],
    func: Callable[[T], V],
    max_workers: int,
) -> List[V]:
    with ThreadPool(processes=max_workers) as pool:
        return pool.map(func=func, iterable=iterable)
