from typing import Generator, Iterable, List, TypeVar, Union

T = TypeVar("T")


def remove_empty_values(dictionary: dict) -> dict:
    return {k: v for k, v in dictionary.items() if v is not None}


def unwrap_single_element_list(sequence: List[T]) -> Union[T, List[T]]:
    if len(sequence) == 1:
        return sequence[0]
    return sequence


def make_batches(
    iterable: Iterable[T], batch_size: int
) -> Generator[List[T], None, None]:
    batch_size = max(batch_size, 1)
    batch = []
    for element in iterable:
        batch.append(element)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
