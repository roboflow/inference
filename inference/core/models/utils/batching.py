from typing import Generator, Iterable, List, TypeVar, Union

B = TypeVar("B")


def calculate_input_elements(input_value: Union[B, List[B]]) -> int:
    return len(input_value) if issubclass(type(input_value), list) else 1


def create_batches(
    sequence: Iterable[B], batch_size: int
) -> Generator[List[B], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if len(current_batch) > 0:
        yield current_batch
