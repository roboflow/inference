from typing import Generator, Iterable, List, TypeVar, Union

T = TypeVar("T")


def remove_empty_values(dictionary: dict) -> dict:
    """Remove empty values from a dictionary.

    Args:
        dictionary: The dictionary to remove empty values from.

    Returns:
        The dictionary with empty values removed.
    """
    return {k: v for k, v in dictionary.items() if v is not None}


def unwrap_single_element_list(sequence: List[T]) -> Union[T, List[T]]:
    """Unwrap a single element list.

    Args:
        sequence: The list to unwrap.

    Returns:
        The unwrapped list.
    """
    if len(sequence) == 1:
        return sequence[0]
    return sequence


def make_batches(
    iterable: Iterable[T], batch_size: int
) -> Generator[List[T], None, None]:
    """Make batches from an iterable.

    Args:
        iterable: The iterable to make batches from.
        batch_size: The size of the batches.

    Returns:
        The batches.
    """
    batch_size = max(batch_size, 1)
    batch = []
    for element in iterable:
        batch.append(element)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
