from typing import List, TypeVar, Union

T = TypeVar("T")


def remove_empty_values(dictionary: dict) -> dict:
    return {k: v for k, v in dictionary.items() if v is not None}


def unwrap_single_element_list(sequence: List[T]) -> Union[T, List[T]]:
    if len(sequence) == 1:
        return sequence[0]
    return sequence
