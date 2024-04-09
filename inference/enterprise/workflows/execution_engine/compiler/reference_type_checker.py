from typing import List

from inference.enterprise.workflows.entities.types import Kind


def validate_reference_types(
    expected: List[Kind],
    actual: List[Kind],
    error_message: str,
) -> None:
    expected_kind_names = set(e.name for e in expected)
    actual_kind_names = set(a.name for a in actual)
    if "*" in expected_kind_names or "*" in actual_kind_names:
        return None
    if len(expected_kind_names.intersection(actual_kind_names)) == 0:
        raise ValueError(error_message)
