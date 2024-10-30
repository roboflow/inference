from typing import List, Union

from inference.core.workflows.errors import ReferenceTypeError
from inference.core.workflows.execution_engine.entities.types import Kind


def validate_reference_kinds(
    expected: List[Union[Kind, str]],
    actual: List[Union[Kind, str]],
    error_message: str,
) -> None:
    expected_kind_names = set(_get_kind_name(kind=e) for e in expected)
    actual_kind_names = set(_get_kind_name(kind=a) for a in actual)
    if "*" in expected_kind_names or "*" in actual_kind_names:
        return None
    if len(expected_kind_names.intersection(actual_kind_names)) == 0:
        raise ReferenceTypeError(
            public_message=error_message,
            context="workflow_compilation | execution_graph_construction",
        )


def _get_kind_name(kind: Union[Kind, str]) -> str:
    if isinstance(kind, Kind):
        return kind.name
    return kind
