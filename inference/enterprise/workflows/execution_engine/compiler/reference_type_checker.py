from typing import List

from inference.enterprise.workflows.entities.types import Kind
from inference.enterprise.workflows.errors import ReferenceTypeError


def validate_reference_kinds(
    expected: List[Kind],
    actual: List[Kind],
    error_message: str,
) -> None:
    expected_kind_names = set(e.name for e in expected)
    actual_kind_names = set(a.name for a in actual)
    if "*" in expected_kind_names or "*" in actual_kind_names:
        return None
    if len(expected_kind_names.intersection(actual_kind_names)) == 0:
        raise ReferenceTypeError(
            public_message=error_message,
            context="workflow_compilation | execution_graph_construction",
        )
