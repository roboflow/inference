from typing import List

import pytest

from inference.enterprise.workflows.entities.types import (
    INTEGER_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    Kind,
)
from inference.enterprise.workflows.errors import ReferenceTypeError
from inference.enterprise.workflows.execution_engine.compiler.reference_type_checker import (
    validate_reference_kinds,
)


def test_validate_reference_types_when_there_is_strict_kinds_match() -> None:
    # when
    validate_reference_kinds(
        expected=[STRING_KIND],
        actual=[STRING_KIND],
        error_message="some error",
    )

    # then - no error


@pytest.mark.parametrize(
    "expected, actual",
    [
        ([STRING_KIND, INTEGER_KIND], [INTEGER_KIND]),
        ([INTEGER_KIND], [STRING_KIND, INTEGER_KIND]),
    ],
)
def test_validate_reference_types_when_kinds_matches_by_union(
    expected: List[Kind],
    actual: List[Kind],
) -> None:
    # when
    validate_reference_kinds(
        expected=expected,
        actual=actual,
        error_message="some error",
    )

    # then - no error


@pytest.mark.parametrize(
    "expected, actual",
    [
        ([STRING_KIND, INTEGER_KIND], [WILDCARD_KIND]),
        ([WILDCARD_KIND], [STRING_KIND, INTEGER_KIND]),
        ([WILDCARD_KIND], [WILDCARD_KIND]),
    ],
)
def test_validate_reference_types_when_kinds_matches_by_wild_card(
    expected: List[Kind],
    actual: List[Kind],
) -> None:
    # when
    validate_reference_kinds(
        expected=expected,
        actual=actual,
        error_message="some error",
    )

    # then - no error


def test_validate_reference_kinds_when_there_is_no_match_in_kinds() -> None:
    # when
    with pytest.raises(ReferenceTypeError):
        validate_reference_kinds(
            expected=[STRING_KIND],
            actual=[INTEGER_KIND],
            error_message="some error",
        )
