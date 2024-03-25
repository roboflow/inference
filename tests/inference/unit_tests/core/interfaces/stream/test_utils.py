import pytest

from inference.core.interfaces.stream.utils import broadcast_elements, wrap_in_list


def test_wrap_in_list_when_list_provided() -> None:
    # given
    element = [1, 2, 3]

    # when
    result = wrap_in_list(element=element)

    # then
    assert result == [1, 2, 3], "Order of elements must be preserved"
    assert result is element, "The same object should be returned"


def test_wrap_in_list_when_single_element_provided() -> None:
    # given
    element = 1

    # when
    result = wrap_in_list(element=element)

    # then
    assert result == [1], "Expected to wrap element with list"


def test_broadcast_elements_when_desired_length_matches_elements() -> None:
    # given
    element = [1, 2, 3]

    # when
    result = broadcast_elements(
        elements=element, desired_length=3, error_description="some"
    )

    # then
    assert result == [1, 2, 3], "Order of elements must be preserved"
    assert result is element, "The same object should be returned"


def test_broadcast_elements_when_desired_length_do_not_match_elements() -> None:
    # given
    element = [1, 2, 3]

    # when
    with pytest.raises(ValueError):
        _ = broadcast_elements(
            elements=element, desired_length=4, error_description="some"
        )


def test_broadcast_elements_when_desired_length_do_not_match_elements_but_can_be_broadcast() -> (
    None
):
    # given
    element = [1]

    # when
    result = broadcast_elements(
        elements=element, desired_length=3, error_description="some"
    )

    # then
    assert result == [1, 1, 1]


def test_broadcast_elements_when_input_is_empty() -> None:
    # given
    element = []

    # when
    with pytest.raises(ValueError):
        _ = broadcast_elements(
            elements=element, desired_length=3, error_description="some"
        )
