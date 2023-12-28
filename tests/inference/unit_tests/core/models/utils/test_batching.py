import numpy as np

from inference.core.models.utils.batching import (
    calculate_input_elements,
    create_batches,
)


def test_calculate_input_elements_when_non_list_given() -> None:
    # given
    input_value = np.zeros((128, 128, 3))

    # when
    result = calculate_input_elements(input_value=input_value)

    # then
    assert result == 1, "Single element given, so the proper value is 1"


def test_calculate_input_elements_when_empty_list_given() -> None:
    # given
    input_value = []

    # when
    result = calculate_input_elements(input_value=input_value)

    # then
    assert result == 0, "No elements given, so the proper value is 0"


def test_calculate_input_elements_when_single_element_list_given() -> None:
    # given
    input_value = [np.zeros((128, 128, 3))]

    # when
    result = calculate_input_elements(input_value=input_value)

    # then
    assert result == 1, "Single element given, so the proper value is 1"


def test_calculate_input_elements_when_multi_elements_list_given() -> None:
    # given
    input_value = [np.zeros((128, 128, 3))] * 3

    # when
    result = calculate_input_elements(input_value=input_value)

    # then
    assert result == 3, "Three elements given, so the proper value is 3"


def test_create_batches_when_empty_sequence_given() -> None:
    # when
    result = list(create_batches(sequence=[], batch_size=4))

    # then
    assert result == []


def test_create_batches_when_not_allowed_batch_size_given() -> None:
    # when
    result = list(create_batches(sequence=[1, 2, 3], batch_size=0))

    # then
    assert result == [[1], [2], [3]]


def test_create_batches_when_batch_size_larger_than_sequence() -> None:
    # when
    result = list(create_batches(sequence=[1, 2], batch_size=4))

    # then
    assert result == [[1, 2]]


def test_create_batches_when_batch_size_multiplier_fits_sequence_length() -> None:
    # when
    result = list(create_batches(sequence=[1, 2, 3, 4], batch_size=2))

    # then
    assert result == [[1, 2], [3, 4]]


def test_create_batches_when_batch_size_multiplier_does_not_fir_sequence_length() -> (
    None
):
    # when
    result = list(create_batches(sequence=[1, 2, 3, 4], batch_size=3))

    # then
    assert result == [[1, 2, 3], [4]]
