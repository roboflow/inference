from inference_sdk.http.utils.iterables import (
    make_batches,
    remove_empty_values,
    unwrap_single_element_list,
)


def test_remove_empty_values_when_dictionary_is_empty() -> None:
    # when
    result = remove_empty_values(dictionary={})

    # then
    assert result == {}


def test_remove_empty_values_when_dictionary_has_no_empty_values() -> None:
    # given
    dictionary = {"some": "value_1", "other": "value_2"}

    # when
    result = remove_empty_values(dictionary=dictionary)

    # then
    assert result == dictionary
    assert result is not dictionary


def test_remove_empty_values_when_dictionary_has_empty_values() -> None:
    # given
    dictionary = {"some": "value_1", "other": None}

    # when
    result = remove_empty_values(dictionary=dictionary)

    # then
    assert result == {"some": "value_1"}


def test_unwrap_single_element_list_when_list_has_one_element() -> None:
    # when
    result = unwrap_single_element_list(sequence=[1])

    # then
    assert result == 1


def test_unwrap_single_element_list_when_list_has_no_elements() -> None:
    # when
    result = unwrap_single_element_list(sequence=[])

    # then
    assert result == []


def test_unwrap_single_element_list_when_list_has_multiple_elements() -> None:
    # given
    sequence = [1, 2, 3]

    # when
    result = unwrap_single_element_list(sequence=sequence)

    # then
    assert result is sequence


def test_make_batches_when_empty_input_provided() -> None:
    # when
    result = list(make_batches(iterable=[], batch_size=10))

    # then
    assert result == []


def test_make_batches_when_invalid_batch_size_provided() -> None:
    # when
    result = list(make_batches(iterable=[1, 2, 3], batch_size=0))

    # then
    assert result == [[1], [2], [3]]


def test_make_batches_when_non_empty_input_and_valid_batch_size_provided() -> None:
    # when
    result = list(make_batches(iterable=[1, 2, 3, 4, 5], batch_size=2))

    # then
    assert result == [[1, 2], [3, 4], [5]]
