from inference_client.http.utils.iterables import (
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
