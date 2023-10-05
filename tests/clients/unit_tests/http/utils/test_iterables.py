from clients.http.utils.iterables import remove_empty_values


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
