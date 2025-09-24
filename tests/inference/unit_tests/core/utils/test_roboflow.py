from typing import Any

import pytest

from inference.core.exceptions import InvalidModelIDError
from inference.core.utils.roboflow import get_model_id_chunks


@pytest.mark.parametrize("value", ["some/2/invalid"])
def test_get_model_id_chunks_when_invalid_input_given(value: Any) -> None:
    # when
    with pytest.raises(InvalidModelIDError):
        _ = get_model_id_chunks(model_id=value)


@pytest.mark.parametrize("value", ["some/other", "some-2/another-2"])
def test_get_model_id_chunks_when_instant_model_id_given(value: Any) -> None:
    # when
    result = get_model_id_chunks(model_id=value)

    # then
    assert result == (value, None)


def test_get_model_id_chunks_when_valid_input_given() -> None:
    # when
    result = get_model_id_chunks("some/1")

    # then
    assert result == ("some", "1")
