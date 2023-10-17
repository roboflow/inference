from typing import Any

import pytest

from inference.core.exceptions import InvalidModelIDError
from inference.core.registries.roboflow import get_model_id_chunks


@pytest.mark.parametrize("value", ["some", "some/2/invalid", "another-2"])
def test_get_model_id_chunks_when_invalid_input_given(value: Any) -> None:
    # when
    with pytest.raises(InvalidModelIDError):
        _ = get_model_id_chunks(model_id=value)


def test_get_model_id_chunks_when_valid_input_given() -> None:
    # when
    result = get_model_id_chunks("some/1")

    # then
    assert result == ("some", "1")
