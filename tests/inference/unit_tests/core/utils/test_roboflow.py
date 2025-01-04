from typing import Any

import pytest

from inference.core.exceptions import InvalidModelIDError
from inference.core.utils.roboflow import get_model_id_chunks


@pytest.mark.parametrize("value", ["contains/2/slashes", "some/model/id/with/many/slashes"])
def test_get_model_id_chunks_when_invalid_input_given(value: Any) -> None:
    # when
    with pytest.raises(InvalidModelIDError):
        _ = get_model_id_chunks(model_id=value)


@pytest.mark.parametrize("model_id, expected", [
    ("some/1", ("some", "1")),
    ("modelid123", ("modelid123", None)),
    ("model-id-dashes", ("model-id-dashes", None)),
])
def test_get_model_id_chunks_with_various_valid_inputs(model_id: str, expected: tuple) -> None:
    # when
    result = get_model_id_chunks(model_id)

    # then
    assert result == expected
