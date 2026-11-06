from typing import Any, Tuple

import pytest

from inference.core.exceptions import InvalidModelIDError
from inference.core.utils.roboflow import get_model_id_chunks


@pytest.mark.parametrize("value", ["some/2/invalid"])
def test_get_model_id_chunks_when_invalid_input_given(value: Any) -> None:
    # when
    with pytest.raises(InvalidModelIDError):
        _ = get_model_id_chunks(model_id=value)


@pytest.mark.parametrize(
    "value",
    ["some/other", "some-2/another-2", "dinov3/vits16", "siglip2/base-patch16-224"],
)
def test_get_model_id_chunks_when_instant_model_id_given(value: Any) -> None:
    # when
    result = get_model_id_chunks(model_id=value)

    # then
    assert result == (value, None)


@pytest.mark.parametrize(
    "value, expected", [("some/1", ("some", "1")), ("dinov3/12", ("dinov3", "12"))]
)
def test_get_model_id_chunks_when_valid_input_given(
    value: str, expected: Tuple[str, str]
) -> None:
    # when
    result = get_model_id_chunks(value)

    # then
    assert result == expected
