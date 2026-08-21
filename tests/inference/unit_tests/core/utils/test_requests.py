import pytest
from requests import HTTPError, Response

from inference.core.utils.requests import (
    API_KEY_PATTERN,
    api_key_safe_raise_for_status,
    deduct_api_key,
    deduct_api_key_from_string,
)


def test_deduce_api_key_when_no_api_key_available() -> None:
    # given
    url = "https://some.com/endpoint?param_1=12098xx87s&param_2=2982s8x"

    # when
    result = API_KEY_PATTERN.sub(deduct_api_key, url)

    # then
    assert result == url


def test_deduce_api_key_when_short_api_key_available() -> None:
    # given
    url = "https://some.com/endpoint?api_key=my_key&param_2=some_value"

    # when
    result = API_KEY_PATTERN.sub(deduct_api_key, url)

    # then
    assert result == "https://some.com/endpoint?api_key=***&param_2=some_value"


def test_deduce_api_key_when_single_long_api_key_available() -> None:
    # given
    url = "https://some.com/endpoint?api_key=19xjs9-XSXSAos0s&param_2=some_value"

    # when
    result = API_KEY_PATTERN.sub(deduct_api_key, url)

    # then
    assert result == "https://some.com/endpoint?api_key=19***0s&param_2=some_value"


def test_deduce_api_key_when_multiple_api_key_available() -> None:
    # given
    url = "https://some.com/endpoint?api_key=19xjs9-XSXSAos0s&param_2=some_value&api_key=my_key"

    # when
    result = API_KEY_PATTERN.sub(deduct_api_key, url)

    # then
    assert (
        result
        == "https://some.com/endpoint?api_key=19***0s&param_2=some_value&api_key=***"
    )


@pytest.mark.parametrize("status_code", [200, 201, 300])
def test_api_key_safe_raise_for_status_when_no_error_occurs(status_code: int) -> None:
    # given
    response = Response()
    response.status_code = status_code

    # when
    api_key_safe_raise_for_status(response=response)

    # then no error happens


@pytest.mark.parametrize(
    "status_code", [400, 401, 402, 403, 404, 500, 501, 502, 503, 504]
)
def test_api_keysafe_raise_for_status_when_error_occurs(status_code: int) -> None:
    # given
    response = Response()
    response.status_code = status_code
    response.url = (
        "https://some.com/endpoint?api_key=19xjs9-XSXSAos0s&param_2=some_value"
    )

    # when
    with pytest.raises(HTTPError) as expected_error:
        api_key_safe_raise_for_status(response=response)

    # then
    assert "https://some.com/endpoint?api_key=19***0s&param_2=some_value" in str(
        expected_error.value
    )


def test_deduct_api_key_from_string_redacts_url_embedded_key() -> None:
    # given - the shape requests.HTTPError / ConnectionError messages take
    value = (
        "500 Server Error: Internal Server Error for url: "
        "https://api.roboflow.com/inferenceproxy/sam3?api_key=fake12345678&foo=bar"
    )

    # when
    result = deduct_api_key_from_string(value=value)

    # then
    assert "fake12345678" not in result
    assert "api_key=fa***78" in result
    assert "foo=bar" in result


def test_deduct_api_key_from_string_leaves_keyless_text_untouched() -> None:
    # when
    result = deduct_api_key_from_string(value="connection refused to host x")

    # then
    assert result == "connection refused to host x"
