from unittest import mock
from urllib.parse import parse_qs, urlparse

from inference.core.utils import url_utils
from inference.core.utils.url_utils import get_secure_gateway_base_url, wrap_url


@mock.patch.object(url_utils, "SECURE_GATEWAY", "gateway.local")
def test_wrap_url_when_secure_gateway_is_provided() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    result = wrap_url(url=original_url)

    # then
    assert (
        result
        == "https://gateway.local/proxy?url=https%3A%2F%2Fdetection.roboflow.com%2Feye-detection%2F1%3Fapi_key%3DX"
    )
    assert parse_qs(urlparse(result).query)["url"][0] == original_url


@mock.patch.object(url_utils, "SECURE_GATEWAY", None)
def test_wrap_url_when_secure_gateway_is_not_provided() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    result = wrap_url(url=original_url)

    # then
    assert result == original_url


@mock.patch.object(url_utils, "SECURE_GATEWAY", "https://gateway.local")
def test_wrap_url_when_secure_gateway_is_scheme_qualified() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    result = wrap_url(url=original_url)

    # then
    assert (
        result
        == "https://gateway.local/proxy?url=https%3A%2F%2Fdetection.roboflow.com%2Feye-detection%2F1%3Fapi_key%3DX"
    )
    assert parse_qs(urlparse(result).query)["url"][0] == original_url


@mock.patch.object(url_utils, "SECURE_GATEWAY", "https://gateway.local/")
def test_wrap_url_when_scheme_qualified_secure_gateway_has_trailing_slash() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    result = wrap_url(url=original_url)

    # then
    assert (
        result
        == "https://gateway.local/proxy?url=https%3A%2F%2Fdetection.roboflow.com%2Feye-detection%2F1%3Fapi_key%3DX"
    )


@mock.patch.object(url_utils, "SECURE_GATEWAY", "https://gateway.local")
def test_wrap_url_is_idempotent() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    wrapped_once = wrap_url(url=original_url)
    wrapped_twice = wrap_url(url=wrapped_once)

    # then
    assert wrapped_twice == wrapped_once
    assert parse_qs(urlparse(wrapped_twice).query)["url"][0] == original_url


@mock.patch.object(url_utils, "SECURE_GATEWAY", "gateway.local:8080/")
def test_wrap_url_when_bare_host_secure_gateway_has_trailing_slash() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    result = wrap_url(url=original_url)

    # then - no double slash in the proxy path, idempotence still holds
    assert result.startswith("https://gateway.local:8080/proxy?url=")
    assert "//proxy" not in result
    assert wrap_url(url=result) == result


@mock.patch.object(url_utils, "SECURE_GATEWAY", None)
def test_get_secure_gateway_base_url_when_not_configured() -> None:
    # when
    result = get_secure_gateway_base_url()

    # then
    assert result is None


@mock.patch.object(url_utils, "SECURE_GATEWAY", "")
def test_get_secure_gateway_base_url_when_configured_empty() -> None:
    # when
    result = get_secure_gateway_base_url()

    # then
    assert result is None


@mock.patch.object(url_utils, "SECURE_GATEWAY", "gateway.local")
def test_get_secure_gateway_base_url_when_bare_host_is_provided() -> None:
    # when
    result = get_secure_gateway_base_url()

    # then - bare hosts default to TLS
    assert result == "https://gateway.local"


@mock.patch.object(url_utils, "SECURE_GATEWAY", "10.9.50.228:8080")
def test_get_secure_gateway_base_url_when_bare_host_with_port_is_provided() -> None:
    # when
    result = get_secure_gateway_base_url()

    # then
    assert result == "https://10.9.50.228:8080"


@mock.patch.object(url_utils, "SECURE_GATEWAY", "https://gateway.local")
def test_get_secure_gateway_base_url_when_scheme_qualified() -> None:
    # when
    result = get_secure_gateway_base_url()

    # then
    assert result == "https://gateway.local"


@mock.patch.object(url_utils, "SECURE_GATEWAY", "https://gateway.local/")
def test_get_secure_gateway_base_url_strips_trailing_slash() -> None:
    # when
    result = get_secure_gateway_base_url()

    # then
    assert result == "https://gateway.local"


def test_gateway_scheme_policy_is_shared_with_model_provider():
    import pytest

    from inference_models.weights_providers.roboflow import (
        validate_secure_gateway_url as model_validate,
    )

    for validate in [url_utils.validate_secure_gateway_url, model_validate]:
        for gateway in [
            "http://gateway.local",
            "http://10.0.0.1",
            "http://127.0.0.1.example",
            "ftp://gateway.local",
            "https://user:secret@gateway.local",
            "https://gateway.local?token=x",
            "https://gateway.local\n",
            "https://gateway.local#fragment",
        ]:
            with pytest.raises(ValueError):
                validate(gateway)
        assert validate("gateway.local:443/edge/") == "https://gateway.local:443/edge"
        for gateway in [
            "http://127.0.0.1:8080",
            "http://[::1]:8080",
            "http://localhost:8080",
        ]:
            assert validate(gateway) == gateway
