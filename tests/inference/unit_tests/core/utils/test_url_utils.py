from unittest import mock
from urllib.parse import parse_qs, urlparse

import pytest

from inference.core.utils import url_utils
from inference.core.utils.url_utils import wrap_url


@mock.patch.object(url_utils, "SECURE_GATEWAY", "gateway.local")
def test_wrap_url_when_secure_gateway_is_provided() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    result = wrap_url(url=original_url)

    # then
    assert (
        result
        == "http://gateway.local/proxy?url=https%3A%2F%2Fdetection.roboflow.com%2Feye-detection%2F1%3Fapi_key%3DX"
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
    assert result.startswith("http://gateway.local:8080/proxy?url=")
    assert "//proxy" not in result
    assert wrap_url(url=result) == result


# `inference_models.weights_providers.roboflow.roboflow_secure_gateway_proxy_url_builder`
# implements the same `/proxy?url=` contract as `wrap_url`, for weights traffic rather
# than server traffic. They are separate packages and cannot share an implementation
# without a dependency between them, so parity is pinned here instead. A divergence is
# invisible at runtime: weights simply get fetched from a different place than
# everything else. See issue #2662.
@pytest.mark.parametrize(
    "gateway",
    [
        "gateway.local",
        "gateway.local:8080",
        "https://gateway.local",
        "https://gateway.local:8443/",
        "https://gw.local/edge",
        "https://gw.local/edge/",
    ],
)
def test_weights_proxy_builder_matches_wrap_url(gateway: str) -> None:
    # given
    roboflow_module = pytest.importorskip("inference_models.weights_providers.roboflow")
    original_url = "https://api.roboflow.com/models/v1/weights"

    # when
    with mock.patch.object(url_utils, "SECURE_GATEWAY", gateway), mock.patch.object(
        roboflow_module, "SECURE_GATEWAY", gateway
    ):
        from_server = wrap_url(url=original_url)
        from_weights = roboflow_module.roboflow_secure_gateway_proxy_url_builder(
            url=original_url, query=None
        )

        # then - identical output, and both idempotent
        assert from_weights == from_server
        assert wrap_url(url=from_server) == from_server
        assert (
            roboflow_module.roboflow_secure_gateway_proxy_url_builder(
                url=from_weights, query=None
            )
            == from_weights
        )
