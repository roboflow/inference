"""The offline platform client and the structural ``PlatformClient`` port."""

from typing import Any, Dict, List

import pytest
from roboflow_workflows.v2.blocks.platform import (
    OFFLINE_PLATFORM_CLIENT,
    OfflinePlatformClient,
    PlatformClient,
    PlatformUnavailableError,
)

SENTINEL = "sk-test-SENTINEL-7f3a"


class FakePlatformClient:
    """Records requests; a host-supplied client of the V1 shape."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []

    def post(
        self,
        endpoint,
        api_key,
        payload=None,
        params=None,
        http_errors_handlers=None,
    ) -> dict:
        self.requests.append({"endpoint": endpoint, "api_key": api_key})
        return {"status": "ok"}

    def build_api_headers(self, explicit_headers=None):
        return {"X-Host": "fake", **(explicit_headers or {})}

    def build_weights_provider_headers(self, countinference=None, service_secret=None):
        return None

    def wrap_url(self, url: str) -> str:
        return f"https://gateway.test/?u={url}"


def test_offline_post_refuses_and_names_the_resource_but_not_the_key() -> None:
    # when
    with pytest.raises(PlatformUnavailableError) as raised:
        OFFLINE_PLATFORM_CLIENT.post(
            "apiproxy/email", api_key=SENTINEL, payload={"to": "a@b.test"}
        )

    # then
    assert "'platform_client'" in str(raised.value)
    assert "apiproxy/email" in str(raised.value)
    assert SENTINEL not in str(raised.value)


def test_offline_helpers_work_without_a_platform() -> None:
    # given
    client = OfflinePlatformClient()
    explicit = {"Accept": "json"}

    # when
    headers = client.build_api_headers(explicit)

    # then
    assert headers == explicit
    assert headers is not explicit
    assert client.build_api_headers() == {}
    assert client.build_weights_provider_headers(countinference=True) is None
    assert client.wrap_url("https://example.test/x") == "https://example.test/x"
    assert repr(client) == "OfflinePlatformClient()"


def test_a_host_client_with_the_v1_shape_fits_the_port() -> None:
    # given
    client: PlatformClient = FakePlatformClient()

    # when
    response = client.post("apiproxy/email", api_key=SENTINEL, payload={})

    # then
    assert response == {"status": "ok"}
    assert client.requests == [{"endpoint": "apiproxy/email", "api_key": SENTINEL}]


def test_the_port_declares_exactly_the_v1_methods() -> None:
    # when
    declared = {
        name
        for name in vars(PlatformClient)
        if not name.startswith("_") and callable(getattr(PlatformClient, name))
    }

    # then
    assert declared == {
        "post",
        "build_api_headers",
        "build_weights_provider_headers",
        "wrap_url",
    }
    assert all(hasattr(OfflinePlatformClient, name) for name in declared)
