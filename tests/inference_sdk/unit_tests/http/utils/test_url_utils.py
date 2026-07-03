import socket

import pytest
import requests
from urllib3.connectionpool import HTTPSConnectionPool

from inference_sdk import config
from inference_sdk.http.utils import url_utils
from inference_sdk.http.utils.url_utils import (
    InvalidURLImageInput,
    SSRFProtectedHTTPAdapter,
    URLAddressNotAllowedError,
    address_is_global,
    fetch_url_bytes,
    resolve_and_validate_ips,
    validate_url_destination,
)


def _fake_getaddrinfo(ip: str):
    def _inner(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]

    return _inner


@pytest.mark.parametrize(
    "address, expected",
    [
        ("8.8.8.8", True),
        ("127.0.0.1", False),
        ("10.0.0.1", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("::1", False),
        ("::ffff:127.0.0.1", False),
    ],
)
def test_address_is_global(address: str, expected: bool) -> None:
    assert address_is_global(address) is expected


def test_validate_url_destination_rejects_backslash_authority() -> None:
    with pytest.raises(InvalidURLImageInput):
        validate_url_destination("https://localhost:6666\\@evil.example")


def test_validate_url_destination_rejects_non_https_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(config, "ALLOW_NON_HTTPS_URL_INPUT", False)
    with pytest.raises(InvalidURLImageInput):
        validate_url_destination("http://some.com/file.jpg")


def test_validate_url_destination_respects_blacklist(monkeypatch) -> None:
    monkeypatch.setattr(
        config, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", {"blocked.example"}
    )
    with pytest.raises(InvalidURLImageInput):
        validate_url_destination("https://blocked.example/file.jpg")


def test_validate_url_destination_respects_whitelist(monkeypatch) -> None:
    monkeypatch.setattr(
        config, "WHITELISTED_DESTINATIONS_FOR_URL_INPUT", {"allowed.example"}
    )
    with pytest.raises(InvalidURLImageInput):
        validate_url_destination("https://not-allowed.example/file.jpg")
    # allowed host passes validation
    assert validate_url_destination("https://allowed.example/file.jpg").startswith(
        "https://allowed.example"
    )


def test_resolve_blocks_non_global_when_disallowed(monkeypatch) -> None:
    monkeypatch.setattr(
        url_utils.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")
    )
    with pytest.raises(URLAddressNotAllowedError):
        resolve_and_validate_ips(
            host="metadata.example", port=443, allow_non_global_addresses=False
        )


def test_adapter_pins_to_resolved_global_ip(monkeypatch) -> None:
    monkeypatch.setattr(url_utils.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection("https://example.com/image.jpg")
    assert isinstance(pool, HTTPSConnectionPool)
    assert pool.host == "8.8.8.8"
    assert pool.assert_hostname == "example.com"


def test_adapter_blocks_non_global_ip_literal() -> None:
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection("https://127.0.0.1/image.jpg")


def test_fetch_url_bytes_validated_redirects_revalidate_hops(
    requests_mock, monkeypatch
) -> None:
    monkeypatch.setattr(config, "VALIDATE_IMAGE_URL_REDIRECTS", True)
    monkeypatch.setattr(config, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
    monkeypatch.setattr(
        config, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", {"internal.example"}
    )
    start = "https://start.example/image.jpg"
    blocked = "https://internal.example/secret"
    requests_mock.get(start, status_code=302, headers={"Location": blocked})

    with pytest.raises(InvalidURLImageInput):
        fetch_url_bytes(start)


def test_fetch_url_bytes_enforces_max_redirects(requests_mock, monkeypatch) -> None:
    monkeypatch.setattr(config, "VALIDATE_IMAGE_URL_REDIRECTS", True)
    monkeypatch.setattr(config, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
    monkeypatch.setattr(config, "MAX_IMAGE_URL_REDIRECTS", 2)
    looping = "https://loop.example/image.jpg"
    requests_mock.get(looping, status_code=302, headers={"Location": looping})

    from inference_sdk.http.errors import HTTPClientError

    with pytest.raises(HTTPClientError):
        fetch_url_bytes(looping)


@pytest.mark.asyncio
async def test_validating_resolver_blocks_non_global(monkeypatch) -> None:
    resolver = url_utils.ValidatingResolver(allow_non_global_addresses=False)

    async def _fake_resolve(host, port=0, family=socket.AF_INET):
        return [
            {
                "hostname": host,
                "host": "169.254.169.254",
                "port": port,
                "family": family,
                "proto": 0,
                "flags": 0,
            }
        ]

    monkeypatch.setattr(resolver._delegate, "resolve", _fake_resolve)
    with pytest.raises(URLAddressNotAllowedError):
        await resolver.resolve("metadata.example", 443)
    await resolver.close()


@pytest.mark.asyncio
async def test_fetch_url_bytes_async_validated_redirect_blocked(monkeypatch) -> None:
    from aioresponses import aioresponses

    monkeypatch.setattr(config, "VALIDATE_IMAGE_URL_REDIRECTS", True)
    monkeypatch.setattr(config, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
    monkeypatch.setattr(
        config, "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT", {"internal.example"}
    )
    start = "https://start.example/image.jpg"
    blocked = "https://internal.example/secret"
    with aioresponses() as m:
        m.get(start, status=302, headers={"Location": blocked})
        with pytest.raises(InvalidURLImageInput):
            await url_utils.fetch_url_bytes_async(start)
