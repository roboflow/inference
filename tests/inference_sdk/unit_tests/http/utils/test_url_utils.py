import socket

import aiohttp

import pytest
import requests
from urllib3.connectionpool import HTTPSConnectionPool

from inference_sdk import config
from inference_sdk.http.utils import url_utils
from inference_sdk.http.utils.url_utils import (
    SSRFProtectedHTTPAdapter,
    address_is_global,
    fetch_url_bytes,
    resolve_and_validate_ips,
    validate_url_destination,
)
from inference_sdk.http.errors import InvalidURLImageInput, URLAddressNotAllowedError


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


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/image.jpg",
        "https://192.168.0.1/image.jpg",
        "https://internalhost/image.jpg",
        "https://internal.corp/image.jpg",
        "https://metadata.google.internal/latest/meta-data",
    ],
)
def test_validate_url_destination_fqdn_gate_rejects_suffixless_hosts(
    url: str, monkeypatch
) -> None:
    # Mirrors the server: with FQDN enforced, IPs / bare hosts / suffix-less
    # internal names are rejected on the URL string alone.
    monkeypatch.setattr(config, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
    monkeypatch.setattr(config, "ALLOW_NON_HTTPS_URL_INPUT", True)
    with pytest.raises(InvalidURLImageInput):
        validate_url_destination(url)


def test_validate_url_destination_fqdn_gate_allows_public_domain(monkeypatch) -> None:
    monkeypatch.setattr(config, "ALLOW_URL_INPUT_WITHOUT_FQDN", False)
    assert validate_url_destination("https://sub.example.com/image.jpg").startswith(
        "https://sub.example.com"
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


_HAS_TLS_CONTEXT = hasattr(
    SSRFProtectedHTTPAdapter, "build_connection_pool_key_attributes"
)


def _prepared_get(url: str):
    from requests.models import PreparedRequest

    request = PreparedRequest()
    request.prepare(method="GET", url=url)
    return request


# get_connection_with_tls_context is the requests>=2.32 send() path.
@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_blocks_hostname_resolving_to_non_global(monkeypatch) -> None:
    monkeypatch.setattr(
        url_utils.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")
    )
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection_with_tls_context(
            _prepared_get("https://metadata.example/latest/meta-data"), verify=True
        )


@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_pins_to_resolved_global_ip(monkeypatch) -> None:
    monkeypatch.setattr(url_utils.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection_with_tls_context(
        _prepared_get("https://example.com/image.jpg"), verify=True
    )
    assert isinstance(pool, HTTPSConnectionPool)
    assert pool.host == "8.8.8.8"
    assert pool.assert_hostname == "example.com"
    assert pool.conn_kw.get("server_hostname") == "example.com"
    conn = pool._new_conn()
    assert conn.host == "8.8.8.8"
    assert getattr(conn, "server_hostname", None) == "example.com"


@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_pins_plain_http_without_tls_kwargs(monkeypatch) -> None:
    # Regression: http:// pinned pool must build a real connection (assert_hostname
    # / server_hostname are HTTPS-only and crash a plain HTTPConnection).
    from urllib3.connectionpool import HTTPConnectionPool

    monkeypatch.setattr(url_utils.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection_with_tls_context(
        _prepared_get("http://example.com/image.jpg"), verify=True
    )
    assert isinstance(pool, HTTPConnectionPool)
    assert pool.host == "8.8.8.8"
    conn = pool._new_conn()  # must not raise
    assert conn.host == "8.8.8.8"


@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_defers_and_warns_under_proxy(monkeypatch) -> None:
    monkeypatch.setattr(
        url_utils, "select_proxy", lambda url, proxies: "http://proxy.local:3128"
    )
    warned = []
    monkeypatch.setattr(
        url_utils, "_warn_proxy_bypasses_ssrf_protection", lambda: warned.append(1)
    )
    monkeypatch.setattr(
        url_utils.socket,
        "getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not resolve")),
    )
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    adapter.get_connection_with_tls_context(
        _prepared_get("https://metadata.example/x"), verify=True
    )
    assert warned == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/image.jpg",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/image.jpg",
    ],
)
async def test_fetch_url_bytes_async_blocks_non_global_ip_literal(
    target: str, monkeypatch
) -> None:
    # aiohttp skips its resolver for IP literals; the explicit pre-check must
    # still block non-global literal targets (no network is touched).
    monkeypatch.setattr(config, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", False)
    monkeypatch.setattr(config, "ALLOW_NON_HTTPS_URL_INPUT", True)
    monkeypatch.setattr(config, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
    with pytest.raises(URLAddressNotAllowedError):
        await url_utils.fetch_url_bytes_async(target)


@pytest.mark.asyncio
async def test_fetch_url_bytes_async_legacy_blocks_redirect_to_literal(
    monkeypatch,
) -> None:
    # Parity with sync: even in legacy redirect mode (VALIDATE=False), with
    # non-global blocking on, a redirect to a literal metadata IP is rejected.
    from aioresponses import aioresponses

    monkeypatch.setattr(config, "VALIDATE_IMAGE_URL_REDIRECTS", False)
    monkeypatch.setattr(config, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", False)
    monkeypatch.setattr(config, "ALLOW_NON_HTTPS_URL_INPUT", True)
    monkeypatch.setattr(config, "ALLOW_URL_INPUT_WITHOUT_FQDN", True)
    start = "https://start.example/image.jpg"
    metadata = "http://169.254.169.254/latest/meta-data"
    with aioresponses() as m:
        m.get(start, status=302, headers={"Location": metadata})
        with pytest.raises(URLAddressNotAllowedError):
            await url_utils.fetch_url_bytes_async(start)


@pytest.mark.asyncio
async def test_fetch_url_bytes_async_keeps_default_timeout(monkeypatch) -> None:
    # Passing an empty ClientTimeout() would disable aiohttp's 300s default;
    # when the caller gives no timeout we must not override it.
    monkeypatch.setattr(config, "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES", True)
    captured = {}
    real_session = url_utils.aiohttp.ClientSession

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return real_session(*args, **kwargs)

    monkeypatch.setattr(url_utils.aiohttp, "ClientSession", _capture)

    from aioresponses import aioresponses

    with aioresponses() as m:
        m.get("https://some.com/image.jpg", body=b"bytes")
        await url_utils.fetch_url_bytes_async("https://some.com/image.jpg")

    assert "timeout" not in captured  # aiohttp default (total=300s) left intact


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


def test_fetch_url_bytes_translates_transport_error_to_http_client_error(
    requests_mock,
) -> None:
    from inference_sdk.http.errors import HTTPClientError

    url = "https://some.com/file.jpg"
    requests_mock.get(url, exc=requests.exceptions.ConnectionError("boom"))

    with pytest.raises(HTTPClientError) as error:
        fetch_url_bytes(url)
    # It must not leak the raw requests exception past the SDK error contract.
    assert not isinstance(error.value, requests.exceptions.RequestException)


def test_fetch_url_bytes_preserves_http_status_error(requests_mock) -> None:
    from requests import HTTPError

    url = "https://some.com/file.jpg"
    requests_mock.get(url, status_code=500)

    # Status errors keep flowing as HTTPError so wrap_errors -> HTTPCallErrorError.
    with pytest.raises(HTTPError):
        fetch_url_bytes(url)


@pytest.mark.asyncio
async def test_fetch_url_bytes_async_translates_transport_error(monkeypatch) -> None:
    from aioresponses import aioresponses

    from inference_sdk.http.errors import HTTPClientError

    url = "https://some.com/file.jpg"
    with aioresponses() as m:
        m.get(url, exception=aiohttp.ClientConnectionError("boom"))
        with pytest.raises(HTTPClientError):
            await url_utils.fetch_url_bytes_async(url)


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
