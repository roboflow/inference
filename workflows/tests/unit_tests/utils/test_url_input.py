"""Security cases for `roboflow_workflows.utils.url_input`.

Mirrors the transport-layer cases from
`tests/inference/unit_tests/core/utils/test_url_input.py` without moving or
changing the server tests. Cases specific to image response draining, image
payload limits, image-side redirect policy, and server-only deprecation
warnings are intentionally left out - the Webhook Sink does not use them.
"""

import socket

import pytest
import requests
from requests.models import PreparedRequest
from roboflow_workflows.utils import url_input
from roboflow_workflows.utils.url_input import (
    SSRFProtectedHTTPAdapter,
    URLAddressNotAllowedError,
    address_is_global,
    resolve_and_validate_ips,
)
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

_HAS_TLS_CONTEXT = hasattr(
    SSRFProtectedHTTPAdapter, "build_connection_pool_key_attributes"
)


def _prepared_get(url: str) -> PreparedRequest:
    request = PreparedRequest()
    request.prepare(method="GET", url=url)
    return request


def _fake_getaddrinfo(*ips: str):
    def _inner(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
            for ip in ips
        ]

    return _inner


@pytest.mark.parametrize(
    "address, expected",
    [
        ("8.8.8.8", True),
        ("1.1.1.1", True),
        ("127.0.0.1", False),
        ("10.0.0.5", False),
        ("192.168.1.10", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("0.0.0.0", False),
        ("::1", False),
        ("fc00::1", False),
        ("::ffff:127.0.0.1", False),
        ("2001:4860:4860::8888", True),
        ("224.0.0.1", False),
        ("ff02::1", False),
        ("not-an-ip", False),
    ],
)
def test_address_is_global(address: str, expected: bool) -> None:
    assert address_is_global(address) is expected


def test_resolve_and_validate_ips_blocks_non_global(monkeypatch) -> None:
    monkeypatch.setattr(
        url_input.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")
    )
    with pytest.raises(URLAddressNotAllowedError):
        resolve_and_validate_ips(
            host="metadata.attacker.example",
            port=443,
            allow_non_global_addresses=False,
        )


def test_resolve_and_validate_ips_rejects_mixed_answer(monkeypatch) -> None:
    # Any non-global answer poisons the whole set: the pinned IP could be the
    # public one on the first send and the private one after a rebind.
    monkeypatch.setattr(
        url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8", "127.0.0.1")
    )
    with pytest.raises(URLAddressNotAllowedError):
        resolve_and_validate_ips(
            host="mixed.example",
            port=443,
            allow_non_global_addresses=False,
        )


def test_resolve_and_validate_ips_allows_global(monkeypatch) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    assert resolve_and_validate_ips(
        host="example.com", port=443, allow_non_global_addresses=False
    ) == ["8.8.8.8"]


def test_resolve_raises_connection_error_when_host_unresolvable(monkeypatch) -> None:
    def _raise(host, port, *args, **kwargs):
        raise socket.gaierror("nope")

    monkeypatch.setattr(url_input.socket, "getaddrinfo", _raise)
    with pytest.raises(requests.exceptions.ConnectionError):
        resolve_and_validate_ips(
            host="does-not-exist.example",
            port=443,
            allow_non_global_addresses=False,
        )


def test_adapter_blocks_non_global_ip_literal() -> None:
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection("http://127.0.0.1:8080/secret")


def test_adapter_allows_non_global_ip_literal_when_enabled() -> None:
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=True)
    pool = adapter.get_connection("http://127.0.0.1:8080/webhook")
    assert pool.host == "127.0.0.1"


def test_adapter_blocks_hostname_resolving_to_non_global(monkeypatch) -> None:
    monkeypatch.setattr(
        url_input.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")
    )
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection("https://169-254-169-254.nip.io/latest/meta-data")


def test_adapter_blocks_ipv4_mapped_ipv6_literal() -> None:
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection("http://[::ffff:127.0.0.1]:80/x")


def test_adapter_pins_https_connection_to_resolved_ip(monkeypatch) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection("https://example.com/image.jpg")
    assert isinstance(pool, HTTPSConnectionPool)
    assert pool.host == "8.8.8.8"
    assert pool.assert_hostname == "example.com"


def test_adapter_pins_http_connection(monkeypatch) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection("http://example.com/image.jpg")
    assert isinstance(pool, HTTPConnectionPool)
    assert pool.host == "8.8.8.8"
    # http:// must NOT receive HTTPS-only assert_hostname/server_hostname, which
    # would raise TypeError at connect time; build a real connection to prove it.
    conn = pool._new_conn()
    assert conn.host == "8.8.8.8"


def test_adapter_resolves_host_only_once(monkeypatch) -> None:
    # Rebinding: first resolution is a public IP, a second call would return a
    # private IP. The pinned pool must connect to the first result.
    calls = {"n": 0}
    results = [
        [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("8.8.8.8", 443),
            )
        ],
        [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", 443),
            )
        ],
    ]

    def _resolve(host, port, *args, **kwargs):
        calls["n"] += 1
        return results.pop(0)

    monkeypatch.setattr(url_input.socket, "getaddrinfo", _resolve)
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection("https://example.com/x")
    assert pool.host == "8.8.8.8"
    assert calls["n"] == 1  # only one DNS lookup during pinning


def test_adapter_rejects_proxy() -> None:
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection(
            "https://example.com/x",
            proxies={"https": "http://proxy.local:3128"},
        )


def test_adapter_uses_proxy_when_non_global_allowed(monkeypatch) -> None:
    monkeypatch.setattr(
        url_input.socket,
        "getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=True)
    pool = adapter.get_connection(
        "https://example.com/x",
        proxies={"https": "http://proxy.local:3128"},
    )
    # Stock HTTPAdapter path: pool tunnels through the proxy, not pinned.
    assert pool.host == "example.com"
    assert pool.proxy.host == "proxy.local"
    assert pool.proxy.port == 3128


@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_blocks_hostname_resolving_to_non_global(monkeypatch) -> None:
    monkeypatch.setattr(
        url_input.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")
    )
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection_with_tls_context(
            _prepared_get("https://metadata.example/latest/meta-data"), verify=True
        )


@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_blocks_non_global_ip_literal() -> None:
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection_with_tls_context(
            _prepared_get("https://127.0.0.1/image.jpg"), verify=True
        )


@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_pins_to_resolved_global_ip(monkeypatch) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection_with_tls_context(
        _prepared_get("https://example.com/image.jpg"), verify=True
    )
    assert isinstance(pool, HTTPSConnectionPool)
    assert pool.host == "8.8.8.8"
    assert pool.assert_hostname == "example.com"
    assert pool.conn_kw.get("server_hostname") == "example.com"


@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_rejects_proxy(monkeypatch) -> None:
    # Resolution must NOT run when a proxy is configured; the adapter refuses
    # to hand the destination off to a proxy that resolves it externally.
    monkeypatch.setattr(
        url_input.socket,
        "getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection_with_tls_context(
            _prepared_get("https://metadata.example/x"),
            verify=True,
            proxies={"https": "http://proxy.local:3128"},
        )


def test_send_sets_host_header_for_pinned_pool(monkeypatch) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    captured = {}

    class _Sentinel(Exception):
        pass

    def _fake_super_send(self, request, **kwargs):
        captured["host"] = request.headers.get("Host")
        raise _Sentinel

    monkeypatch.setattr(url_input.HTTPAdapter, "send", _fake_super_send, raising=True)
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    request = _prepared_get("https://example.com:8443/x")
    with pytest.raises(_Sentinel):
        adapter.send(request)
    assert captured["host"] == "example.com:8443"
