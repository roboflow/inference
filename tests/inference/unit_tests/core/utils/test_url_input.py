import socket

import pytest
import requests
from requests.models import PreparedRequest
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from inference.core.utils import url_input
from inference.core.utils.url_input import (
    SSRFProtectedHTTPAdapter,
    URLAddressNotAllowedError,
    address_is_global,
    fetch_url_content_validating_redirects,
    resolve_and_validate_ips,
)

_HAS_TLS_CONTEXT = hasattr(SSRFProtectedHTTPAdapter, "build_connection_pool_key_attributes")


def _prepared_get(url: str) -> PreparedRequest:
    request = PreparedRequest()
    request.prepare(method="GET", url=url)
    return request


def _fake_getaddrinfo(ip: str):
    def _inner(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]

    return _inner


@pytest.mark.parametrize(
    "address, expected",
    [
        ("8.8.8.8", True),
        ("1.1.1.1", True),
        ("127.0.0.1", False),  # loopback
        ("10.0.0.5", False),  # private
        ("192.168.1.10", False),  # private
        ("169.254.169.254", False),  # link-local / cloud metadata
        ("100.64.0.1", False),  # CGNAT
        ("0.0.0.0", False),  # unspecified
        ("::1", False),  # IPv6 loopback
        ("fc00::1", False),  # IPv6 ULA
        ("::ffff:127.0.0.1", False),  # IPv4-mapped loopback must not slip through
        ("2001:4860:4860::8888", True),  # public IPv6 (Google DNS)
        ("not-an-ip", False),
    ],
)
def test_address_is_global(address: str, expected: bool) -> None:
    assert address_is_global(address) is expected


def test_resolve_and_validate_ips_blocks_non_global_when_disallowed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        url_input.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")
    )
    with pytest.raises(URLAddressNotAllowedError):
        resolve_and_validate_ips(
            host="metadata.attacker.example",
            port=443,
            allow_non_global_addresses=False,
        )


def test_resolve_and_validate_ips_allows_non_global_when_permitted(
    monkeypatch,
) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("127.0.0.1"))
    result = resolve_and_validate_ips(
        host="localhost.example",
        port=80,
        allow_non_global_addresses=True,
    )
    assert result == ["127.0.0.1"]


def test_resolve_and_validate_ips_allows_global(monkeypatch) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    result = resolve_and_validate_ips(
        host="example.com",
        port=443,
        allow_non_global_addresses=False,
    )
    assert result == ["8.8.8.8"]


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


def test_adapter_blocks_ip_literal_that_is_non_global() -> None:
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        adapter.get_connection("http://127.0.0.1:8080/secret")


def test_adapter_blocks_hostname_resolving_to_non_global(monkeypatch) -> None:
    monkeypatch.setattr(
        url_input.socket, "getaddrinfo", _fake_getaddrinfo("169.254.169.254")
    )
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    with pytest.raises(URLAddressNotAllowedError):
        # nip.io-style: a valid FQDN that resolves to the metadata address.
        adapter.get_connection("https://169-254-169-254.nip.io/latest/meta-data")


def test_adapter_pins_connection_to_resolved_global_ip(monkeypatch) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection("https://example.com/image.jpg")
    assert isinstance(pool, HTTPSConnectionPool)
    assert pool.host == "8.8.8.8"  # pinned to the validated IP
    assert pool.assert_hostname == "example.com"  # cert still checked vs hostname


def test_adapter_pins_http_connection(monkeypatch) -> None:
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    pool = adapter.get_connection("http://example.com/image.jpg")
    assert isinstance(pool, HTTPConnectionPool)
    assert pool.host == "8.8.8.8"


# The methods below drive get_connection_with_tls_context, which is the method
# actually on the requests>=2.32 send() path (get_connection is not).
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
    assert pool.host == "8.8.8.8"  # dialled at the validated IP
    assert pool.assert_hostname == "example.com"  # cert checked vs hostname
    assert pool.conn_kw.get("server_hostname") == "example.com"  # SNI vs hostname
    # A real connection object can be built (guards against latent conn_kw bugs).
    conn = pool._new_conn()
    assert conn.host == "8.8.8.8"
    assert getattr(conn, "server_hostname", None) == "example.com"


@pytest.mark.skipif(not _HAS_TLS_CONTEXT, reason="requests < 2.32")
def test_tls_context_allows_global_and_preserves_custom_ca(monkeypatch) -> None:
    # verify as a CA-bundle path must flow through to the pinned pool.
    monkeypatch.setattr(url_input.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8"))
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=True)
    pool = adapter.get_connection_with_tls_context(
        _prepared_get("https://example.com/image.jpg"), verify=True
    )
    # allow_non_global=True still pins to the resolved IP.
    assert pool.host == "8.8.8.8"
    assert pool.assert_hostname == "example.com"


def test_validating_redirects_revalidates_each_hop(requests_mock) -> None:
    start = "https://start.example/image.jpg"
    hop = "https://hop.example/real.jpg"
    requests_mock.get(start, status_code=302, headers={"Location": hop})
    requests_mock.get(hop, content=b"image-bytes")

    validated = []

    def _validate(url: str) -> str:
        validated.append(url)
        return url

    content = fetch_url_content_validating_redirects(
        url=start,
        allow_non_global_addresses=True,
        max_redirects=30,
        validate_redirect=_validate,
    )

    assert content == b"image-bytes"
    assert validated == [hop]  # the redirect hop was re-validated before following


def test_validating_redirects_propagates_validation_rejection(requests_mock) -> None:
    start = "https://start.example/image.jpg"
    hop = "http://169.254.169.254/latest/meta-data"
    requests_mock.get(start, status_code=302, headers={"Location": hop})

    def _reject(url: str) -> str:
        raise ValueError("blocked hop")

    with pytest.raises(ValueError, match="blocked hop"):
        fetch_url_content_validating_redirects(
            url=start,
            allow_non_global_addresses=True,
            max_redirects=30,
            validate_redirect=_reject,
        )


def test_validating_redirects_enforces_max_redirects(requests_mock) -> None:
    # An endless redirect loop must terminate with TooManyRedirects.
    looping = "https://loop.example/image.jpg"
    requests_mock.get(looping, status_code=302, headers={"Location": looping})

    with pytest.raises(requests.exceptions.TooManyRedirects):
        fetch_url_content_validating_redirects(
            url=looping,
            allow_non_global_addresses=True,
            max_redirects=3,
            validate_redirect=lambda url: url,
        )
