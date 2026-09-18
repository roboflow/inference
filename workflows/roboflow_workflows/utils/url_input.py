"""Workflows-owned SSRF transport primitives for outbound HTTP.

Behaviourally mirrors ``inference/core/utils/url_input.py`` but stays inside
the Workflows tree so this package has no import into the inference server.
The Webhook Sink uses this module directly; the block does not consume
response bodies, does not follow redirects, and never allows a non-global
destination.

Advisory: GHSA-6r6j-3cw6-wjcr.
"""

import ipaddress
import socket
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3.util
from requests.adapters import HTTPAdapter
from requests.utils import select_proxy
from urllib3.connectionpool import HTTPConnectionPool


class URLAddressNotAllowedError(Exception):
    """Raised when a URL resolves to a destination that is not permitted."""


def address_is_global(address: str) -> bool:
    """Return True only for public, routable unicast addresses.

    ``ipaddress.is_global`` excludes loopback, private (RFC1918), link-local
    (incl. 169.254.169.254 metadata), CGNAT (100.64/10), ULA (fc00::/7),
    unspecified and reserved ranges. IPv4-mapped IPv6 is unwrapped so
    ``::ffff:127.0.0.1`` cannot smuggle a loopback target past the check.
    """
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return parsed.is_global and not parsed.is_multicast


def _strip_ipv6_brackets(host: str) -> str:
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _host_is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(_strip_ipv6_brackets(host))
        return True
    except ValueError:
        return False


def resolve_and_validate_ips(
    host: str,
    port: int,
    allow_non_global_addresses: bool,
) -> List[str]:
    """Resolve ``host`` and require every resolved IP to be global unless
    non-global is explicitly allowed. Returns the resolved IPs.

    Rejecting when *any* resolved address is non-global blocks mixed
    public/private DNS answers from steering the pinned connection to the
    non-global one.
    """
    try:
        addr_infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as error:
        raise requests.exceptions.ConnectionError(
            f"Could not resolve host: {host}"
        ) from error
    resolved_ips = [info[4][0] for info in addr_infos]
    if not resolved_ips:
        raise requests.exceptions.ConnectionError(f"Could not resolve host: {host}")
    if not allow_non_global_addresses:
        for ip in resolved_ips:
            if not address_is_global(ip):
                raise URLAddressNotAllowedError(
                    f"Host '{host}' resolves to non-global address '{ip}'."
                )
    return resolved_ips


class SSRFProtectedHTTPAdapter(HTTPAdapter):
    """``requests`` adapter that validates + pins the destination IP.

    For hostname targets it resolves once, validates every answer, and
    connects the pool to the resolved IP while preserving the original
    hostname for TLS SNI, cert verification and the ``Host`` header - so a
    second resolution (rebinding) cannot redirect the socket. For IP-literal
    targets it validates the literal directly and lets ``requests`` connect
    normally.

    A configured proxy is rejected outright: a forward proxy would resolve
    the destination outside this adapter, breaking the pinning guarantee.
    """

    def __init__(self, *, allow_non_global_addresses: bool, **kwargs):
        self._allow_non_global_addresses = allow_non_global_addresses
        super().__init__(**kwargs)

    def send(self, request, **kwargs):
        parsed = urllib3.util.parse_url(request.url)
        if parsed.host is not None and not _host_is_ip_literal(parsed.host):
            host_header = parsed.host
            if parsed.port is not None:
                host_header = f"{host_header}:{parsed.port}"
            request.headers["Host"] = host_header
        return super().send(request, **kwargs)

    def _reject_proxy(self, url: str, proxies) -> None:
        if select_proxy(url, proxies):
            raise URLAddressNotAllowedError(
                "Webhook transport refuses to use an HTTP(S) proxy: a proxy "
                "would resolve the destination outside the validating adapter."
            )

    def _resolve_pin_target(self, url: str) -> Optional[Tuple[str, str]]:
        parsed = urllib3.util.parse_url(url)
        host = parsed.host
        if host is None:
            return None
        scheme = parsed.scheme or "https"
        port = parsed.port or (443 if scheme == "https" else 80)
        if _host_is_ip_literal(host):
            literal = _strip_ipv6_brackets(host)
            if not self._allow_non_global_addresses and not address_is_global(literal):
                raise URLAddressNotAllowedError(
                    f"URL points to non-global address '{literal}'."
                )
            return None
        resolved_ips = resolve_and_validate_ips(
            host=host,
            port=port,
            allow_non_global_addresses=self._allow_non_global_addresses,
        )
        return host, resolved_ips[0]

    def _build_pinned_pool(
        self,
        host_params: Dict[str, Any],
        pool_kwargs: Dict[str, Any],
        hostname: str,
        pinned_ip: str,
    ) -> HTTPConnectionPool:
        host_params = dict(host_params)
        host_params["host"] = pinned_ip
        pool_kwargs = dict(pool_kwargs)
        is_https = host_params.get("scheme") == "https"
        if is_https:
            pool_kwargs["assert_hostname"] = hostname
        pool = self.poolmanager.connection_from_host(
            **host_params, pool_kwargs=pool_kwargs
        )
        if is_https:
            pool.conn_kw["server_hostname"] = hostname
        return pool

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        self._reject_proxy(request.url, proxies)
        pin = self._resolve_pin_target(request.url)
        if pin is None:
            return super().get_connection_with_tls_context(
                request, verify, proxies=proxies, cert=cert
            )
        hostname, pinned_ip = pin
        host_params, pool_kwargs = self.build_connection_pool_key_attributes(
            request, verify, cert
        )
        return self._build_pinned_pool(host_params, pool_kwargs, hostname, pinned_ip)

    def get_connection(self, url, proxies=None):
        self._reject_proxy(url, proxies)
        pin = self._resolve_pin_target(url)
        if pin is None:
            return super().get_connection(url, proxies)
        hostname, pinned_ip = pin
        parsed = urllib3.util.parse_url(url)
        scheme = parsed.scheme or "https"
        port = parsed.port or (443 if scheme == "https" else 80)
        host_params = {"scheme": scheme, "host": pinned_ip, "port": port}
        pool_kwargs = {"cert_reqs": "CERT_REQUIRED"} if scheme == "https" else {}
        return self._build_pinned_pool(host_params, pool_kwargs, hostname, pinned_ip)
