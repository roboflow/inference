"""SSRF-hardened primitives for fetching caller-supplied URL image input.

This module owns the *network* side of URL image loading: resolving hostnames,
rejecting non-global destinations, pinning the connection to the validated IP so
DNS rebinding cannot swap the target after validation, and (optionally)
re-validating every redirect hop instead of letting ``requests`` follow
redirects blindly.

The *URL-string* policy (scheme / FQDN / allow-list / block-list) stays in
``inference.core.utils.image_utils`` and is injected here as a ``validate_redirect``
callback so this module has no opinion about how a URL string is judged.

Background: GHSA-hjmm-hr52-vrp2.
"""

import ipaddress
import socket
import threading
import urllib.parse
import warnings
from typing import Callable, List, Optional, Tuple

import requests
import urllib3.util
from requests.adapters import HTTPAdapter
from requests.utils import select_proxy
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from inference.core import logger
from inference.core.utils.requests import api_key_safe_raise_for_status
from inference.core.warnings import InferenceDeprecationWarning

# ---------------------------------------------------------------------------
# Deprecation notices (emitted once per process to avoid hot-path log spam).
# ---------------------------------------------------------------------------
_WARNING_LOCK = threading.Lock()
_LEGACY_REDIRECT_WARNING_EMITTED = False
_NON_GLOBAL_WARNING_EMITTED = False


class URLAddressNotAllowedError(Exception):
    """Raised when a URL resolves to a destination that is not permitted."""


def _warn_once(flag_name: str, message: str) -> None:
    global _LEGACY_REDIRECT_WARNING_EMITTED, _NON_GLOBAL_WARNING_EMITTED
    with _WARNING_LOCK:
        if flag_name == "VALIDATE_IMAGE_URL_REDIRECTS":
            if _LEGACY_REDIRECT_WARNING_EMITTED:
                return
            _LEGACY_REDIRECT_WARNING_EMITTED = True
        else:
            if _NON_GLOBAL_WARNING_EMITTED:
                return
            _NON_GLOBAL_WARNING_EMITTED = True
    warnings.warn(message, category=InferenceDeprecationWarning, stacklevel=2)
    logger.warning(message)


def _warn_legacy_redirect_handling() -> None:
    _warn_once(
        "VALIDATE_IMAGE_URL_REDIRECTS",
        "URL image redirects are being followed without per-hop validation "
        "(VALIDATE_IMAGE_URL_REDIRECTS=False). This is an SSRF-sensitive default "
        "and is scheduled to change to True in Q4 2026. Set "
        "VALIDATE_IMAGE_URL_REDIRECTS=True to opt in early.",
    )


def _warn_non_global_allowed() -> None:
    _warn_once(
        "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES",
        "URL image input is allowed to reach non-global addresses "
        "(ALLOW_URL_TO_NON_GLOBAL_ADDRESSES=True). This is an SSRF-sensitive "
        "default and is scheduled to change to False in Q4 2026. Set "
        "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES=False to opt in early.",
    )


# ---------------------------------------------------------------------------
# Address classification + resolution.
# ---------------------------------------------------------------------------
def address_is_global(address: str) -> bool:
    """Return True only for public, routable unicast addresses.

    ``ipaddress.is_global`` already excludes loopback, private (RFC1918),
    link-local (incl. 169.254.169.254 metadata), CGNAT (100.64/10), ULA
    (fc00::/7), unspecified and reserved ranges, so a single check covers the
    destinations the advisory asks us to block. IPv4-mapped IPv6 is unwrapped so
    ``::ffff:127.0.0.1`` cannot smuggle a loopback target past the check.
    """
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        parsed = parsed.ipv4_mapped
    return parsed.is_global


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
    """Resolve ``host`` and, unless non-global is allowed, require every
    resolved IP to be global. Returns the resolved IPs (validated ones first
    would be identical since all must pass).

    Rejecting when *any* resolved address is non-global is deliberately
    conservative: it prevents a rebinding-style response that mixes a global and
    a non-global A-record from later steering the pinned connection to the
    non-global one.
    """
    try:
        addr_infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as error:
        # Unresolvable host is a normal connection failure, not an SSRF block.
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


# ---------------------------------------------------------------------------
# Connection pinning adapter.
# ---------------------------------------------------------------------------
class SSRFProtectedHTTPAdapter(HTTPAdapter):
    """``requests`` adapter that validates + pins the destination IP.

    For hostname targets it resolves once, validates, and connects the pool to
    the resolved IP while preserving the original hostname for TLS SNI, cert
    verification, and the ``Host`` header — so a second resolution (rebinding)
    cannot redirect the socket. For IP-literal targets it validates the literal
    directly and lets ``requests`` connect normally.

    The adapter is mounted on every hop, so even when ``requests`` follows
    redirects itself (legacy mode) each redirect connection is still validated.
    """

    def __init__(self, *, allow_non_global_addresses: bool, **kwargs):
        self._allow_non_global_addresses = allow_non_global_addresses
        super().__init__(**kwargs)

    def send(self, request, **kwargs):
        # Preserve the vhost/Host header when the pool is pinned to a raw IP.
        parsed = urllib3.util.parse_url(request.url)
        if parsed.host is not None and not _host_is_ip_literal(parsed.host):
            host_header = parsed.host
            if parsed.port is not None:
                host_header = f"{host_header}:{parsed.port}"
            request.headers["Host"] = host_header
        return super().send(request, **kwargs)

    def _resolve_pin_target(self, url: str) -> Optional[Tuple[str, str]]:
        """Return ``(hostname, pinned_ip)`` to pin, or ``None`` to connect
        normally. Raises :class:`URLAddressNotAllowedError` when the destination
        is a blocked non-global address.
        """
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
            # The literal already is the validated address; connect directly.
            return None
        resolved_ips = resolve_and_validate_ips(
            host=host,
            port=port,
            allow_non_global_addresses=self._allow_non_global_addresses,
        )
        return host, resolved_ips[0]

    def _build_pinned_pool(self, host_params, pool_kwargs, hostname, pinned_ip):
        # Reuse requests' own host params / TLS pool kwargs (so proxies, custom
        # CA bundles and mTLS client certs are preserved), but point the pool at
        # the validated IP and verify TLS against the original hostname.
        host_params = dict(host_params)
        host_params["host"] = pinned_ip
        pool_kwargs = dict(pool_kwargs)
        pool_kwargs["assert_hostname"] = hostname
        pool = self.poolmanager.connection_from_host(
            **host_params, pool_kwargs=pool_kwargs
        )
        # SNI must target the original hostname even though we dial the IP.
        pool.conn_kw["server_hostname"] = hostname
        return pool

    def get_connection_with_tls_context(
        self, request, verify, proxies=None, cert=None
    ):
        # This is the method on the send() path for requests >= 2.32 (the pinned
        # floor). With a forward proxy the proxy resolves the target, so
        # client-side pinning is moot -> defer and keep proxy/CA/mTLS intact.
        if select_proxy(request.url, proxies):
            return super().get_connection_with_tls_context(
                request, verify, proxies=proxies, cert=cert
            )
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
        # Fallback for requests < 2.32 (below the pinned floor); kept so the
        # protection also holds if an older requests is somehow installed.
        if select_proxy(url, proxies):
            return super().get_connection(url, proxies)
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


def _build_ssrf_protected_session(allow_non_global_addresses: bool) -> requests.Session:
    session = requests.Session()
    if allow_non_global_addresses:
        _warn_non_global_allowed()
        return session
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# Fetch entry points.
# ---------------------------------------------------------------------------
def fetch_url_content_legacy(
    *,
    url: str,
    allow_non_global_addresses: bool,
    max_redirects: int,
    request_timeout: Optional[float] = None,
) -> bytes:
    """Legacy fetch: ``requests`` follows redirects itself (capped at
    ``max_redirects``). Behaviour matches the pre-hardening implementation; the
    only additions are the explicit redirect cap and — when non-global is
    disallowed — per-connection IP validation/pinning via the mounted adapter.
    """
    _warn_legacy_redirect_handling()
    session = _build_ssrf_protected_session(allow_non_global_addresses)
    session.max_redirects = max_redirects
    try:
        response = session.get(
            url, stream=True, allow_redirects=True, timeout=request_timeout
        )
        api_key_safe_raise_for_status(response=response)
        return response.content
    finally:
        session.close()


def fetch_url_content_validating_redirects(
    *,
    url: str,
    allow_non_global_addresses: bool,
    max_redirects: int,
    validate_redirect: Callable[[str], str],
    request_timeout: Optional[float] = None,
) -> bytes:
    """Hardened fetch: follow redirects one hop at a time, re-running the full
    URL-string policy (via ``validate_redirect``) on every hop before the next
    request is issued. IP validation/pinning is applied on every hop by the
    mounted adapter.
    """
    session = _build_ssrf_protected_session(allow_non_global_addresses)
    current_url = url
    try:
        for _ in range(max_redirects + 1):
            response = session.get(
                current_url,
                stream=True,
                allow_redirects=False,
                timeout=request_timeout,
            )
            if response.is_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise requests.exceptions.RequestException(
                        "Redirect response did not contain a Location header."
                    )
                next_url = urllib.parse.urljoin(current_url, location)
                # Re-run scheme / FQDN / allow-list / block-list on the hop.
                current_url = validate_redirect(next_url)
                continue
            api_key_safe_raise_for_status(response=response)
            return response.content
        raise requests.exceptions.TooManyRedirects(
            f"Exceeded maximum of {max_redirects} redirects."
        )
    finally:
        session.close()
