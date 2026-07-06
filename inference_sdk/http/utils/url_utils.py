"""SSRF-hardened URL image fetching for the inference SDK.

Mirrors the protections in ``inference.core.utils.url_input`` / ``image_utils``
but is self-contained (the SDK must not import from the ``inference`` package).

Covers both the sync (``requests``) and async (``aiohttp``) code paths:

* URL-string policy: scheme / FQDN / allow-list / block-list.
* Non-global destination blocking with connection pinning (sync: a pinning
  adapter; async: a validating resolver), so DNS rebinding cannot swap the
  target after validation.
* Redirect handling: either legacy (client follows redirects, capped) or
  hardened (follow one hop at a time, re-validating each hop URL).

FQDN handling mirrors the inference server exactly: ``tldextract`` derives the
FQDN (empty for IPs and hosts without a public suffix) for the
``ALLOW_URL_INPUT_WITHOUT_FQDN`` gate, and the allow/block lists are matched
against the concatenated ``subdomain.domain.suffix`` network location.
"""

import ipaddress
import socket
import urllib.parse
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import requests
import tldextract
import urllib3.util
from aiohttp import ClientResponseError
from aiohttp.abc import AbstractResolver
from requests import HTTPError
from requests.adapters import HTTPAdapter
from requests.utils import select_proxy
from tldextract.tldextract import ExtractResult
from urllib3.connectionpool import HTTPConnectionPool

from inference_sdk import config
from inference_sdk.http.errors import HTTPClientError
from inference_sdk.http.utils.requests import (
    api_key_safe_raise_for_status,
    deduct_api_key_from_string,
)

class InvalidURLImageInput(HTTPClientError):
    """Raised when a URL image reference fails the SDK URL-string policy."""


class URLAddressNotAllowedError(HTTPClientError):
    """Raised when a URL resolves to a destination that is not permitted."""


# `warnings.warn` deduplicates identical warnings per call site under the
# default filter, so these do not spam the hot path.
def _warn_legacy_redirect_handling() -> None:
    warnings.warn(
        "URL image redirects are followed without per-hop validation "
        "(VALIDATE_IMAGE_URL_REDIRECTS=False). This default is scheduled to "
        "change to True in Q4 2026.",
        category=DeprecationWarning,
        stacklevel=2,
    )


def _warn_non_global_allowed() -> None:
    warnings.warn(
        "URL image input is allowed to reach non-global addresses "
        "(ALLOW_URL_TO_NON_GLOBAL_ADDRESSES=True). This default is scheduled to "
        "change to False in Q4 2026.",
        category=DeprecationWarning,
        stacklevel=2,
    )


def _warn_proxy_bypasses_ssrf_protection() -> None:
    warnings.warn(
        "An HTTP(S) proxy is configured for URL image fetching; the proxy "
        "resolves the destination, so non-global address blocking and DNS "
        "rebinding pinning are NOT enforced for these requests.",
        category=UserWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Address classification.
# ---------------------------------------------------------------------------
def address_is_global(address: str) -> bool:
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
    host: str, port: int, allow_non_global_addresses: bool
) -> List[str]:
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


# ---------------------------------------------------------------------------
# URL-string policy (scheme / FQDN / allow-list / block-list).
# ---------------------------------------------------------------------------
def validate_url_destination(value: str) -> str:
    """Validate the URL string against the SDK policy and return the prepared
    URL. Raises :class:`InvalidURLImageInput` on rejection.
    """
    if not config.ALLOW_URL_INPUT:
        raise InvalidURLImageInput("Providing images via URL is not enabled.")
    try:
        original_parsed = urllib.parse.urlparse(value)
        if "\\" in original_parsed.netloc:
            raise ValueError("URL authority contains a backslash")
        prepared_url = requests.Request(method="GET", url=value).prepare().url
        parsed = urllib.parse.urlparse(prepared_url)
    except (requests.exceptions.RequestException, ValueError) as error:
        raise InvalidURLImageInput("Provided image URL is invalid.") from error

    if parsed.scheme != "https" and not config.ALLOW_NON_HTTPS_URL_INPUT:
        raise InvalidURLImageInput("Providing images via non-https URL is not enabled.")

    network_location = parsed.hostname or ""
    if ":" in network_location:
        network_location = f"[{network_location}]"
    domain_extraction_result = tldextract.TLDExtract(suffix_list_urls=())(
        network_location
    )  # strip ports and parse FQDNs (mirrors the inference server)
    if not domain_extraction_result.fqdn and not config.ALLOW_URL_INPUT_WITHOUT_FQDN:
        raise InvalidURLImageInput(
            "Providing images via URL without FQDN is not enabled."
        )
    destination = _concatenate_chunks_of_network_location(
        extraction_result=domain_extraction_result
    )  # even if there is no FQDN but an address, this allows allow/block matching
    _ensure_not_blocked_by_lists(destination=destination)
    return prepared_url


def _concatenate_chunks_of_network_location(extraction_result: ExtractResult) -> str:
    chunks = [
        extraction_result.subdomain,
        extraction_result.domain,
        extraction_result.suffix,
    ]
    non_empty_chunks = [chunk for chunk in chunks if chunk]
    result = ".".join(non_empty_chunks)
    if result.startswith("[") and result.endswith("]"):
        # dropping brackets for IPv6
        return result[1:-1]
    return result


def _ensure_not_blocked_by_lists(destination: str) -> None:
    whitelist: Optional[Set[str]] = config.WHITELISTED_DESTINATIONS_FOR_URL_INPUT
    blacklist: Optional[Set[str]] = config.BLACKLISTED_DESTINATIONS_FOR_URL_INPUT
    if whitelist is not None and destination not in whitelist:
        raise InvalidURLImageInput(
            "URL destination is not on the allow-list (whitelisted)."
        )
    if blacklist is not None and destination in blacklist:
        raise InvalidURLImageInput(
            "URL destination is on the block-list (blacklisted)."
        )


# ---------------------------------------------------------------------------
# Sync path (requests) — pinning adapter, mirrors the server.
# ---------------------------------------------------------------------------
class SSRFProtectedHTTPAdapter(HTTPAdapter):
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

    def _resolve_pin_target(self, url: str) -> Optional[Tuple[str, str]]:
        """Return ``(hostname, pinned_ip)`` to pin, or ``None`` to connect
        normally. Raises :class:`URLAddressNotAllowedError` on a blocked
        non-global destination.
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
        # Reuse requests' own host params / TLS pool kwargs (preserving proxies,
        # custom CA bundles and mTLS client certs), but dial the validated IP and
        # verify TLS against the original hostname.
        host_params = dict(host_params)
        host_params["host"] = pinned_ip
        pool_kwargs = dict(pool_kwargs)
        is_https = host_params.get("scheme") == "https"
        if is_https:
            # assert_hostname / server_hostname are HTTPS-only; forwarding them
            # to a plain HTTPConnection raises TypeError at connect time.
            pool_kwargs["assert_hostname"] = hostname
        pool = self.poolmanager.connection_from_host(
            **host_params, pool_kwargs=pool_kwargs
        )
        if is_https:
            pool.conn_kw["server_hostname"] = hostname
        return pool

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        # send() path for requests >= 2.32 (the pinned floor). Defer under a
        # forward proxy (the proxy resolves the target, so pinning is moot).
        if select_proxy(request.url, proxies):
            if not self._allow_non_global_addresses:
                _warn_proxy_bypasses_ssrf_protection()
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
        # Fallback for requests < 2.32 (below the pinned floor).
        if select_proxy(url, proxies):
            if not self._allow_non_global_addresses:
                _warn_proxy_bypasses_ssrf_protection()
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


def _build_sync_session(allow_non_global_addresses: bool) -> requests.Session:
    session = requests.Session()
    if allow_non_global_addresses:
        _warn_non_global_allowed()
        return session
    adapter = SSRFProtectedHTTPAdapter(allow_non_global_addresses=False)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_url_bytes(url: str, request_timeout: Optional[float] = None) -> bytes:
    """Validate ``url`` and fetch its bytes with SSRF protections applied,
    honouring the SDK config flags. This is the sync entry point used by the
    loaders.
    """
    prepared_url = validate_url_destination(url)
    session = _build_sync_session(config.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES)
    try:
        if config.VALIDATE_IMAGE_URL_REDIRECTS:
            return _fetch_validating_redirects_sync(
                session=session, url=prepared_url, request_timeout=request_timeout
            )
        return _fetch_legacy_sync(
            session=session, url=prepared_url, request_timeout=request_timeout
        )
    except HTTPError:
        # HTTP-status errors keep flowing so wrap_errors maps them to
        # HTTPCallErrorError (status code + api message preserved).
        raise
    except HTTPClientError:
        # Our own validation / SSRF / redirect errors are already the right type.
        raise
    except requests.exceptions.RequestException as error:
        # ConnectionError, TooManyRedirects, Timeout, ... -> uniform SDK error.
        raise HTTPClientError(
            f"Could not load image from URL. Details: "
            f"{deduct_api_key_from_string(str(error))}"
        ) from error
    finally:
        session.close()


def _fetch_legacy_sync(
    session: requests.Session, url: str, request_timeout: Optional[float]
) -> bytes:
    _warn_legacy_redirect_handling()
    session.max_redirects = config.MAX_IMAGE_URL_REDIRECTS
    response = session.get(
        url, stream=True, allow_redirects=True, timeout=request_timeout
    )
    api_key_safe_raise_for_status(response=response)
    return response.content


def _fetch_validating_redirects_sync(
    session: requests.Session, url: str, request_timeout: Optional[float]
) -> bytes:
    current_url = url
    for _ in range(config.MAX_IMAGE_URL_REDIRECTS + 1):
        response = session.get(
            current_url, stream=True, allow_redirects=False, timeout=request_timeout
        )
        if response.is_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise HTTPClientError("Redirect response missing Location header.")
            next_url = urllib.parse.urljoin(current_url, location)
            current_url = validate_url_destination(next_url)
            continue
        api_key_safe_raise_for_status(response=response)
        return response.content
    raise requests.exceptions.TooManyRedirects(
        f"Exceeded maximum of {config.MAX_IMAGE_URL_REDIRECTS} redirects."
    )


# ---------------------------------------------------------------------------
# Async path (aiohttp) — validating resolver + optional manual redirect loop.
# ---------------------------------------------------------------------------
class ValidatingResolver(AbstractResolver):
    """aiohttp resolver that rejects non-global resolved addresses. Because
    aiohttp connects to the addresses the resolver returns, filtering here both
    validates and pins the connection to a checked IP on every hop.
    """

    def __init__(self, allow_non_global_addresses: bool):
        self._allow_non_global_addresses = allow_non_global_addresses
        self._delegate = aiohttp.ThreadedResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        hosts = await self._delegate.resolve(host, port, family)
        if self._allow_non_global_addresses:
            return hosts
        for entry in hosts:
            if not address_is_global(entry["host"]):
                raise URLAddressNotAllowedError(
                    f"Host '{host}' resolves to non-global address '{entry['host']}'."
                )
        return hosts

    async def close(self) -> None:
        await self._delegate.close()


def _build_async_connector(allow_non_global_addresses: bool) -> aiohttp.TCPConnector:
    if allow_non_global_addresses:
        _warn_non_global_allowed()
        return aiohttp.TCPConnector()
    return aiohttp.TCPConnector(
        resolver=ValidatingResolver(allow_non_global_addresses=False)
    )


def _ensure_ip_literal_allowed(url: str) -> None:
    """aiohttp skips its resolver for IP-literal hosts, so a literal non-global
    target (e.g. the cloud metadata IP ``169.254.169.254``) would otherwise
    bypass :class:`ValidatingResolver`. Validate literals explicitly before the
    request is issued (the resolver still covers hostname targets).
    """
    if config.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES:
        return
    parsed = urllib3.util.parse_url(url)
    host = parsed.host
    if host and _host_is_ip_literal(host):
        literal = _strip_ipv6_brackets(host)
        if not address_is_global(literal):
            raise URLAddressNotAllowedError(
                f"URL points to non-global address '{literal}'."
            )


async def fetch_url_bytes_async(
    url: str, request_timeout: Optional[float] = None
) -> bytes:
    """Async counterpart of :func:`fetch_url_bytes`."""
    prepared_url = validate_url_destination(url)
    _ensure_ip_literal_allowed(prepared_url)
    connector = _build_async_connector(config.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES)
    # Leave aiohttp's default ClientTimeout (total=300s) in place unless the
    # caller sets one; passing an empty ClientTimeout() would disable it.
    session_kwargs = {"connector": connector}
    if request_timeout is not None:
        session_kwargs["timeout"] = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(**session_kwargs) as session:
        try:
            if config.VALIDATE_IMAGE_URL_REDIRECTS:
                return await _fetch_manual_redirects_async(
                    session, prepared_url, revalidate_string_policy=True
                )
            if not config.ALLOW_URL_TO_NON_GLOBAL_ADDRESSES:
                # Non-global blocking is on: follow redirects manually so every
                # hop (including IP literals, which aiohttp's resolver skips) is
                # checked -- parity with the sync per-hop adapter.
                _warn_legacy_redirect_handling()
                return await _fetch_manual_redirects_async(
                    session, prepared_url, revalidate_string_policy=False
                )
            return await _fetch_legacy_async(session=session, url=prepared_url)
        except ClientResponseError:
            # HTTP-status errors keep flowing so wrap_errors_async maps them to
            # HTTPCallErrorError (status code + api message preserved).
            raise
        except HTTPClientError:
            # Our own validation / SSRF / redirect errors are already correct.
            raise
        except aiohttp.ClientError as error:
            # TooManyRedirects, connection errors, ... -> uniform SDK error.
            raise HTTPClientError(
                f"Could not load image from URL. Details: "
                f"{deduct_api_key_from_string(str(error))}"
            ) from error


async def _fetch_legacy_async(session: aiohttp.ClientSession, url: str) -> bytes:
    _warn_legacy_redirect_handling()
    async with session.get(
        url, allow_redirects=True, max_redirects=config.MAX_IMAGE_URL_REDIRECTS
    ) as response:
        response.raise_for_status()
        return await response.read()


async def _fetch_manual_redirects_async(
    session: aiohttp.ClientSession, url: str, revalidate_string_policy: bool
) -> bytes:
    """Follow redirects one hop at a time. Every hop gets the IP-literal
    non-global check; when ``revalidate_string_policy`` is True the full
    URL-string policy (scheme / FQDN / allow-block) is re-applied too.
    """
    current_url = url
    for _ in range(config.MAX_IMAGE_URL_REDIRECTS + 1):
        async with session.get(current_url, allow_redirects=False) as response:
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise HTTPClientError("Redirect response missing Location header.")
                next_url = urllib.parse.urljoin(current_url, location)
                if revalidate_string_policy:
                    next_url = validate_url_destination(next_url)
                _ensure_ip_literal_allowed(next_url)
                current_url = next_url
                continue
            response.raise_for_status()
            return await response.read()
    raise HTTPClientError(
        f"Exceeded maximum of {config.MAX_IMAGE_URL_REDIRECTS} redirects."
    )
