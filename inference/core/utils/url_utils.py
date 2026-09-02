import urllib.parse
from typing import Optional

from inference.core.env import SECURE_GATEWAY


def get_secure_gateway_base_url() -> Optional[str]:
    """Base URL of the configured secure gateway, or None when not configured.

    The secure gateway serves TLS on 443 by default, so SECURE_GATEWAY may be
    scheme-qualified (https://gateway.local). Bare host[:port] values keep the
    historical http:// behaviour for legacy license servers. A trailing slash
    is stripped so callers can append paths directly.
    """
    if not SECURE_GATEWAY:
        return None
    gateway = SECURE_GATEWAY.rstrip("/")
    if "://" in gateway:
        return gateway
    return f"http://{gateway}"


def wrap_url(url: str) -> str:
    gateway_base = get_secure_gateway_base_url()
    if gateway_base is None:
        return url
    gateway_prefix = f"{gateway_base}/proxy?url="
    # Idempotent: values may already be wrapped (e.g. env overrides configured
    # with a gateway URL) - wrapping twice would proxy the proxy.
    if url.startswith(gateway_prefix):
        return url
    return gateway_prefix + urllib.parse.quote(url, safe="~()*!'")
