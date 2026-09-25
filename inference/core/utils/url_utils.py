import urllib.parse
from typing import Optional

from inference.core.env import SECURE_GATEWAY
from inference.core.utils.secure_gateway import validate_secure_gateway_url


def get_secure_gateway_base_url() -> Optional[str]:
    """Return the TLS gateway base; explicit loopback HTTP supports local tunnels."""
    if not SECURE_GATEWAY:
        return None
    return validate_secure_gateway_url(SECURE_GATEWAY)


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
