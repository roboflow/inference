import urllib

from inference.core.env import SECURE_GATEWAY


def wrap_url(url: str) -> str:
    if not SECURE_GATEWAY:
        return url
    # The secure gateway serves TLS on 443 by default, so SECURE_GATEWAY may
    # be scheme-qualified (https://gateway.local). Bare host[:port] values
    # keep the historical http:// behaviour for legacy license servers.
    if "://" in SECURE_GATEWAY:
        gateway_base = SECURE_GATEWAY.rstrip("/")
    else:
        gateway_base = f"http://{SECURE_GATEWAY}"
    gateway_prefix = f"{gateway_base}/proxy?url="
    # Idempotent: values may already be wrapped (e.g. env overrides configured
    # with a gateway URL) - wrapping twice would proxy the proxy.
    if url.startswith(gateway_prefix):
        return url
    return gateway_prefix + urllib.parse.quote(url, safe="~()*!'")
