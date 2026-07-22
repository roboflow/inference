from urllib.parse import urlparse, urlunparse


def sanitize_source_reference(ref: str) -> str:
    """Strip credentials and query parameters from URLs for observability surfaces."""
    parsed = urlparse(ref)
    if parsed.scheme and parsed.hostname:
        netloc = parsed.hostname + (f":{parsed.port}" if parsed.port else "")
        sanitized = parsed._replace(netloc=netloc, query="", fragment="")
        return urlunparse(sanitized)
    return ref
